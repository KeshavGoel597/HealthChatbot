"""
Medical term extraction using Qwen LLM.

Given a natural language patient query, extracts normalized
medical terms (symptoms, conditions, treatments) as a list of strings.

E.g. "I have a headache, what should I do?" → ["headache"]

Loaded once at startup, queried per request.  Produces cleaner inputs
for SapBERT CUI search than raw conversational text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

@dataclass
class ExtractionResult:
    """Structured output from the term extractor."""
    intent: str               # "broad", "specific", or "mixed"
    categories: list[str]     # EMR category strings for broad/mixed queries
    terms: list[str]          # Clinical terms for specific/mixed CUI search


_VALID_CATEGORIES: frozenset[str] = frozenset({
    "diagnosis", "symptom", "comorbidity", "lab", "vitals",
    "medicine", "history", "comment", "recommended_labs",
    "demographics", "discharge",
})


def _parse_extraction_response(response: str, fallback_query: str) -> ExtractionResult:
    """Parse Qwen model JSON output into ExtractionResult.

    Validates category strings against _VALID_CATEGORIES so the pipeline
    never receives a category name that does not exist in the EMR parser.
    """
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            intent = parsed.get("intent", "specific")
            if intent not in ("broad", "specific", "mixed"):
                intent = "specific"
            raw_categories = parsed.get("categories", [])
            categories = [
                c for c in raw_categories
                if isinstance(c, str) and c.strip() in _VALID_CATEGORIES
            ]
            raw_terms = parsed.get("terms", [])
            terms = [s for s in raw_terms if isinstance(s, str) and s.strip()]
            return ExtractionResult(intent=intent, categories=categories, terms=terms)
        elif isinstance(parsed, list) and parsed and all(isinstance(t, str) for t in parsed):
            return ExtractionResult(
                intent="specific",
                categories=[],
                terms=[t for t in parsed if t.strip()],
            )
    except json.JSONDecodeError:
        pass
    return ExtractionResult(intent="specific", categories=[], terms=[fallback_query])


_DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

_SYSTEM_MSG = (
    "You extract medical retrieval intent and terms. "
    "Return only valid JSON."
)

_FEW_SHOT = """Extract medical information for EMR retrieval.

Return exactly one JSON object with:
- "intent": one of "broad", "specific", "mixed"
- "categories": a JSON list — use only strings from this fixed set:
  diagnosis, symptom, comorbidity, lab, vitals, medicine,
  history, comment, recommended_labs, demographics, discharge
- "terms": a JSON list of normalized clinical term strings

Rules:
- "broad" = user wants whole record categories, not a specific finding.
  Set categories to the relevant EMR sections. Leave terms empty.
- "specific" = user asks about a symptom, condition, medication, or lab value.
  Set terms. Leave categories empty.
- "mixed" = both a broad category and a specific clinical detail are present.
  Set both categories and terms.
- Remove negated terms. Negation applies only to the phrase it directly modifies.
- Normalize colloquial phrases to standard medical terms.
- Keep useful details like body part or laterality when relevant.
- Do not add symptoms that are not mentioned.
- Do not output explanations or markdown.

Examples:

Text: "I don't have body pain"
Output: {"intent":"specific","categories":[],"terms":[]}

Text: "I don't have eye pain, but I do have finger pain"
Output: {"intent":"specific","categories":[],"terms":["finger pain"]}

Text: "No fever, but I have a cough"
Output: {"intent":"specific","categories":[],"terms":["cough"]}

Text: "List all my medications"
Output: {"intent":"broad","categories":["medicine"],"terms":[]}

Text: "Tell me all my diagnoses"
Output: {"intent":"broad","categories":["diagnosis","comorbidity"],"terms":[]}

Text: "Show my full medical history"
Output: {"intent":"broad","categories":["diagnosis","symptom","comorbidity","medicine","lab","vitals","history"],"terms":[]}

Text: "What are my latest lab results?"
Output: {"intent":"broad","categories":["lab"],"terms":[]}

Text: "What are my allergies and conditions?"
Output: {"intent":"broad","categories":["comorbidity"],"terms":[]}

Text: "Show my diabetes medications"
Output: {"intent":"mixed","categories":["medicine"],"terms":["diabetes"]}

Text: "What lab results relate to my kidney disease?"
Output: {"intent":"mixed","categories":["lab"],"terms":["kidney disease","creatinine"]}

Text: "Why is my HbA1c high?"
Output: {"intent":"specific","categories":[],"terms":["high HbA1c"]}
"""


class TermExtractor:
    """Extracts normalized medical terms from natural language queries.

    Load once at startup (heavy model load), then call ``extract()``
    per request — inference is fast.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        cuda = torch.cuda.is_available()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if cuda else "cpu",
            torch_dtype=torch.float16 if cuda else torch.float32,
        )
        print(f"[TermExtractor] Loaded {model_name} on {'GPU' if cuda else 'CPU'}")

    def extract(self, query: str) -> ExtractionResult:
        """Extract EMR categories and clinical terms from a natural language query.

        Returns:
            ExtractionResult with intent, categories (EMR category strings for
            broad/mixed queries), and terms (clinical terms for CUI search).
            Falls back to ExtractionResult(intent="specific", categories=[],
            terms=[query]) on parse failure.
        """
        prompt = _FEW_SHOT + f'Text: "{query}"\nOutput:'
        messages = [
            {"role": "system", "content": _SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
        )

        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Model sometimes wraps output in ```json ... ``` fences — strip them
        if response.startswith("```"):
            lines = response.splitlines()
            inner = [l for l in lines[1:] if l.strip() != "```"]
            response = "\n".join(inner).strip()

        return _parse_extraction_response(response, query)
