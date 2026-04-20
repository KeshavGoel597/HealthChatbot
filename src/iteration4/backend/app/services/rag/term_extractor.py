"""
Medical term extraction using two-pass LLM inference.

Pass 1: Extract EMR category strings from a natural language query.
Pass 2: Extract normalized clinical terms from the same query.
Intent is then derived from the array lengths in Python.

E.g. "I have a headache" → ExtractionResult(intent="specific", categories=[], terms=["headache"])
     "Show my medications" → ExtractionResult(intent="broad", categories=["medicine"], terms=[])

Loaded once at startup, queried per request.
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


def _parse_categories(response: str) -> list[str]:
    """Parse pass-1 JSON: {"categories": [...]}."""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            raw = parsed.get("categories", [])
            return [c for c in raw if isinstance(c, str) and c.strip() in _VALID_CATEGORIES]
    except json.JSONDecodeError:
        pass
    return []


def _parse_terms(response: str) -> list[str]:
    """Parse pass-2 JSON: {"terms": [...]}."""
    try:
        parsed = json.loads(response)
        if isinstance(parsed, dict):
            raw = parsed.get("terms", [])
            return [t for t in raw if isinstance(t, str) and t.strip()]
    except json.JSONDecodeError:
        pass
    return []


def _compute_intent(categories: list[str], terms: list[str]) -> str:
    if categories and terms:
        return "mixed"
    if categories:
        return "broad"
    return "specific"


_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_CATEGORIES_PROMPT = """\
You are a headless clinical data extraction pipeline. Your sole function is to identify which EMR record sections a patient query is requesting. Output only a JSON object. Do not converse. Do not explain.

OUTPUT SCHEMA:
{{"categories": [array of strings]}}

ALLOWED VALUES (use only these exact strings):
diagnosis, symptom, comorbidity, lab, vitals, medicine, history, comment, recommended_labs, demographics, discharge

RULES:
- Add a category ONLY if the user is explicitly requesting that section (e.g. "medications" → "medicine", "lab results" → "lab").
- If the user asks about a specific condition or symptom without requesting a section, return an empty array.
- Negation: if the user says they do NOT want a section, exclude it.

EXAMPLES:
Input: "I have a headache"
Output: {{"categories": []}}

Input: "Show my medications"
Output: {{"categories": ["medicine"]}}

Input: "Diabetes medications"
Output: {{"categories": ["medicine"]}}

Input: "What are my lab results and current diagnoses?"
Output: {{"categories": ["lab", "diagnosis"]}}

Input: "I have left leg pain. Look at my current medications."
Output: {{"categories": ["medicine"]}}

Process the following input and output ONLY the JSON object.

Input: {query}
Output:"""

_TERMS_PROMPT = """\
You are a headless clinical data extraction pipeline. Your sole function is to extract specific clinical terms from a patient query. Output only a JSON object. Do not converse. Do not explain.

OUTPUT SCHEMA:
{{"terms": [array of strings]}}

RULES:
- Normalize: Convert colloquial phrases to standard clinical terms (e.g. "sugar" → "blood glucose").
- Detail Retention: Keep anatomical location and laterality attached (e.g. "left leg pain").
- Negation Filtration: REMOVE any term the user says they do NOT have (e.g. "no fever" → exclude "fever").
- Exclude: Do not include general section names (e.g. "medications", "lab results") — only specific conditions, symptoms, or findings.
- If no specific clinical terms are present, return an empty array.

EXAMPLES:
Input: "I have a headache"
Output: {{"terms": ["headache"]}}

Input: "Show my medications"
Output: {{"terms": []}}

Input: "Diabetes medications"
Output: {{"terms": ["diabetes"]}}

Input: "I don't have fever but have cough"
Output: {{"terms": ["cough"]}}

Input: "I have left leg pain. Look at my current medications."
Output: {{"terms": ["left leg pain"]}}

Process the following input and output ONLY the JSON object.

Input: {query}
Output:"""


class TermExtractor:
    """Extracts normalized medical terms from natural language queries via two-pass inference.

    Load once at startup (heavy model load), then call ``extract()`` per request.
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

    def _infer(self, prompt: str) -> str:
        """Run one inference pass, return decoded response string."""
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.1,
            top_k=50,
            repetition_penalty=1.05,
        )
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        if response.startswith("```"):
            lines = response.splitlines()
            response = "\n".join(l for l in lines[1:] if l.strip() != "```").strip()
        return response

    def extract(self, query: str) -> ExtractionResult:
        """Extract EMR categories and clinical terms via two focused inference passes.

        Falls back to terms=[query] if both passes return empty results.
        """
        categories = _parse_categories(self._infer(_CATEGORIES_PROMPT.format(query=query)))
        terms = _parse_terms(self._infer(_TERMS_PROMPT.format(query=query)))

        if not categories and not terms:
            terms = [query]

        return ExtractionResult(
            intent=_compute_intent(categories, terms),
            categories=categories,
            terms=terms,
        )
