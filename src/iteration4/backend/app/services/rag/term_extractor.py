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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_DEFAULT_MODEL = "Qwen/Qwen3-1.7B"

_SYSTEM_MSG = (
    "You extract medical retrieval intent and terms. "
    "Return only valid JSON."
)

_FEW_SHOT = """Extract medical information for EMR retrieval.

Return exactly one JSON object with:
- "intent": one of "broad", "specific", "mixed"
- "terms": a JSON list of strings

Rules:
- "broad" = user wants whole categories or records, not a specific symptom.
  Use category terms such as: medications, diagnoses, comorbidities, labs, vitals, history, notes, allergies, procedures.
- "specific" = user asks about a symptom, condition, medication, or lab value.
- "mixed" = both broad category intent and specific clinical detail are present.
- Remove negated terms.
- Negation applies only to the phrase it directly modifies.
- Normalize colloquial phrases to standard medical terms.
- Keep useful details like body part or laterality when relevant.
- Do not add symptoms that are not mentioned.
- Do not output explanations or markdown.

Examples:

Text: "I don't have body pain"
Output: {"intent":"specific","terms":[]}

Text: "I don't have eye pain, but I do have finger pain"
Output: {"intent":"specific","terms":["finger pain"]}

Text: "No fever, but I have a cough"
Output: {"intent":"specific","terms":["cough"]}

Text: "Show my full medical history"
Output: {"intent":"broad","terms":["history"]}

Text: "List all my medications"
Output: {"intent":"broad","terms":["medications"]}

Text: "Tell me all my diagnoses"
Output: {"intent":"broad","terms":["diagnoses","comorbidities"]}

Text: "Show my diabetes medications"
Output: {"intent":"mixed","terms":["medications","diabetes"]}

Text: "Why is my HbA1c high?"
Output: {"intent":"specific","terms":["high HbA1c"]}
"""


class TermExtractor:
    """Extracts normalized medical terms from natural language queries.

    Load once at startup (heavy model load), then call ``extract()``
    per request — inference is fast.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        device_map = "auto" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        device_label = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"[TermExtractor] Loaded {model_name} on {device_label}")

    def extract(self, query: str) -> list[str]:
        """Extract medical terms from a natural language query.

        Args:
            query: Patient's natural language input
                   (e.g. "I have a headache, what should I do?").

        Returns:
            List of normalized medical term strings
            (e.g. ["headache"]).
            Falls back to [query] if extraction fails.
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

        # Strip input tokens from output
        gen_ids = output_ids[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        # Model sometimes wraps output in ```json ... ``` fences — strip them
        if response.startswith("```"):
            lines = response.splitlines()
            # Remove first line (```json) and last line (```)
            inner = [l for l in lines[1:] if l.strip() != "```"]
            response = "\n".join(inner).strip()

        try:
            parsed = json.loads(response)
            # Model sometimes returns a dict (e.g. {"medical_symptoms": [...]})
            # instead of a flat list — extract all string values from it
            if isinstance(parsed, dict):
                terms = []
                for v in parsed.values():
                    if isinstance(v, list):
                        terms.extend(s for s in v if isinstance(s, str) and s.strip())
                if terms:
                    return terms
            elif isinstance(parsed, list) and parsed and all(isinstance(t, str) for t in parsed):
                return [t for t in parsed if t.strip()]
        except json.JSONDecodeError:
            pass

        # Fallback: use the raw query as a single term
        return [query]
