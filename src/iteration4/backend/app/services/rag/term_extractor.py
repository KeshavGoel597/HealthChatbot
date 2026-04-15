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

_DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

_SYSTEM_MSG = (
    "Extract medical retrieval keywords. "
    "Output only a valid JSON list of strings."
)

_FEW_SHOT = """Extract medical keywords for retrieval.

Rules:
- Output only a JSON list of strings.
- Include symptoms, diagnoses, comorbidities, medications, labs, and history terms when relevant.
- Normalize to standard medical terms; map colloquial phrases to clinical terms.
- Ignore negated symptoms.
- Keep useful detail (body part, laterality, pain quality, duration) when present.
- If the query is about records/history/chart, prefer broad record categories over specific symptoms.

Text: "I have the runs and both of my knees are swollen."
Output: ["diarrhea", "swollen knees"]

Text: "My stomach hurts but I don't have any nausea or vomiting."
Output: ["stomach ache"]

Text: "Tell me about my medical history and records."
Output: ["diagnosis", "comorbidity", "medications", "lab results", "symptoms", "patient history", "vitals", "discharge summary"]
"""


class TermExtractor:
    """Extracts normalized medical terms from natural language queries.

    Load once at startup (heavy model load), then call ``extract()``
    per request — inference is fast.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print(f"[TermExtractor] Loaded {model_name}")

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
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
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
