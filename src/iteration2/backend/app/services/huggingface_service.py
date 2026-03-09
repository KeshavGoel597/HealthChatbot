"""
HuggingFaceService — GDPR-compliant, context-compaction-aware
=============================================================
Uses deterministic compaction (no extra LLM call) since Qwen-0.5B is too
small to reliably self-summarise clinical conversations.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import asyncio
from dotenv import load_dotenv
import re

from app.services.context_compaction import (
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    compact_deterministic,
)

load_dotenv()

GDPR_SYSTEM_SUFFIX = (
    "\n\nCRITICAL RULES:\n"
    "1. You are an AI assistant, NOT a licensed physician. Always say this.\n"
    "2. End EVERY response with: 'I am Robert, an AI assistant. Please verify all "
    "medical information with a licensed healthcare professional.'\n"
    "3. NEVER diagnose, prescribe, or recommend treatment changes.\n"
    "4. If the patient reports incorrect records, tell them to contact their healthcare "
    "provider — you cannot modify medical records.\n"
)


class HuggingFaceService:
    def __init__(self):
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = None
        self.model = None

    def _summarize_emr_context(self, raw_data: str) -> tuple[str, list]:
        """
        Extracts key clinical information from the raw EMR string.
        Returns (summary_text, list_of_field_names_used) for GDPR Art. 15.
        """
        summary = []
        fields_used = []

        age_match = re.search(r'age: "([^"]+)"', raw_data)
        sex_match = re.search(r'sex: "([^"]+)"', raw_data)
        if age_match or sex_match:
            age = age_match.group(1) if age_match else "?"
            sex = sex_match.group(1) if sex_match else "?"
            summary.append(f"PATIENT: Age {age}, Sex {sex}")
            fields_used.append("Patient Demographics")

        diagnoses = set(re.findall(r'"diag" => "([^"]+)"', raw_data))
        cleaned_diagnoses = [d.strip() for d in diagnoses if d.strip() and d.strip() != "@10"]
        if cleaned_diagnoses:
            summary.append(f"DIAGNOSES: {', '.join(cleaned_diagnoses)}")
            fields_used.append("Medical Diagnoses")

        symptoms = set(re.findall(r'"sym" => "([^"]+)"', raw_data))
        cleaned_symptoms = [s.strip() for s in symptoms if s.strip() and s.strip() != "FCU"]
        if cleaned_symptoms:
            summary.append(f"SYMPTOMS: {', '.join(cleaned_symptoms)}")
            fields_used.append("Recorded Symptoms")

        meds = set(re.findall(r'"medicine" => "([^"]+)"', raw_data))
        if meds:
            summary.append(f"MEDICATIONS: {', '.join(list(meds)[:10])}...")
            fields_used.append("Prescribed Medications")

        lab_summary = []
        for lab in ["Hemoglobin", "RBS", "Total WBC Count", "Platelet Count"]:
            matches = re.findall(
                f'"name" => "{lab}", "value" => "([^"]+)", "date" => "([^"]+)"',
                raw_data,
            )
            if matches:
                last_val, last_date = matches[-1]
                lab_summary.append(f"{lab}: {last_val} ({last_date})")
        if lab_summary:
            summary.append(f"RECENT LABS: {', '.join(lab_summary)}")
            fields_used.append("Laboratory Results")

        return "\n".join(summary), fields_used

    def _load_model(self):
        if self.model is None:
            print(f"Loading {self.model_name}...")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                print(f"{self.model_name} loaded.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name}'. "
                    f"Ensure you are authenticated with Hugging Face. Error: {str(e)}"
                )

    def get_patient_data(self, patient_id: str) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", f"{patient_id}.json")

        if not os.path.exists(data_path):
            return "{}"

        try:
            with open(data_path, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading patient data: {e}")
            return "{}"

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self._load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    async def chat(
        self,
        message: str,
        patient_id: str = "patient101",
        history: list = None,
        compacted_summary: str = None,
        emr_consent: bool = False,
    ) -> dict:
        if history is None:
            history = []

        self._load_model()

        emr_fields_used = []
        clinical_summary = ""

        # GDPR Art. 5(1)(a,c) — only load EMR if patient has consented
        if emr_consent:
            raw_data = self.get_patient_data(patient_id)
            clinical_summary, emr_fields_used = self._summarize_emr_context(raw_data)
            emr_section = f"PATIENT SUMMARY (consented, read-only):\n{clinical_summary}"
        else:
            emr_section = (
                "EMR ACCESS: Patient has NOT consented to EMR access. "
                "Do NOT reference any specific medical records. "
                "Provide only general medical guidance."
            )

        # --- Deterministic compaction (GDPR Art. 5(1)(c)) ---
        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print(f"[COMPACTION] Running deterministic compaction for HF model...")
            old_turns, _ = split_for_compaction(history)
            new_summary = compact_deterministic(old_turns)
            # Append fresh summary to any existing summary
            if compacted_summary:
                new_compacted_summary = compacted_summary + "\n\n" + new_summary
            else:
                new_compacted_summary = new_summary
            was_compacted = True
            print(f"[COMPACTION] Done.")

        # Build windowed history
        history_for_llm = build_llm_history(history, new_compacted_summary)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Robert, a helpful AI medical assistant for patients.\n"
                    "You are NOT a licensed physician.\n"
                    f"{emr_section}\n"
                    "Be empathetic, professional, and calm.\n"
                    "NEVER congratulate a user on symptoms or illness.\n"
                    "If the user shares negative symptoms, acknowledge them with concern.\n"
                    "If the user mentions incorrect records, tell them to contact their "
                    "healthcare provider — you cannot modify medical records.\n"
                    f"{GDPR_SYSTEM_SUFFIX}"
                ),
            }
        ]

        for msg in history_for_llm:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": message})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, prompt)

        input_tokens = len(self.tokenizer.encode(prompt))
        output_tokens = len(self.tokenizer.encode(response_text))

        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model_name": self.model_name,
            "emr_fields_used": emr_fields_used,
            "was_compacted": was_compacted,
            "new_compacted_summary": new_compacted_summary,
        }
