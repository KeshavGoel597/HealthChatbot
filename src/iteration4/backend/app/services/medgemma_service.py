"""
MedGemmaService — GDPR-compliant, context-compaction-aware
===========================================================
Uses deterministic compaction (no extra model call) since we run on local hardware.
"""

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import asyncio
import re
from dotenv import load_dotenv

from app.services.context_compaction import (
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    compact_deterministic,
)
from app.services.emr_summary import summarize_emr_context

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


class MedGemmaService:
    """Service for Google MedGemma 4B medical model."""

    def __init__(self):
        self.model_name = "google/medgemma-1.5-4b-it"
        self.processor = None
        self.model = None

    def _summarize_emr_context(self, raw_data: str) -> tuple[str, list]:
        return summarize_emr_context(raw_data)

    def _load_model(self):
        if self.model is None:
            print(f"Loading MedGemma ({self.model_name})...")
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_name)
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16
                )
                print("MedGemma loaded successfully.")
            except Exception as e:
                print(f"Failed to load MedGemma: {e}")
                raise RuntimeError(
                    f"Failed to load MedGemma. Ensure you have HuggingFace access "
                    f"to this gated model (huggingface-cli login). Error: {e}"
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

    def _generate(self, messages: list, max_new_tokens: int = 512) -> str:
        """Generate text using MedGemma's processor and model."""
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    async def chat(
        self,
        message: str,
        patient_id: str = "patient101",
        history: list = None,
        compacted_summary: str = None,
        emr_consent: bool = False,
        *,
        system_prompt: str = "",
    ) -> dict:
        if history is None:
            history = []

        self._load_model()

        emr_fields_used = []
        clinical_summary_str = ""

        # --- Build EMR section ---
        if system_prompt:
            # RAG pipeline provided focused clinical context
            emr_section = system_prompt
            emr_fields_used = ["RAG Pipeline (SNOMED Knowledge Graph)"]
        elif emr_consent:
            # No RAG prompt but consent given — fall back to regex-based EMR summary
            raw_data = self.get_patient_data(patient_id)
            clinical_summary_str, emr_fields_used = self._summarize_emr_context(raw_data)
            emr_section = f"PATIENT MEDICAL RECORDS (consented, read-only):\n{clinical_summary_str}"
        else:
            emr_section = (
                "EMR ACCESS: Patient has NOT consented to EMR access. "
                "Do NOT reference any specific medical records. "
                "Provide only general medical guidance."
            )

        # --- Deterministic compaction ---
        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print(f"[COMPACTION] Running deterministic compaction for MedGemma...")
            old_turns, _ = split_for_compaction(history)
            new_summary = compact_deterministic(old_turns)
            if compacted_summary:
                new_compacted_summary = compacted_summary + "\n\n" + new_summary
            else:
                new_compacted_summary = new_summary
            was_compacted = True
            print(f"[COMPACTION] Done.")

        # Build windowed history
        history_for_llm = build_llm_history(history, new_compacted_summary)

        system_content = (
            "You are Robert, a warm and helpful AI medical assistant. You are NOT a licensed physician.\n\n"
            f"{emr_section}\n\n"
            "Your responsibilities:\n"
            "1. Directly address the patient's current question or complaint.\n"
            "2. Relate it to their medical background when relevant (and consented).\n"
            "3. If the symptom is new, provide helpful general medical guidance.\n"
            "4. Be empathetic, concise (2-3 paragraphs).\n"
            "5. If the patient mentions incorrect records, say: 'Please contact your healthcare "
            "provider or system administrator to correct your records — I cannot modify them.'\n"
            f"{GDPR_SYSTEM_SUFFIX}"
        )

        # MedGemma uses a specific message format
        messages_for_model = []
        # Inject system content into the first user message (MedGemma's format)
        first_user_content = f"{system_content}\n\n"

        # Prepend compacted history as a text block if present
        if history_for_llm:
            history_text = "\n".join(
                [f"{'Patient' if m['role'] == 'user' else 'Robert'}: {m['content']}"
                 for m in history_for_llm]
            )
            first_user_content += f"[PRIOR CONVERSATION]\n{history_text}\n\n"

        first_user_content += f"Patient says: {message}"

        messages_for_model = [
            {"role": "user", "content": [{"type": "text", "text": first_user_content}]}
        ]

        print("MedGemma: Generating response...")
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, messages_for_model)

        # Token counting (approximate for local model)
        input_text = system_content + message
        input_tokens = len(input_text.split()) * 2
        output_tokens = len(response_text.split()) * 2

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
