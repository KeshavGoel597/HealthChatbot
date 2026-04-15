
"""
MedGemmaService — GDPR-compliant, context-compaction-aware
===========================================================
Refactored to use local Ollama inference for Mac hardware acceleration.
"""

import os
import asyncio
import ollama
from dotenv import load_dotenv

from app.services.context_compaction import (
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    compact_deterministic,
)
from app.services.emr_summary import summarize_emr_context
from app.services.presidio_anonymizer import anonymize_history_for_llm, anonymize_text_for_llm

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
    """Service for Google MedGemma 4B medical model running via Ollama."""

    def __init__(self):
        # This matches the tag you pulled: MedAIBase/MedGemma1.5:4b
        self.model_name = "MedAIBase/MedGemma1.5:4b"

    def _summarize_emr_context(self, raw_data: str) -> tuple[str, list]:
        return summarize_emr_context(raw_data)

    def _load_model(self):
        """Checks if the Ollama service is reachable."""
        try:
            ollama.list()
            print(f"Ollama is active. Using model: {self.model_name}")
        except Exception as e:
            print(f"Failed to connect to Ollama: {e}")
            raise RuntimeError(
                "Ollama service not detected. Please run 'brew services start ollama' "
                "or open the Ollama application on your Mac."
            )

    def get_patient_data(self, patient_id: str) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", f"{patient_id}.json")

        try:
            with open(data_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "{}"
        except Exception as e:
            print(f"Error reading patient data: {e}")
            return "{}"

    def _generate(self, messages: list, max_new_tokens: int = 512) -> str:
        """Generate text using the Ollama local API."""
        # Ollama expects content as a string, not a list of objects like HF Processor
        formatted_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=formatted_messages,
            options={
                "num_predict": max_new_tokens,
                "temperature": 0.2, # Low temperature for clinical accuracy
                "top_p": 0.9
            }
        )
        return response['message']['content'].strip()

    async def chat(
        self,
        message: str,
        patient_id: str = "patient101",
        history: list = None,
        compacted_summary: str = None,
        emr_consent: bool = False,
        *,
        system_prompt: str = "",
        presidio_analyzer=None,
        presidio_anonymizer=None,
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

        # Anonymization logic (GDPR compliance)
        sanitized_system_content = anonymize_text_for_llm(
            system_content, presidio_analyzer, presidio_anonymizer,
        )
        sanitized_history_for_llm = anonymize_history_for_llm(
            history_for_llm, presidio_analyzer, presidio_anonymizer,
        )
        sanitized_message = anonymize_text_for_llm(
            message, presidio_analyzer, presidio_anonymizer,
        )

        # Build message block for Ollama
        first_user_content = f"{sanitized_system_content}\n\n"

        if sanitized_history_for_llm:
            history_text = "\n".join(
                [f"{'Patient' if m['role'] == 'user' else 'Robert'}: {m['content']}"
                 for m in sanitized_history_for_llm]
            )
            first_user_content += f"[PRIOR CONVERSATION]\n{history_text}\n\n"

        first_user_content += f"Patient says: {sanitized_message}"

        messages_for_model = [
            {"role": "user", "content": first_user_content}
        ]

        print("MedGemma (Ollama): Generating response...")
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, messages_for_model)

        # Token counting (approximate for local model)
        input_tokens = len(first_user_content.split()) * 2
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