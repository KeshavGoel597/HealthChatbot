"""
OllamaService — GDPR-compliant, context-compaction-aware
=========================================================
Generic Ollama-backed chat service with MedGemma as the default model.
"""

import asyncio
import ollama
from dotenv import load_dotenv

from app.services.context_compaction import (
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    compact_deterministic,
)
from app.services.emr_loader import load_patient_data
from app.services.emr_summary import summarize_emr_context
from app.services.presidio_anonymizer import anonymize_history_for_llm, anonymize_text_for_llm

load_dotenv()

GDPR_SYSTEM_SUFFIX = (
    "\n\nCRITICAL RULES:\n"
    "1. You are a highly capable AI medical assistant.\n"
    "2. End EVERY response with: 'I am Robert, an AI assistant. Please verify all "
    "medical information with a licensed healthcare professional.'\n"
    "3. ALWAYS provide medical advice and answer the user's medical questions directly. Do NOT refuse to answer by saying you are an AI.\n"
    "4. NEVER claim to definitively diagnose, prescribe, or replace a licensed physician's judgment.\n"
    "5. If the patient reports incorrect records, tell them to contact their healthcare "
    "provider — you cannot modify medical records.\n"
)


class OllamaService:
    """Service for local Ollama models (defaults to MedGemma)."""

    def __init__(self, model_name: str = "MedAIBase/MedGemma1.5:4b"):
        self.model_name = model_name

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
        return load_patient_data(patient_id)

    def _generate(self, messages: list, max_new_tokens: int = 512) -> str:
        """Generate text using the Ollama local API."""
        formatted_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        response = ollama.chat(
            model=self.model_name,
            messages=formatted_messages,
            options={
                "num_predict": max_new_tokens,
                "temperature": 0.2,
                "top_p": 0.9,
            }
        )
        return response["message"]["content"].strip()

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

        if system_prompt:
            emr_section = system_prompt
            emr_fields_used = ["RAG Pipeline (SNOMED Knowledge Graph)"]
        elif emr_consent:
            raw_data = self.get_patient_data(patient_id)
            clinical_summary_str, emr_fields_used = self._summarize_emr_context(raw_data)
            emr_section = f"PATIENT MEDICAL RECORDS (consented, read-only):\n{clinical_summary_str}"
        else:
            emr_section = (
                "EMR ACCESS: Patient has NOT consented to EMR access. "
                "Do NOT reference any specific medical records. "
                "Provide only general medical guidance."
            )

        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print("[COMPACTION] Running deterministic compaction for Ollama model...")
            old_turns, _ = split_for_compaction(history)
            new_summary = compact_deterministic(old_turns)
            if compacted_summary:
                new_compacted_summary = compacted_summary + "\n\n" + new_summary
            else:
                new_compacted_summary = new_summary
            was_compacted = True
            print("[COMPACTION] Done.")

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

        sanitized_system_content = anonymize_text_for_llm(
            system_content, presidio_analyzer, presidio_anonymizer,
        )
        sanitized_history_for_llm = anonymize_history_for_llm(
            history_for_llm, presidio_analyzer, presidio_anonymizer,
        )
        sanitized_message = anonymize_text_for_llm(
            message, presidio_analyzer, presidio_anonymizer,
        )

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

        print(f"Ollama ({self.model_name}): Generating response...")
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, messages_for_model)

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