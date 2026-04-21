"""
OllamaService — GDPR-compliant, context-compaction-aware
=========================================================
Generic Ollama-backed chat service with MedGemma as the default model.
"""

import asyncio
import ollama
from dotenv import load_dotenv
from app.services.llm_base import BaseLLMService, GDPR_SYSTEM_SUFFIX

load_dotenv()


class OllamaService(BaseLLMService):
    """Service for local Ollama models (defaults to MedGemma)."""

    def __init__(self, model_name: str = "MedAIBase/MedGemma1.5:4b"):
        self.model_name = model_name

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

        emr_section, emr_fields_used = self._build_emr_section(
            system_prompt=system_prompt,
            emr_consent=emr_consent,
            patient_id=patient_id,
            consent_prefix="PATIENT MEDICAL RECORDS (consented, read-only):\n",
        )

        was_compacted, new_compacted_summary, history_for_llm = self._run_deterministic_compaction(
            history,
            compacted_summary,
            label="Ollama model",
        )

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

        sanitized_system_content, sanitized_history_for_llm, sanitized_message = self._anonymize_for_llm(
            system_content=system_content,
            history_for_llm=history_for_llm,
            message=message,
            presidio_analyzer=presidio_analyzer,
            presidio_anonymizer=presidio_anonymizer,
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