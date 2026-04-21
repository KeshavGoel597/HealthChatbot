from app.services.context_compaction import (
    build_llm_history,
    compact_deterministic,
    needs_compaction,
    split_for_compaction,
)
from app.services.emr_loader import load_patient_data
from app.services.emr_summary import summarize_emr_context
from app.services.presidio_anonymizer import (
    anonymize_history_for_llm,
    anonymize_text_for_llm,
)

GDPR_SYSTEM_SUFFIX = (
    "\n\nCRITICAL RULES:\n"
    "1. You are a highly capable AI medical assistant.\n"
    "2. End EVERY response with: 'I am Robert, an AI assistant. Please verify all "
    "medical information with a licensed healthcare professional.'\n"
    "3. ALWAYS provide medical advice and answer the user's medical questions directly. Do NOT refuse to answer by saying you are an AI.\n"
    "4. NEVER claim to definitively diagnose, prescribe, or replace a licensed physician's judgment.\n"
    "5. If the patient reports incorrect records, tell them to contact their healthcare "
    "provider - you cannot modify medical records.\n"
)


class BaseLLMService:
    """Shared helper methods for provider-specific LLM services."""

    def get_patient_data(self, patient_id: str) -> str:
        return load_patient_data(patient_id)

    def _summarize_emr_context(self, raw_data: str) -> tuple[str, list]:
        return summarize_emr_context(raw_data)

    @staticmethod
    def _no_emr_access_section() -> str:
        return (
            "EMR ACCESS: Patient has NOT consented to EMR access. "
            "Do NOT reference any specific medical records. "
            "Provide only general medical guidance."
        )

    def _build_emr_section(
        self,
        *,
        system_prompt: str,
        emr_consent: bool,
        patient_id: str,
        consent_prefix: str,
    ) -> tuple[str, list[str]]:
        if system_prompt:
            return system_prompt, ["RAG Pipeline (SNOMED Knowledge Graph)"]

        if emr_consent:
            raw_data = self.get_patient_data(patient_id)
            clinical_summary, emr_fields_used = self._summarize_emr_context(raw_data)
            return f"{consent_prefix}{clinical_summary}", emr_fields_used

        return self._no_emr_access_section(), []

    def _run_deterministic_compaction(
        self,
        history: list,
        compacted_summary: str | None,
        *,
        label: str,
    ) -> tuple[bool, str | None, list]:
        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print(f"[COMPACTION] Running deterministic compaction for {label}...")
            old_turns, _ = split_for_compaction(history)
            new_summary = compact_deterministic(old_turns)
            if compacted_summary:
                new_compacted_summary = compacted_summary + "\n\n" + new_summary
            else:
                new_compacted_summary = new_summary
            was_compacted = True
            print("[COMPACTION] Done.")

        history_for_llm = build_llm_history(history, new_compacted_summary)
        return was_compacted, new_compacted_summary, history_for_llm

    def _anonymize_for_llm(
        self,
        *,
        system_content: str,
        history_for_llm: list,
        message: str,
        presidio_analyzer,
        presidio_anonymizer,
    ) -> tuple[str, list, str]:
        sanitized_system_content = anonymize_text_for_llm(
            system_content,
            presidio_analyzer,
            presidio_anonymizer,
        )
        sanitized_history_for_llm = anonymize_history_for_llm(
            history_for_llm,
            presidio_analyzer,
            presidio_anonymizer,
        )
        sanitized_message = anonymize_text_for_llm(
            message,
            presidio_analyzer,
            presidio_anonymizer,
        )
        return sanitized_system_content, sanitized_history_for_llm, sanitized_message
