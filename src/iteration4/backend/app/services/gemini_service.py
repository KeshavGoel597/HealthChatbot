"""
GeminiService — GDPR-compliant, RAG-enhanced, context-compaction-aware
=======================================================================
GDPR Compliance:
  Art. 5(1)(a)  — AI disclaimer + physician verification in every response
  Art. 5(1)(c)  — EMR access only when patient has provided explicit consent
  Art. 15       — Returns emr_fields_used so the UI can show evidence panel
  Art. 22       — System prompt explicitly forbids autonomous medical decisions

RAG Integration:
  When a system_prompt is provided (from the RAG pipeline), it is used as the
  primary clinical context. Otherwise falls back to raw EMR extraction.
"""

import os
import json
import logging
from google import genai
from google.genai import types
from dotenv import load_dotenv
from app.services.context_compaction import (
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    compact_with_gemini,
)

load_dotenv()

logger = logging.getLogger(__name__)

# GDPR Art. 5(1)(a) + Art. 22 — mandatory disclaimer injected into every response prompt
GDPR_RESPONSE_DISCLAIMER = (
    "\n\nIMPORTANT — MANDATORY INSTRUCTION: At the end of every response you give, "
    "include a brief disclaimer such as: "
    "' I am Robert, an AI assistant. This information is for educational purposes only "
    "and does not constitute medical advice. Please consult a licensed healthcare "
    "professional before making any medical decisions.' "
    "Do NOT skip this disclaimer under any circumstances."
)

# GDPR Art. 16 + Art. 22 — EMR is read-only; corrections go through healthcare provider
GDPR_EMR_READONLY_NOTE = (
    "IMPORTANT: You are NOT authorised to modify, update, or correct any medical records. "
    "If the patient mentions incorrect information in their records, instruct them to "
    "contact their healthcare provider or a system administrator to request corrections. "
    "The chatbot only reads records — it never writes to them."
)


class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")

        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-lite"

    def get_patient_data(self, patient_id: str) -> str:
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        data_path = os.path.join(base_dir, "data", f"{patient_id}.json")

        try:
            with open(data_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            return "{}"
        except Exception as e:
            print(f"Error reading patient data: {e}")
            return "{}"

    async def extract_clinical_data(self, raw_text: str) -> tuple[str, list]:
        """
        Extracts structured clinical data from raw EMR text.
        Returns (structured_json_string, list_of_field_names_used).
        The field list is returned for GDPR Art. 15 evidence transparency.
        """
        prompt = f"""
        Act as a Clinical Coder.
        TASK: Extract Diagnosis, Medications, Labs, Symptoms from the text below.
        CRITICAL: Keep Dates. Map to SNOMED IDs where possible.
        RAW DATA: {raw_text}
        OUTPUT: Valid JSON List. Each item: {{"type": "...", "value": "...", "date": "...", "snomed_id": "..."}}
        """

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            structured = response.text.replace("```json", "").replace("```", "").strip()

            # Determine which EMR categories were non-empty — for GDPR Art. 15
            fields_used = []
            try:
                parsed = json.loads(structured)
                types_found = set(item.get("type", "").lower() for item in parsed if isinstance(item, dict))
                field_map = {
                    "diagnosis": "Medical Diagnoses",
                    "medication": "Prescribed Medications",
                    "lab": "Laboratory Results",
                    "symptom": "Recorded Symptoms",
                }
                for key, label in field_map.items():
                    if any(key in t for t in types_found):
                        fields_used.append(label)
            except Exception:
                fields_used = ["Medical Records (general)"]

            return structured, fields_used
        except Exception as e:
            print(f"Error extracting data: {e}")
            return "[]", []

    async def compact_history(self, messages: list) -> str:
        """Delegate to Gemini-based compaction for best clinical fidelity."""
        return await compact_with_gemini(messages, self.client, self.model_name)

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

        emr_fields_used = []
        clinical_context = ""

        # --- Check if compaction is needed on the current history ---
        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print(f"[COMPACTION] Token threshold exceeded. Running Gemini compaction...")
            old_turns, _ = split_for_compaction(history)
            fresh_summary = await self.compact_history(old_turns)
            if compacted_summary:
                merge_prompt = (
                    f"Merge these two clinical summaries into one concise summary (max 250 words). "
                    f"Retain all clinical facts. Do not hallucinate.\n\n"
                    f"SUMMARY 1:\n{compacted_summary}\n\nSUMMARY 2:\n{fresh_summary}"
                )
                try:
                    merge_resp = await self.client.aio.models.generate_content(
                        model=self.model_name,
                        contents=merge_prompt,
                        config=types.GenerateContentConfig(temperature=0.1)
                    )
                    new_compacted_summary = merge_resp.text.strip()
                except Exception:
                    new_compacted_summary = fresh_summary
            else:
                new_compacted_summary = fresh_summary
            was_compacted = True
            print(f"[COMPACTION] Done. Summary length: {len(new_compacted_summary)} chars")

        # Build the windowed history to pass to LLM
        history_for_llm = build_llm_history(history, new_compacted_summary)

        # --- Build system instruction ---
        if system_prompt:
            # RAG pipeline provided a focused system prompt — use it as EMR context
            emr_section = (
                f"{system_prompt}\n\n{GDPR_EMR_READONLY_NOTE}"
            )
            emr_fields_used = ["RAG Pipeline (SNOMED Knowledge Graph)"]
        elif emr_consent:
            # No RAG prompt but consent given — fall back to raw EMR extraction
            raw_data = self.get_patient_data(patient_id)
            clinical_context_raw, emr_fields_used = await self.extract_clinical_data(raw_data)
            clinical_context = clinical_context_raw if clinical_context_raw != "[]" else raw_data
            emr_section = (
                f"PATIENT EMR CONTEXT (consented by patient, read-only):\n{clinical_context}\n\n"
                f"{GDPR_EMR_READONLY_NOTE}"
            )
        else:
            emr_section = (
                "EMR ACCESS: The patient has NOT consented to EMR data access for this session. "
                "You must NOT reference any specific medical records. Provide only general "
                "medical information and strongly encourage the patient to consult their doctor."
            )

        system_instruction_text = (
            f"You are Robert, a helpful AI medical assistant. "
            f"You assist patients in understanding their health but you are NOT a licensed physician.\n\n"
            f"{emr_section}\n\n"
            f"GUIDELINES:\n"
            f"- Avoid medical jargon where possible, or explain it clearly if used.\n"
            f"- TONE: Be empathetic, professional, and calm.\n"
            f"- NEVER congratulate a user on symptoms or illness "
            f"(e.g., do NOT say 'It's great you have a cold').\n"
            f"- If the patient shares negative symptoms, acknowledge them with concern "
            f"(e.g., 'I am sorry to hear that'), not excitement.\n"
            f"- Base answers on the provided EMR context (if consented) and general medical knowledge.\n"
            f"- NEVER claim to diagnose, prescribe, or replace a licensed physician's judgment.\n"
            f"- If the user notices incorrect information in their records, inform them to "
            f"contact their healthcare provider or system administrator for corrections — "
            f"you cannot modify medical records.\n"
            f"{GDPR_RESPONSE_DISCLAIMER}"
        )

        # Build formatted contents for Gemini API
        formatted_contents = []
        for msg in history_for_llm:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])]
                )
            )

        # Add current user message
        formatted_contents.append(
            types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_text,
                    temperature=0.7
                )
            )

            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count if usage else 0
            output_tokens = usage.candidates_token_count if usage else 0
            total_tokens = usage.total_token_count if usage else 0

            logger.debug("[TOKEN_USAGE] input=%d, output=%d, total=%d", input_tokens, output_tokens, total_tokens)

            return {
                "response": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name,
                "emr_fields_used": emr_fields_used,
                "was_compacted": was_compacted,
                "new_compacted_summary": new_compacted_summary,
            }
        except Exception as e:
            print(f"GenAI Error: {e}")
            return {
                "response": "Ensure you have the correct API KEY. Error: " + str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_name": self.model_name,
                "emr_fields_used": [],
                "was_compacted": False,
                "new_compacted_summary": compacted_summary,
            }

