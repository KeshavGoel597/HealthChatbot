"""
HuggingFaceService — GDPR-compliant, RAG-enhanced, context-compaction-aware
=============================================================================
Uses deterministic compaction (no extra LLM call) since Qwen-0.5B is too
small to reliably self-summarise clinical conversations.

RAG Integration:
  When a system_prompt is provided (from the RAG pipeline), it is used as the
  primary system content. Otherwise falls back to regex-based EMR summary.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import asyncio
from dotenv import load_dotenv
from app.services.torch_runtime import detect_torch_runtime

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


class HuggingFaceService:
    def __init__(self):
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = None
        self.model = None
        self.backend_name = "cpu"
        self.device = torch.device("cpu")

    def _summarize_emr_context(self, raw_data: str) -> tuple[str, list]:
        return summarize_emr_context(raw_data)

    def _load_model(self):
        if self.model is None:
            print(f"Loading {self.model_name}...")
            try:
                backend_name, device, dtype, use_device_map_auto = detect_torch_runtime()
                self.backend_name = backend_name
                self.device = device
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                kwargs = {"torch_dtype": dtype}
                if use_device_map_auto:
                    kwargs["device_map"] = "auto"

                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **kwargs,
                )
                if not use_device_map_auto:
                    self.model.to(self.device)
                print(f"{self.model_name} loaded on {self.backend_name}.")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load model '{self.model_name}'. "
                    f"Ensure you are authenticated with Hugging Face. Error: {str(e)}"
                )

    def get_patient_data(self, patient_id: str) -> str:
        return load_patient_data(patient_id)

    def _generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self._load_model()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
        *,
        system_prompt: str = "",
        presidio_analyzer=None,
        presidio_anonymizer=None,
    ) -> dict:
        if history is None:
            history = []

        self._load_model()

        emr_fields_used = []
        clinical_summary = ""

        # --- Deterministic compaction (GDPR Art. 5(1)(c)) ---
        was_compacted = False
        new_compacted_summary = compacted_summary

        if needs_compaction(history):
            print(f"[COMPACTION] Running deterministic compaction for HF model...")
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

        # --- Build system content ---
        if system_prompt:
            # RAG pipeline provided focused clinical context
            emr_section = system_prompt
            emr_fields_used = ["RAG Pipeline (SNOMED Knowledge Graph)"]
        elif emr_consent:
            # No RAG prompt but consent given — fall back to regex-based EMR summary
            raw_data = self.get_patient_data(patient_id)
            clinical_summary, emr_fields_used = self._summarize_emr_context(raw_data)
            emr_section = f"PATIENT SUMMARY (consented, read-only):\n{clinical_summary}"
        else:
            emr_section = (
                "EMR ACCESS: Patient has NOT consented to EMR access. "
                "Do NOT reference any specific medical records. "
                "Provide only general medical guidance."
            )

        system_content = (
            "You are Robert, a helpful AI medical assistant for patients.\n"
            "You are NOT a licensed physician.\n"
            f"{emr_section}\n"
            "Be empathetic, professional, and calm.\n"
            "NEVER congratulate a user on symptoms or illness.\n"
            "If the user shares negative symptoms, acknowledge them with concern.\n"
            "If the user mentions incorrect records, tell them to contact their "
            "healthcare provider — you cannot modify medical records.\n"
            f"{GDPR_SYSTEM_SUFFIX}"
        )

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

        print("=== OUTBOUND LLM DEBUG (HF) ===", flush=True)
        print(
            f"presidio_analyzer={'on' if presidio_analyzer is not None else 'off'} "
            f"presidio_anonymizer={'on' if presidio_anonymizer is not None else 'off'}",
            flush=True,
        )
        print(f"message_sanitized={sanitized_message}", flush=True)
        print(f"system_content_sanitized={sanitized_system_content[:1500]}", flush=True)
        print("=== END OUTBOUND LLM DEBUG (HF) ===", flush=True)

        messages = [{"role": "system", "content": sanitized_system_content}]

        for msg in sanitized_history_for_llm:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": sanitized_message})

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        print(f"Generating for model {self.model_name} with prompt length: {len(prompt)}")
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

