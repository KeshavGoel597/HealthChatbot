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
from app.services.llm_base import BaseLLMService, GDPR_SYSTEM_SUFFIX

load_dotenv()


class HuggingFaceService(BaseLLMService):
    def __init__(self):
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = None
        self.model = None
        self.backend_name = "cpu"
        self.device = torch.device("cpu")

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

        was_compacted, new_compacted_summary, history_for_llm = self._run_deterministic_compaction(
            history,
            compacted_summary,
            label="HF model",
        )

        emr_section, emr_fields_used = self._build_emr_section(
            system_prompt=system_prompt,
            emr_consent=emr_consent,
            patient_id=patient_id,
            consent_prefix="PATIENT SUMMARY (consented, read-only):\n",
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

        sanitized_system_content, sanitized_history_for_llm, sanitized_message = self._anonymize_for_llm(
            system_content=system_content,
            history_for_llm=history_for_llm,
            message=message,
            presidio_analyzer=presidio_analyzer,
            presidio_anonymizer=presidio_anonymizer,
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

