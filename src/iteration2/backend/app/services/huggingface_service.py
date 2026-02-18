import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import asyncio
from dotenv import load_dotenv
import re

load_dotenv()


class HuggingFaceService:
    def __init__(self):
        self.model_name = "Qwen/Qwen2-0.5B-Instruct"
        self.tokenizer = None
        self.model = None

    def _summarize_context(self, raw_data: str) -> str:
        """
        Extracts key clinical information from the raw string data.
        Returns a concise summary for the LLM.
        """
        summary = []

        # 1. Demographics
        age_match = re.search(r'age: "([^"]+)"', raw_data)
        sex_match = re.search(r'sex: "([^"]+)"', raw_data)
        if age_match or sex_match:
            age = age_match.group(1) if age_match else "?"
            sex = sex_match.group(1) if sex_match else "?"
            summary.append(f"PATIENT: Age {age}, Sex {sex}")

        # 2. Diagnoses
        diagnoses = set(re.findall(r'"diag" => "([^"]+)"', raw_data))
        cleaned_diagnoses = [
            d.strip() for d in diagnoses if d.strip() and d.strip() != "@10"
        ]
        if cleaned_diagnoses:
            summary.append(f"DIAGNOSES: {', '.join(cleaned_diagnoses)}")

        # 3. Symptoms
        symptoms = set(re.findall(r'"sym" => "([^"]+)"', raw_data))
        cleaned_symptoms = [
            s.strip() for s in symptoms if s.strip() and s.strip() != "FCU"
        ]
        if cleaned_symptoms:
            summary.append(f"SYMPTOMS: {', '.join(cleaned_symptoms)}")

        # 4. Medications
        meds = set(re.findall(r'"medicine" => "([^"]+)"', raw_data))
        if meds:
            summary.append(f"MEDICATIONS: {', '.join(list(meds)[:10])}...")

        # 5. Recent Labs
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

        return "\n".join(summary)

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
        base_dir = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
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

    async def extract_clinical_data(self, raw_text: str) -> str:
        prompt = f"""<start_of_turn>user
Act as a Clinical Coder.
TASK: Extract Diagnosis, Meds, Labs, Symptoms from the text below.
CRITICAL: Keep Dates. Map to SNOMED IDs.
RAW DATA: {raw_text}
OUTPUT: Valid JSON List.<end_of_turn>
<start_of_turn>model
"""

        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, prompt)

        if "<start_of_turn>model" in response_text:
            response_text = response_text.split("<start_of_turn>model")[-1].strip()

        return response_text.replace("```json", "").replace("```", "").strip()

    async def chat(self, message: str, patient_id: str = "patient101", history: list = None) -> dict:
        if history is None:
            history = []

        self._load_model()
        raw_data = self.get_patient_data(patient_id)

        clinical_summary = self._summarize_context(raw_data)
        print(f"DEBUG: Clinical summary for {patient_id}: {clinical_summary[:200]}")

        messages = [
            {
                "role": "system",
                "content": (
                    "You are Robert, a helpful medical assistant for patients. "
                    "Use the provided patient summary to answer questions. "
                    "Be empathetic, professional, and calm. "
                    "NEVER congratulate a user on symptoms or sickness "
                    "(e.g., do not say 'It's great you have a cold'). "
                    "If the user shares negative symptoms, acknowledge them with concern "
                    "(e.g., 'I am sorry to hear that'), not excitement. "
                    f"PATIENT SUMMARY:\n{clinical_summary}"
                ),
            }
        ]

        # Insert conversation history
        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        # Add current user message
        messages.append({"role": "user", "content": message})

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
        }
