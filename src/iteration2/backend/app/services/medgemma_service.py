import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
import os
import asyncio
import re
from dotenv import load_dotenv

load_dotenv()


class MedGemmaService:
    """Service for Google MedGemma 4B medical model."""
    
    def __init__(self):
        self.model_name = "google/medgemma-1.5-4b-it"
        self.processor = None
        self.model = None

    def _summarize_context(self, raw_data: str) -> str:
        """Extract key clinical info from raw patient data string."""
        summary = []
        
        # Demographics
        age_match = re.search(r'age: "([^"]+)"', raw_data)
        sex_match = re.search(r'sex: "([^"]+)"', raw_data)
        if age_match or sex_match:
            age = age_match.group(1) if age_match else "?"
            sex = sex_match.group(1) if sex_match else "?"
            summary.append(f"PATIENT: Age {age}, Sex {sex}")
            
        # Diagnoses
        diagnoses = set(re.findall(r'"diag" => "([^"]+)"', raw_data))
        cleaned_diagnoses = [d.strip() for d in diagnoses if d.strip() and d.strip() != "@10"]
        if cleaned_diagnoses:
            summary.append(f"DIAGNOSES: {', '.join(cleaned_diagnoses)}")
            
        # Symptoms
        symptoms = set(re.findall(r'"sym" => "([^"]+)"', raw_data))
        cleaned_symptoms = [s.strip() for s in symptoms if s.strip() and s.strip() != "FCU"]
        if cleaned_symptoms:
            summary.append(f"SYMPTOMS: {', '.join(cleaned_symptoms)}")
            
        # Medications
        meds = set(re.findall(r'"medicine" => "([^"]+)"', raw_data))
        if meds:
            summary.append(f"MEDICATIONS: {', '.join(list(meds)[:10])}")
            
        # Labs
        lab_summary = []
        for lab in ["Hemoglobin", "RBS", "Total WBC Count", "Platelet Count"]:
            matches = re.findall(f'"name" => "{lab}", "value" => "([^"]+)", "date" => "([^"]+)"', raw_data)
            if matches:
                last_val, last_date = matches[-1]
                lab_summary.append(f"{lab}: {last_val} ({last_date})")
        if lab_summary:
            summary.append(f"RECENT LABS: {', '.join(lab_summary)}")

        return "\n".join(summary)

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
                print(f"MedGemma loaded successfully.")
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
        # Decode only the new tokens (skip the input prompt)
        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    async def chat(self, message: str, patient_id: str = "patient101", history: list = []) -> dict:
        self._load_model()
        raw_data = self.get_patient_data(patient_id)
        clinical_summary = self._summarize_context(raw_data)
        print(f"MedGemma: Clinical summary for {patient_id}: {clinical_summary[:200]}")

        # Build messages for MedGemma (text-only, no images needed for chat)
        system_content = (
            "You are Robert, a warm and helpful AI medical assistant. "
            "You have access to the patient's medical records below. "
            "Your responsibilities:\n"
            "1. First, directly address the patient's CURRENT question or complaint.\n"
            "2. Then, relate it to their medical background when relevant.\n"
            "3. If the symptom is new (not in records), provide helpful general medical guidance.\n"
            "4. Be empathetic, concise (2-3 paragraphs), and always recommend consulting their doctor.\n\n"
            f"PATIENT MEDICAL RECORDS:\n{clinical_summary}"
        )

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": f"{system_content}\n\nPatient says: {message}"}
            ]}
        ]

        print(f"MedGemma: Generating response...")
        loop = asyncio.get_event_loop()
        response_text = await loop.run_in_executor(None, self._generate, messages)

        # Token counting (approximate)
        input_text = system_content + message
        input_tokens = len(input_text.split()) * 2  # rough estimate
        output_tokens = len(response_text.split()) * 2

        return {
            "response": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "model_name": self.model_name
        }
