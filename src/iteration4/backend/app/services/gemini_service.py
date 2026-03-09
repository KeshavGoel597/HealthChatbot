import os
import json
import asyncio
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Use Standard Async Client
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = "gemini-2.5-flash-lite" 
        
    def get_patient_data(self, patient_id: str) -> str:
        # Load patient data from json file
        # Assuming data is in iteration2/backend/data
        # We need to correctly locate the file
        
        # Try finding the data file relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        data_path = os.path.join(base_dir, "data", f"{patient_id}.json")
        
        if not os.path.exists(data_path):
             # Fallback: check if it's in iteration1 just in case, or default to empty
             return "{}"

        try:
            with open(data_path, "r") as f:
                return f.read()
        except Exception as e:
            print(f"Error reading patient data: {e}")
            return "{}"

    async def extract_clinical_data(self, raw_text: str) -> str:
        prompt = f"""
        Act as a Clinical Coder.
        TASK: Extract Diagnosis, Meds, Labs, Symptoms from the text below.
        CRITICAL: Keep Dates. Map to SNOMED IDs.
        RAW DATA: {raw_text}
        OUTPUT: Valid JSON List.
        """
        
        try:
            # Use async client
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            print(f"Error extracting data: {e}")
            return "[]"


    async def chat(self, message: str, patient_id: str = "patient101", history: list = [], *, system_prompt: str = "") -> dict:
        
        # Use RAG-assembled prompt if provided, otherwise fall back to raw EMR
        if system_prompt:
            system_instruction_text = system_prompt
        else:
            raw_data = self.get_patient_data(patient_id)
            clinical_context = await self.extract_clinical_data(raw_data)
            if clinical_context == "[]":
                clinical_context = raw_data

            system_instruction_text = (
                f"You are Robert, a helpful AI medical assistant for patients. "
                f"Your goal is to explain their medical records to them in simple, easy-to-understand language. "
                f"Avoid medical jargon where possible, or explain it if necessary. "
                f"Always be empathetic and clear. "
                f"Context from their Electronic Medical Records (EMR): {clinical_context}"
            )
        
        try:
           
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction_text
                )
            )
            
            # Extract usage
            usage = response.usage_metadata
            input_tokens = usage.prompt_token_count if usage else 0
            output_tokens = usage.candidates_token_count if usage else 0
            total_tokens = usage.total_token_count if usage else 0
            
            return {
                "response": response.text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name
            }
        except Exception as e:
            print(f"GenAI Error: {e}")
            # Try to print more auth info if possible
            return {
                "response": "Ensure you have the correct API KEY. Error: " + str(e),
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model_name": self.model_name
            }
