# backend/app/services/llm_factory.py
from functools import lru_cache
from app.services.gemini_service import GeminiService
from app.services.ollama_service import OllamaService
from app.services.huggingface_service import HuggingFaceService

# Pre-instantiate lightweight services
gemini_service = GeminiService()
ollama_service = OllamaService()

@lru_cache(maxsize=1)
def get_hf_service():
    """Lazy load HuggingFace service to avoid memory penalty on startup."""
    return HuggingFaceService()

def get_llm_service(model_name: str):
    """
    Factory to retrieve the appropriate LLM service based on the model name.
    """
    model_selection = model_name.lower() if model_name else "huggingface"
    
    if model_selection in ("ollama", "medgemma"):
        return ollama_service
    elif "gemini" in model_selection:
        return gemini_service
    else:
        return get_hf_service()
