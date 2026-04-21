# backend/app/services/llm_factory.py
from functools import lru_cache
from app.services.ollama_service import OllamaService

@lru_cache(maxsize=1)
def get_hf_service():
    """Lazy load HuggingFace service to avoid memory penalty on startup."""
    from app.services.huggingface_service import HuggingFaceService
    return HuggingFaceService()


@lru_cache(maxsize=1)
def get_ollama_service():
    """Lazy load Ollama service to avoid provider side effects on import."""
    return OllamaService()


@lru_cache(maxsize=1)
def get_gemini_service():
    """Lazy load Gemini service so missing API key doesn't break app startup."""
    from app.services.gemini_service import GeminiService
    return GeminiService()

def get_llm_service(model_name: str):
    """
    Factory to retrieve the appropriate LLM service based on the model name.
    """
    model_selection = model_name.lower() if model_name else "huggingface"
    
    if model_selection in ("ollama", "medgemma"):
        return get_ollama_service()
    elif "gemini" in model_selection:
        return get_gemini_service()
    else:
        return get_hf_service()
