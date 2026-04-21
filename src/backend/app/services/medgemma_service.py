"""Backward-compatible alias for the Ollama service."""

from app.services.ollama_service import OllamaService


class MedGemmaService(OllamaService):
    """Compatibility wrapper: MedGemma remains the default Ollama model."""

    pass
