import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from app.models.chat_models import ChatMessage, ChatResponse
from app.services.gemini_service import GeminiService
from app.services.huggingface_service import HuggingFaceService
from app.services.ollama_service import OllamaService
from app.services.rag.pipeline import run_pipeline
from app.services.safety import check_safety

logger = logging.getLogger(__name__)

router = APIRouter()
gemini_service = GeminiService()
# Ollama is an API client, so it doesn't need lazy loading like HF
ollama_service = OllamaService()

# Lazy init for HF service to avoid loading 8GB on startup if not used
@lru_cache(maxsize=1)
def get_hf_service():
    return HuggingFaceService()


def _get_emr_path(patient_id: str) -> str:
    """Resolve patient EMR file path."""
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base_dir, "data", f"{patient_id}.json")


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatMessage, req: Request):
    try:
        print(f"[CHAT] Received request: model={request.model}, patient={request.patient_id}, consent={request.emr_consent}", flush=True)
        
        # 1. Safety Guardrail
        safety = check_safety(request.message)
        if safety.triggered:
            return ChatResponse(
                response=safety.response,
                input_tokens=0,
                output_tokens=0,
                total_tokens=0,
                model_name="safety-guardrail",
                emr_fields_used=[],
                was_compacted=False,
            )

        # 2. RAG Pipeline (Knowledge Graph + EMR Matching)
        system_prompt = ""
        if request.emr_consent:
            index = req.app.state.embedding_index
            graph = req.app.state.knowledge_graph
            extractor = req.app.state.term_extractor
            emr_path = _get_emr_path(request.patient_id)

            # run_pipeline returns clinical context gathered from the Knowledge Graph
            result = run_pipeline(
                query=request.message,
                emr_path=emr_path,
                index=index,
                graph=graph,
                patient_id=request.patient_id,
                extractor=extractor,
            )
            system_prompt = result.system_prompt

        print("=== SYSTEM PROMPT SENT TO LLM ===", flush=True)
        print(system_prompt if system_prompt else "(empty — no consent or no EMR)", flush=True)
        print("=== END SYSTEM PROMPT ===", flush=True)

        # 3. Privacy Anonymization Tools
        presidio_analyzer = getattr(req.app.state, "presidio_analyzer", None)
        presidio_anonymizer = getattr(req.app.state, "presidio_anonymizer", None)

        # 4. Route to local Ollama, Gemini, or HuggingFace
        model_selection = request.model.lower() if request.model else "huggingface"

        if model_selection in ("ollama", "medgemma"):
            llm_result = await ollama_service.chat(
                request.message, 
                request.patient_id,
                system_prompt=system_prompt,
                emr_consent=request.emr_consent,
                presidio_analyzer=presidio_analyzer,
                presidio_anonymizer=presidio_anonymizer,
            )
        elif "gemini" in model_selection:
            llm_result = await gemini_service.chat(
                request.message, 
                request.patient_id,
                system_prompt=system_prompt,
                emr_consent=request.emr_consent,
                presidio_analyzer=presidio_analyzer,
                presidio_anonymizer=presidio_anonymizer,
            )
        else:
            # Default to HuggingFace (loaded locally)
            service = get_hf_service()
            llm_result = await service.chat(
                request.message, 
                request.patient_id,
                system_prompt=system_prompt,
                emr_consent=request.emr_consent,
                presidio_analyzer=presidio_analyzer,
                presidio_anonymizer=presidio_anonymizer,
            )

        return ChatResponse(**llm_result)

    except RuntimeError as e:
        if "Ollama" in str(e) or "Hugging Face" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    return {"status": "ok"}