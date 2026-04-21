import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException, Request
from app.models.chat_models import ChatMessage, ChatResponse
from app.services.llm_factory import get_llm_service
from app.services.rag_orchestrator import build_rag_system_prompt
from app.services.safety import check_safety

logger = logging.getLogger(__name__)

router = APIRouter()




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
        system_prompt, emr_fields_used_from_rag = build_rag_system_prompt(
            req=req,
            message=request.message,
            patient_id=request.patient_id,
            emr_consent=request.emr_consent,
        )

        print("=== SYSTEM PROMPT SENT TO LLM ===", flush=True)
        print(system_prompt if system_prompt else "(empty — no consent or no EMR)", flush=True)
        print("=== END SYSTEM PROMPT ===", flush=True)

        # 3. Privacy Anonymization Tools
        presidio_analyzer = getattr(req.app.state, "presidio_analyzer", None)
        presidio_anonymizer = getattr(req.app.state, "presidio_anonymizer", None)

        # 4. Route to local Ollama, Gemini, or HuggingFace
        llm_service = get_llm_service(request.model)
        llm_result = await llm_service.chat(
            request.message, 
            request.patient_id,
            system_prompt=system_prompt,
            emr_consent=request.emr_consent,
            presidio_analyzer=presidio_analyzer,
            presidio_anonymizer=presidio_anonymizer,
        )

        emr_fields_used = llm_result.get("emr_fields_used", [])
        if emr_fields_used == ["RAG Pipeline (SNOMED Knowledge Graph)"] and emr_fields_used_from_rag:
            llm_result["emr_fields_used"] = emr_fields_used_from_rag

        return ChatResponse(**llm_result)

    except RuntimeError as e:
        if "Ollama" in str(e) or "Hugging Face" in str(e):
            raise HTTPException(status_code=503, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("Chat error")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/health")
async def health_check():
    return {"status": "ok"}