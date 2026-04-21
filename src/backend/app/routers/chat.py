import logging

from fastapi import APIRouter, HTTPException, Request
from app.models.chat_models import ChatMessage, ChatResponse
from app.services.chat_orchestrator import run_llm_turn
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

        # 2. Shared orchestration (RAG + anonymization + provider call)
        llm_result = await run_llm_turn(
            req=req,
            message=request.message,
            patient_id=request.patient_id,
            model_name=request.model,
            emr_consent=bool(request.emr_consent),
        )

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