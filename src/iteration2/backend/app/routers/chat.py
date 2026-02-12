from fastapi import APIRouter, HTTPException
from app.models.chat_models import ChatMessage, ChatResponse
from app.services.gemini_service import GeminiService
from app.services.huggingface_service import HuggingFaceService

router = APIRouter()
gemini_service = GeminiService()
# Lazy init for HF service to avoid loading 8GB on startup if not used
hf_service = None

def get_hf_service():
    global hf_service
    if hf_service is None:
        hf_service = HuggingFaceService()
    return hf_service

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatMessage):
    try:
        # Logic: 
        # If model is specified and starts with "gemini", use Gemini.
        # If model is specified and is something else, use HF.
        # If model is NOT specified, default to HF (medgemma) as requested.
        
        use_gemini = False
        if request.model and request.model.startswith("gemini"):
            use_gemini = True
        elif request.model is None:
            use_gemini = False # Default to HF
            
        if use_gemini:
            result = await gemini_service.chat(request.message, request.patient_id)
        else:
            service = get_hf_service()
            result = await service.chat(request.message, request.patient_id)
            
        return ChatResponse(**result)
    except RuntimeError as e:
        # Catch our specific HF loading error
        if "Hugging Face" in str(e):
             raise HTTPException(status_code=403, detail=str(e))
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "ok"}
