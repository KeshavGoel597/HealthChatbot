from fastapi import APIRouter, HTTPException
from app.models.chat_models import ChatSession, SessionMessage, ChatResponse, ChatMessage
from app.routers.chat import gemini_service, get_hf_service
import os
import json
import uuid
from app.services.sarvam_service import SarvamService
from app.services.medgemma_service import MedGemmaService
from datetime import datetime
from typing import List

sarvam_service = SarvamService()

_medgemma_service = None
def get_medgemma_service():
    global _medgemma_service
    if _medgemma_service is None:
        _medgemma_service = MedGemmaService()
    return _medgemma_service

router = APIRouter()

# Data storage path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

def _get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")

def _load_session(session_id: str) -> ChatSession:
    path = _get_session_path(session_id)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return ChatSession(**data)
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None

def _save_session(session: ChatSession):
    path = _get_session_path(session.id)
    with open(path, "w") as f:
        json.dump(session.dict(), f, indent=2)

@router.get("/sessions/{patient_id}", response_model=List[ChatSession])
async def list_sessions(patient_id: str):
    """List all sessions for a patient."""
    sessions = []
    if not os.path.exists(SESSIONS_DIR):
        return []
        
    for filename in os.listdir(SESSIONS_DIR):
        if filename.endswith(".json"):
            # Load basic info (could be optimized to not read full file)
            session_id = filename.replace(".json", "")
            session = _load_session(session_id)
            if session and session.patient_id == patient_id:
                 # Exclude messages for list view to save bandwidth? 
                 # For now, return full object, it's fine for small history
                sessions.append(session)
    
    # Sort by created_at desc
    sessions.sort(key=lambda x: x.created_at, reverse=True)
    return sessions

@router.post("/sessions/{patient_id}", response_model=ChatSession)
async def create_session(patient_id: str):
    """Create a new chat session."""
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    new_session = ChatSession(
        id=session_id,
        patient_id=patient_id,
        created_at=now,
        messages=[]
    )
    _save_session(new_session)
    return new_session

@router.get("/sessions/{session_id}/messages", response_model=ChatSession)
async def get_session(session_id: str):
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@router.post("/sessions/{session_id}/message", response_model=ChatResponse)
async def send_message(session_id: str, request: ChatMessage):
    """Send a message to a session and get response."""
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
        
    # 1. Add user message
    user_msg = SessionMessage(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat()
    )
    session.messages.append(user_msg)
    
    # 2. Update title if it's the first message
    if len(session.messages) == 1:
        # Simple title generation: first 30 chars of message
        session.title = request.message[:30] + "..." if len(request.message) > 30 else request.message
        
    _save_session(session) # Save user message first
    
    # 3. Call LLM
    try:
        # Determine service
        use_gemini = False
        use_medgemma = False
        if request.model and request.model.startswith("gemini"):
            use_gemini = True
        elif request.model and request.model == "medgemma":
            use_medgemma = True
        elif request.model is None:
            use_gemini = False # Default to HF
            
        # Translation Logic (Input)
        target_lang = request.language or "en-IN"
        llm_input_message = request.message
        
        if target_lang != "en-IN":
            print(f"Translating input from {target_lang} to en-IN...")
            translated_input = sarvam_service.translate(request.message, target_lang, "en-IN")
            if translated_input:
                llm_input_message = translated_input
                print(f"Translated input: {llm_input_message}")

        # Convert session history to list format expected by services
        # (Assuming services accept list of dicts: [{'role': 'user', 'content': '...'}])
        history_for_llm = [
            {"role": m.role, "content": m.content} 
            for m in session.messages[:-1] # Exclude current user message if service adds it? 
                                           # Usually services expect history + current message separate
        ]
        
        # NOTE: We are passing the original history (which might be in mixed languages)
        # and the potentially translated *current* message (in English) to the LLM.
        # This is a compromise. Ideally we'd translate history too.
        
        if use_gemini:
            result = await gemini_service.chat(llm_input_message, session.patient_id, history=history_for_llm)
        elif use_medgemma:
            service = get_medgemma_service()
            result = await service.chat(llm_input_message, session.patient_id, history=history_for_llm)
        else:
            service = get_hf_service()
            result = await service.chat(llm_input_message, session.patient_id, history=history_for_llm)
            
        # Translation Logic (Output)
        final_response_text = result["response"]
        audio_content = None
        
        if target_lang != "en-IN":
             print(f"Translating response to {target_lang}...")
             translated_response = sarvam_service.translate(final_response_text, "en-IN", target_lang)
             if translated_response:
                 final_response_text = translated_response
        
        # TTS Logic
        if request.audio_requested:
            print(f"Generating audio for {target_lang}...")
            # Use the final language code (Sarvam handles various indic codes)
            audio_content = sarvam_service.text_to_speech(final_response_text, target_lang)

        # 4. Add assistant message (Save the final displayed text)
        assistant_msg = SessionMessage(
            role="assistant",
            content=final_response_text,
            timestamp=datetime.now().isoformat()
        )
        session.messages.append(assistant_msg)
        _save_session(session)
        
        # Update result with final details
        result["response"] = final_response_text
        result["language"] = target_lang
        result["audio_content"] = audio_content
        print(f"DEBUG: audio_content present: {audio_content is not None}, length: {len(audio_content) if audio_content else 0}")
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
