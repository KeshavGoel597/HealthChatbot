from fastapi import APIRouter, HTTPException, Request
from app.models.chat_models import ChatSession, SessionMessage, ChatResponse, ChatMessage
from app.routers.chat import gemini_service, get_hf_service, _get_emr_path
from app.services.rag.pipeline import run_pipeline
import os
import json
import uuid
from datetime import datetime
from typing import List

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
async def send_message(session_id: str, request: ChatMessage, req: Request):
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
        session.title = request.message[:30] + "..." if len(request.message) > 30 else request.message
        
    _save_session(session) # Save user message first
    
    # 3. Run RAG pipeline for focused context
    try:
        index = req.app.state.embedding_index
        graph = req.app.state.knowledge_graph
        emr_path = _get_emr_path(session.patient_id)

        pipeline_result = run_pipeline(
            query=request.message,
            emr_path=emr_path,
            index=index,
            graph=graph,
            patient_id=session.patient_id,
        )
        system_prompt = pipeline_result.system_prompt
    except Exception as e:
        print(f"RAG pipeline error: {e}")
        system_prompt = ""

    # 4. Call LLM
    try:
        use_gemini = False
        if request.model and request.model.startswith("gemini"):
            use_gemini = True
        elif request.model is None:
            use_gemini = False

        history_for_llm = [
            {"role": m.role, "content": m.content} 
            for m in session.messages[:-1]
        ]
        
        if use_gemini:
            result = await gemini_service.chat(
                request.message, session.patient_id,
                history=history_for_llm, system_prompt=system_prompt,
            )
        else:
            service = get_hf_service()
            result = await service.chat(
                request.message, session.patient_id,
                history=history_for_llm, system_prompt=system_prompt,
            )
            
        # 5. Add assistant message
        assistant_msg = SessionMessage(
            role="assistant",
            content=result["response"],
            timestamp=datetime.now().isoformat()
        )
        session.messages.append(assistant_msg)
        _save_session(session)
        
        return ChatResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
