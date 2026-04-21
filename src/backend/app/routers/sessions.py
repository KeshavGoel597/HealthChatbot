"""
Sessions Router — GDPR-compliant, RAG-enhanced, context-compaction-aware
=========================================================================
Key GDPR integrations:
  Art. 5(1)(a)  — Consent for EMR access stored per-session
  Art. 5(1)(c)  — Data minimisation via context compaction
  Art. 5(1)(e)  — Storage limitation: messages only saved with explicit consent;
                  sessions auto-expire after RETENTION_DAYS
  Art. 17       — Erasure handled by gdpr_router.py

RAG Pipeline:
  Before every LLM call, the SNOMED knowledge-graph RAG pipeline runs to
  produce a focused system prompt with clinically relevant context.

Context Compaction:
  Before every LLM call, history is checked against MAX_TOKENS_BEFORE_COMPACT.
  If exceeded, older turns are summarised and replaced by a compact_summary block.
"""

import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile, File, Form
from app.models.chat_models import (
    ChatSession, SessionMessage, ChatResponse, ChatMessage, SessionCreateRequest
)
from app.services.chat_orchestrator import run_llm_turn
from app.services.session_store import (
    SESSIONS_DIR, get_session_path, load_session, save_session,
)
from app.services.safety import check_safety
import os
import uuid
from app.services.sarvam_service import SarvamService
from app.services.context_compaction import RETENTION_DAYS
from datetime import datetime, timedelta
from typing import List, Optional

sarvam_service = SarvamService()

logger = logging.getLogger(__name__)

router = APIRouter()

# Aliases for backward compatibility within this module
_get_session_path = get_session_path
_load_session = load_session
_save_session = save_session


def run_retention_cleanup():
    """
    GDPR Art. 5(1)(e) — Storage Limitation.
    Delete sessions whose expires_at datetime has passed.
    Called at startup and can be called periodically.
    """
    try:
        filenames = os.listdir(SESSIONS_DIR)
    except FileNotFoundError:
        return

    now = datetime.now()
    deleted_count = 0

    for filename in filenames:
        if not filename.endswith(".json"):
            continue
        session_id = filename.replace(".json", "")
        session = _load_session(session_id)
        if session and session.expires_at:
            try:
                expiry = datetime.fromisoformat(session.expires_at)
                if now > expiry:
                    os.remove(_get_session_path(session_id))
                    deleted_count += 1
                    print(f"[GDPR Art.5(1)(e)] Expired session {session_id} deleted (expired: {session.expires_at}).")
            except Exception as e:
                print(f"Error checking expiry for session {session_id}: {e}")

    if deleted_count > 0:
        print(f"[GDPR Retention Cleanup] Deleted {deleted_count} expired session(s).")
    else:
        print("[GDPR Retention Cleanup] No expired sessions found.")


@router.get("/sessions/{patient_id}", response_model=List[ChatSession])
async def list_sessions(patient_id: str):
    """List all active (non-expired) sessions for a patient."""
    sessions = []
    try:
        filenames = os.listdir(SESSIONS_DIR)
    except FileNotFoundError:
        return []

    now = datetime.now()
    for filename in filenames:
        if filename.endswith(".json"):
            session_id = filename.replace(".json", "")
            session = _load_session(session_id)
            if session and session.patient_id == patient_id:
                # Skip expired sessions (they will be cleaned on next startup)
                if session.expires_at:
                    try:
                        if now > datetime.fromisoformat(session.expires_at):
                            continue
                    except Exception:
                        pass
                sessions.append(session)

    sessions.sort(key=lambda x: x.created_at, reverse=True)
    return sessions


@router.post("/sessions/{patient_id}", response_model=ChatSession)
async def create_session(patient_id: str, request: Optional[SessionCreateRequest] = None):
    """
    Create a new chat session.
    GDPR Art. 5(1)(a,e): Records explicit consent for EMR access and history storage.
    Sets expires_at = now + RETENTION_DAYS for automatic cleanup.
    """
    if request is None:
        request = SessionCreateRequest()

    session_id = str(uuid.uuid4())
    now = datetime.now()
    expires_at = (now + timedelta(days=RETENTION_DAYS)).isoformat()

    new_session = ChatSession(
        id=session_id,
        patient_id=patient_id,
        created_at=now.isoformat(),
        messages=[],
        emr_consent=request.emr_consent,
        store_history_consent=request.store_history_consent,
        expires_at=expires_at,
    )
    _save_session(new_session)
    print(f"[SESSION] Created {session_id} | EMR consent: {request.emr_consent} | History consent: {request.store_history_consent} | Expires: {expires_at}")
    return new_session


@router.get("/sessions/{session_id}/messages", response_model=ChatSession)
async def get_session(session_id: str):
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.post("/sessions/{session_id}/message", response_model=ChatResponse)
async def send_message(session_id: str, request: ChatMessage, req: Request):
    """
    Send a message to a session and get a response.

    GDPR compliance:
      - EMR access only if session.emr_consent is True (can be overridden per-request)
      - Messages only persisted if session.store_history_consent is True
      - Context compaction runs automatically to minimise token usage (Art. 5(1)(c))
      - emr_fields_used returned for evidence transparency (Art. 15)

    RAG pipeline:
      - Runs SNOMED knowledge-graph pipeline to focus clinical context per query
    """
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Per-request consent can override or supplement the session-level consent
    effective_emr_consent = request.emr_consent if request.emr_consent is not None else session.emr_consent
    effective_history_consent = request.store_history_consent if request.store_history_consent is not None else session.store_history_consent

    # Persist updated consent back to session if changed
    if effective_emr_consent != session.emr_consent or effective_history_consent != session.store_history_consent:
        session.emr_consent = effective_emr_consent
        session.store_history_consent = effective_history_consent
        _save_session(session)

    # 1. Add user message to in-memory session (GDPR: only save if consented)
    user_msg = SessionMessage(
        role="user",
        content=request.message,
        timestamp=datetime.now().isoformat(),
        emr_fields_used=[]
    )

    # Update title from first user message
    if len(session.messages) == 0:
        session.title = request.message[:30] + "..." if len(request.message) > 30 else request.message

    session.messages.append(user_msg)

    # Safety check
    safety = check_safety(request.message)
    if safety.triggered:
        assistant_msg = SessionMessage(
            role="assistant",
            content=safety.response,
            timestamp=datetime.now().isoformat(),
            emr_fields_used=[],
        )
        session.messages.append(assistant_msg)
        if effective_history_consent:
            _save_session(session)
        return ChatResponse(
            response=safety.response,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            model_name="safety-guardrail",
            language=request.language or "en-IN",
            emr_fields_used=[],
            was_compacted=False,
        )

    # GDPR Art. 5(1)(e): Only persist messages if user has opted in to storage
    if effective_history_consent:
        _save_session(session)

    # Build normalized inputs for provider call.
    target_lang = request.language or "en-IN"
    llm_input_message = request.message
    raw_history_for_llm = [
        {"role": msg.role, "content": msg.content}
        for msg in session.messages[:-1]
    ]

    # 2. Call shared orchestration (RAG + anonymization + provider call)
    try:
        result = await run_llm_turn(
            req=req,
            message=llm_input_message,
            patient_id=session.patient_id,
            model_name=request.model,
            emr_consent=effective_emr_consent,
            history=raw_history_for_llm,
            compacted_summary=session.compacted_summary,
        )

        # Translation (Output)
        final_response_text = result["response"]
        audio_content = None

        if target_lang != "en-IN":
            print(f"Translating response to {target_lang}...")
            translated_response = sarvam_service.translate(final_response_text, "en-IN", target_lang)
            if translated_response:
                final_response_text = translated_response

        # TTS
        if request.audio_requested:
            print(f"Generating audio for {target_lang}...")
            audio_content = sarvam_service.text_to_speech(final_response_text, target_lang)

        # 5. Update compacted_summary if the service ran compaction this turn
        if result.get("new_compacted_summary") != session.compacted_summary:
            session.compacted_summary = result.get("new_compacted_summary")

        # 6. Add assistant message with emr_fields_used for GDPR Art. 15
        emr_fields_used = result.get("emr_fields_used", [])

        assistant_msg = SessionMessage(
            role="assistant",
            content=final_response_text,
            timestamp=datetime.now().isoformat(),
            emr_fields_used=emr_fields_used
        )
        session.messages.append(assistant_msg)

        # GDPR Art. 5(1)(e): Only persist if user has consented to storage
        if effective_history_consent:
            _save_session(session)
        else:
            # Still save session metadata (consent flags, title, compacted_summary)
            # but with messages cleared — we store the conversation structure not the content
            session_meta = session.model_copy()
            session_meta.messages = []
            _save_session(session_meta)

        result["response"] = final_response_text
        result["language"] = target_lang
        result["audio_content"] = audio_content
        result["emr_fields_used"] = emr_fields_used
        result["was_compacted"] = result.get("was_compacted", False)

        # Remove internal key before returning
        result.pop("new_compacted_summary", None)

        return ChatResponse(**result)

    except Exception as e:
        logger.exception("Session chat error")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language_code: str = Form(default="unknown"),
):
    """Transcribe audio via Sarvam STT. Keeps API key server-side."""
    audio_bytes = await file.read()
    transcript = sarvam_service.speech_to_text(audio_bytes, language_code)
    if transcript is None:
        raise HTTPException(status_code=502, detail="Transcription failed")
    return {"transcript": transcript}


