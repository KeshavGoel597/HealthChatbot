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

from fastapi import APIRouter, HTTPException, Request
from app.models.chat_models import (
    ChatSession, SessionMessage, ChatResponse, ChatMessage, SessionCreateRequest
)
from app.routers.chat import gemini_service, get_hf_service, _get_emr_path
from app.services.rag.pipeline import run_pipeline
from app.services.session_store import (
    SESSIONS_DIR, get_session_path, load_session, save_session,
)
import os
import json
import uuid
from app.services.sarvam_service import SarvamService
from app.services.medgemma_service import MedGemmaService
from app.services.context_compaction import RETENTION_DAYS
from datetime import datetime, timedelta
from typing import List, Optional

sarvam_service = SarvamService()

_medgemma_service = None
def get_medgemma_service():
    global _medgemma_service
    if _medgemma_service is None:
        _medgemma_service = MedGemmaService()
    return _medgemma_service

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
    if not os.path.exists(SESSIONS_DIR):
        return

    now = datetime.now()
    deleted_count = 0

    for filename in os.listdir(SESSIONS_DIR):
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
    if not os.path.exists(SESSIONS_DIR):
        return []

    now = datetime.now()
    for filename in os.listdir(SESSIONS_DIR):
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

    # GDPR Art. 5(1)(e): Only persist messages if user has opted in to storage
    if effective_history_consent:
        _save_session(session)

    # 2. Run RAG pipeline for focused context (only if EMR consent given)
    system_prompt = ""
    if effective_emr_consent:
        try:
            index = req.app.state.embedding_index
            graph = req.app.state.knowledge_graph
            extractor = getattr(req.app.state, "term_extractor", None)
            emr_path = _get_emr_path(session.patient_id)

            pipeline_result = run_pipeline(
                query=request.message,
                emr_path=emr_path,
                index=index,
                graph=graph,
                patient_id=session.patient_id,
                extractor=extractor,
            )
            system_prompt = pipeline_result.system_prompt
        except Exception as e:
            print(f"RAG pipeline error: {e}")
            system_prompt = ""

    print("=== SYSTEM PROMPT SENT TO LLM ===", flush=True)
    print(system_prompt if system_prompt else "(empty — no consent or no EMR)", flush=True)
    print("=== END SYSTEM PROMPT ===", flush=True)

    # 3. Call LLM
    try:
        use_gemini = False
        use_medgemma = False
        if request.model and request.model.startswith("gemini"):
            use_gemini = True
        elif request.model and request.model == "medgemma":
            use_medgemma = True

        # Translation (Input)
        target_lang = request.language or "en-IN"
        llm_input_message = request.message

        if target_lang != "en-IN":
            print(f"Translating input from {target_lang} to en-IN...")
            translated_input = sarvam_service.translate(request.message, target_lang, "en-IN")
            if translated_input:
                llm_input_message = translated_input

        # Build raw history (exclude current user message — services add it separately)
        raw_history_for_llm = [
            {"role": m.role, "content": m.content}
            for m in session.messages[:-1]
        ]

        # 4. Call the appropriate LLM service (all now accept compacted_summary + emr_consent + system_prompt)
        if use_gemini:
            result = await gemini_service.chat(
                llm_input_message,
                session.patient_id,
                history=raw_history_for_llm,
                compacted_summary=session.compacted_summary,
                emr_consent=effective_emr_consent,
                system_prompt=system_prompt,
            )
        elif use_medgemma:
            service = get_medgemma_service()
            result = await service.chat(
                llm_input_message,
                session.patient_id,
                history=raw_history_for_llm,
                compacted_summary=session.compacted_summary,
                emr_consent=effective_emr_consent,
                system_prompt=system_prompt,
            )
        else:
            service = get_hf_service()
            result = await service.chat(
                llm_input_message,
                session.patient_id,
                history=raw_history_for_llm,
                compacted_summary=session.compacted_summary,
                emr_consent=effective_emr_consent,
                system_prompt=system_prompt,
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
            session_meta = session.copy()
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
        raise HTTPException(status_code=500, detail=str(e))
