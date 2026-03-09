"""
GDPR Router
===========
Implements user-facing GDPR rights as REST endpoints.

Article Coverage:
  Art. 5(1)(a,e) — POST /gdpr/consent/{session_id}   Update consent flags
  Art. 15        — GET  /gdpr/evidence/{session_id}   Right to Access (evidence panel)
  Art. 17        — DELETE /gdpr/sessions/{session_id} Right to Erasure (single session)
  Art. 17        — DELETE /gdpr/patient/{patient_id}  Right to Erasure (all sessions)
"""

import os
import json
from fastapi import APIRouter, HTTPException
from app.models.chat_models import GDPRConsentUpdate, ChatSession
from typing import List
from datetime import datetime

router = APIRouter(prefix="/gdpr", tags=["GDPR"])

# Resolve sessions directory (same as sessions.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")


def _get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def _load_session(session_id: str):
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


# ---------------------------------------------------------------------------
# Art. 5(1)(a,e) — Consent update
# ---------------------------------------------------------------------------
@router.post("/consent/{session_id}", summary="Update GDPR consent flags for a session")
async def update_consent(session_id: str, consent: GDPRConsentUpdate):
    """
    Update the patient's EMR access consent and history storage consent
    for a given session. GDPR Articles 5(1)(a) and 5(1)(e).

    If store_history_consent is revoked (set to False), existing messages in
    the session are purged immediately to comply with the right to not be stored.
    """
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    prev_store_consent = session.store_history_consent
    session.emr_consent = consent.emr_consent
    session.store_history_consent = consent.store_history_consent

    # If storage consent was just revoked, purge existing messages (GDPR Art. 17)
    if prev_store_consent and not consent.store_history_consent:
        session.messages = []
        session.compacted_summary = None
        print(f"[GDPR] Storage consent revoked for session {session_id}. Messages purged.")

    _save_session(session)
    return {
        "gdpr_action": "consent_updated",
        "gdpr_article": "Art. 5(1)(a) + Art. 5(1)(e)",
        "session_id": session_id,
        "emr_consent": session.emr_consent,
        "store_history_consent": session.store_history_consent,
        "messages_purged": prev_store_consent and not consent.store_history_consent,
    }


# ---------------------------------------------------------------------------
# Art. 15 — Right to Access (evidence panel)
# ---------------------------------------------------------------------------
@router.get("/evidence/{session_id}", summary="Get EMR evidence used in this session (Art. 15)")
async def get_evidence(session_id: str):
    """
    Returns the EMR fields referenced during this chat session.
    GDPR Article 15 — Right of Access.

    Patients and clinicians can use this to see exactly what data from the
    EMR influenced the AI's reasoning in each message.
    """
    session = _load_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Aggregate all evidence fields across all assistant messages
    all_fields = set()
    evidence_per_message = []
    for msg in session.messages:
        if msg.role == "assistant" and msg.emr_fields_used:
            for field in msg.emr_fields_used:
                all_fields.add(field)
            evidence_per_message.append({
                "timestamp": msg.timestamp,
                "emr_fields_used": msg.emr_fields_used,
            })

    return {
        "gdpr_article": "Art. 15 — Right of Access",
        "session_id": session_id,
        "patient_id": session.patient_id,
        "emr_consent_given": session.emr_consent,
        "all_emr_fields_accessed": sorted(list(all_fields)),
        "evidence_timeline": evidence_per_message,
        "note": (
            "These are the EMR data categories that influenced the AI's responses. "
            "No raw EMR data is stored in chat logs — only these field labels."
        ),
    }


# ---------------------------------------------------------------------------
# Art. 17 — Right to Erasure (single session)
# ---------------------------------------------------------------------------
@router.delete("/sessions/{session_id}", summary="Delete a chat session (Art. 17 — Right to Erasure)")
async def delete_session(session_id: str):
    """
    Permanently delete a single chat session (conversation history + metadata).
    GDPR Article 17 — Right to Erasure ("Right to be Forgotten").

    The patient's EMR data is NOT affected — only the conversation log is deleted.
    """
    path = _get_session_path(session_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        os.remove(path)
        print(f"[GDPR Art.17] Session {session_id} deleted on erasure request.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")

    return {
        "gdpr_action": "session_deleted",
        "gdpr_article": "Art. 17 — Right to Erasure",
        "session_id": session_id,
        "note": "This chat session has been permanently deleted. EMR records are unaffected.",
        "deleted_at": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Art. 17 — Right to Erasure (all sessions for a patient)
# ---------------------------------------------------------------------------
@router.delete("/patient/{patient_id}", summary="Delete ALL chat sessions for a patient (Art. 17)")
async def delete_all_patient_sessions(patient_id: str):
    """
    Permanently delete all chat sessions belonging to a given patient.
    GDPR Article 17 — Right to Erasure.

    The patient's EMR data is NOT affected — only the conversation logs are deleted.
    """
    if not os.path.exists(SESSIONS_DIR):
        return {
            "gdpr_action": "no_sessions_found",
            "gdpr_article": "Art. 17 — Right to Erasure",
            "patient_id": patient_id,
            "sessions_deleted": 0,
        }

    deleted = []
    errors = []

    for filename in os.listdir(SESSIONS_DIR):
        if not filename.endswith(".json"):
            continue
        session_id = filename.replace(".json", "")
        session = _load_session(session_id)
        if session and session.patient_id == patient_id:
            try:
                os.remove(_get_session_path(session_id))
                deleted.append(session_id)
                print(f"[GDPR Art.17] Session {session_id} deleted for patient {patient_id}.")
            except Exception as e:
                errors.append({"session_id": session_id, "error": str(e)})

    return {
        "gdpr_action": "all_patient_sessions_deleted",
        "gdpr_article": "Art. 17 — Right to Erasure",
        "patient_id": patient_id,
        "sessions_deleted": len(deleted),
        "deleted_session_ids": deleted,
        "errors": errors,
        "note": "All chat sessions for this patient have been permanently deleted. EMR records are unaffected.",
        "deleted_at": datetime.now().isoformat(),
    }
