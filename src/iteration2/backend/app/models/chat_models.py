from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    message: str
    patient_id: Optional[str] = "patient101"  # Default to patient101 for now
    context: Optional[str] = None              # Optional context override
    model: Optional[str] = None                # Optional model selection
    language: Optional[str] = "en-IN"          # Target language: en-IN, hi-IN, ta-IN, etc.
    audio_requested: Optional[bool] = False
    emr_consent: Optional[bool] = False        # GDPR Art. 5(1)(a) — explicit consent to access EMR
    store_history_consent: Optional[bool] = False  # GDPR Art. 5(1)(e) — consent to persist chat

class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str
    audio_content: Optional[str] = None        # Base64 encoded audio
    language: Optional[str] = "en-IN"
    emr_fields_used: Optional[List[str]] = []  # GDPR Art. 15 — evidence transparency
    was_compacted: Optional[bool] = False       # Whether context compaction ran this turn

class SessionMessage(BaseModel):
    role: str                                   # "user" or "assistant"
    content: str
    timestamp: str
    emr_fields_used: Optional[List[str]] = []  # GDPR Art. 15 — per-message evidence log

class ChatSession(BaseModel):
    id: str
    patient_id: str
    title: str = "New Chat"
    created_at: str
    messages: List[SessionMessage] = []
    # Context compaction
    compacted_summary: Optional[str] = None    # Rolling clinical summary of compacted turns
    # GDPR fields
    emr_consent: bool = False                  # GDPR Art. 5(1)(a)
    store_history_consent: bool = False        # GDPR Art. 5(1)(e)
    expires_at: Optional[str] = None           # GDPR Art. 5(1)(e) — auto-deletion datetime

class GDPRConsentUpdate(BaseModel):
    emr_consent: bool
    store_history_consent: bool

class SessionCreateRequest(BaseModel):
    emr_consent: Optional[bool] = False
    store_history_consent: Optional[bool] = False
