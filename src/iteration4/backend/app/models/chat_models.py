from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ChatMessage(BaseModel):
    message: str
    patient_id: Optional[str] = "patient101" # Default to patient101 for now
    context: Optional[str] = None # Optional context override
    model: Optional[str] = None # Optional model selection

class ChatResponse(BaseModel):
    response: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    model_name: str

class SessionMessage(BaseModel):
    role: str # "user" or "assistant"
    content: str
    timestamp: str

class ChatSession(BaseModel):
    id: str
    patient_id: str
    title: str = "New Chat"
    created_at: str
    messages: List[SessionMessage] = []
