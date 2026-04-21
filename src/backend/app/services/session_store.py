"""
Shared session I/O helpers.

Used by sessions.py and gdpr_router.py to avoid duplicating the
load/save/path-resolution logic for session JSON files.
"""

import os
import json
from typing import Optional
from app.models.chat_models import ChatSession

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SESSIONS_DIR = os.path.join(BASE_DIR, "data", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def get_session_path(session_id: str) -> str:
    return os.path.join(SESSIONS_DIR, f"{session_id}.json")


def load_session(session_id: str) -> Optional[ChatSession]:
    path = get_session_path(session_id)
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return ChatSession(**data)
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error loading session {session_id}: {e}")
        return None


def save_session(session: ChatSession):
    path = get_session_path(session.id)
    with open(path, "w") as f:
        json.dump(session.model_dump(), f, indent=2)
