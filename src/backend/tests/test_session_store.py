# backend/tests/test_session_store.py
import pytest
import os
from unittest.mock import patch, mock_open
from app.models.chat_models import ChatSession
from app.services.session_store import load_session, save_session, get_session_path

def test_get_session_path():
    path = get_session_path("test-123")
    assert path.endswith("test-123.json")
    assert "sessions" in path

@patch("app.services.session_store.open", new_callable=mock_open, read_data='{"id": "test-123", "patient_id": "p1", "messages": [], "emr_consent": true, "compacted_summary": null, "created_at": "2024-01-01T00:00:00Z", "updated_at": "2024-01-01T00:00:00Z"}')
@patch("os.path.exists", return_value=True)
def test_load_session_success(mock_exists, mock_file):
    session = load_session("test-123")
    assert session is not None
    assert session.id == "test-123"
    assert session.patient_id == "p1"

@patch("app.services.session_store.open", side_effect=FileNotFoundError)
def test_load_session_not_found(mock_file):
    session = load_session("nonexistent")
    assert session is None

@patch("app.services.session_store.open", side_effect=Exception("Disk error"))
def test_load_session_exception(mock_file):
    session = load_session("error")
    assert session is None

@patch("app.services.session_store.open", new_callable=mock_open)
def test_save_session(mock_file):
    session = ChatSession(id="test-123", patient_id="p1")
    save_session(session)
    mock_file.assert_called_once()
    handle = mock_file()
    assert handle.write.call_count > 0
