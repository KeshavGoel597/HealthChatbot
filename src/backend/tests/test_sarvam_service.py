# backend/tests/test_sarvam_service.py
import pytest
from unittest.mock import patch, MagicMock
from app.services.sarvam_service import SarvamService, _split_text_into_chunks

def test_split_text_into_chunks():
    text = "This is a sentence. This is another sentence! And a third one."
    chunks = _split_text_into_chunks(text, max_chars=30)
    assert len(chunks) > 1
    assert "This is a sentence." in chunks[0]

@patch("app.services.sarvam_service.os.getenv", return_value="test-api-key")
def test_translate_same_lang(mock_env):
    service = SarvamService()
    result = service.translate("Hello", "en-IN", "en-IN")
    assert result == "Hello"

@patch("app.services.sarvam_service.os.getenv", return_value="test-api-key")
@patch("app.services.sarvam_service.requests.post")
def test_translate_diff_lang_success(mock_post, mock_env):
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"translated_text": "Namaste"}
    mock_post.return_value = mock_response

    service = SarvamService()
    result = service.translate("Hello", "en-IN", "hi-IN")
    assert result == "Namaste"
    mock_post.assert_called_once()

@patch("app.services.sarvam_service.os.getenv", return_value=None)
def test_translate_no_api_key(mock_env):
    service = SarvamService()
    result = service.translate("Hello", "en-IN", "hi-IN")
    assert result == "Hello"

@patch("app.services.sarvam_service.os.getenv", return_value="test-api-key")
@patch("app.services.sarvam_service.requests.post")
def test_text_to_speech_success(mock_post, mock_env):
    mock_response = MagicMock()
    mock_response.ok = True
    # mock a simple base64 encoded string
    mock_response.json.return_value = {"audios": ["YXVkaW9fZGF0YQ=="]}
    mock_post.return_value = mock_response

    service = SarvamService()
    result = service.text_to_speech("Hello", "en-IN")
    assert result == "YXVkaW9fZGF0YQ=="
    mock_post.assert_called_once()

@patch("app.services.sarvam_service.os.getenv", return_value="test-api-key")
@patch("app.services.sarvam_service.requests.post")
def test_speech_to_text_success(mock_post, mock_env):
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.json.return_value = {"transcript": "Hello there"}
    mock_post.return_value = mock_response

    service = SarvamService()
    result = service.speech_to_text(b"audio bytes", "en-IN")
    assert result == "Hello there"
    mock_post.assert_called_once()
