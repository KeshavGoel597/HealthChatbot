# backend/tests/test_gemini_service.py
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.services.gemini_service import GeminiService, build_emr_system_prompt

@pytest.fixture
def mock_gemini_client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    with patch("app.services.gemini_service.genai.Client") as mock_client_class:
        mock_client = MagicMock()
        mock_client.aio.models.generate_content = AsyncMock()
        mock_client_class.return_value = mock_client
        yield mock_client

@pytest.mark.asyncio
async def test_extract_clinical_data_success(mock_gemini_client):
    mock_response = MagicMock()
    mock_response.text = '[{"type": "diagnosis", "value": "Hypertension"}]'
    mock_gemini_client.aio.models.generate_content.return_value = mock_response
    
    service = GeminiService()
    structured, fields = await service.extract_clinical_data("patient has hypertension")
    
    assert "diagnosis" in structured
    assert "Medical Diagnoses" in fields

@pytest.mark.asyncio
async def test_extract_clinical_data_error(mock_gemini_client):
    mock_gemini_client.aio.models.generate_content.side_effect = Exception("API error")
    
    service = GeminiService()
    structured, fields = await service.extract_clinical_data("patient has hypertension")
    
    assert structured == "[]"
    assert fields == []

@pytest.mark.asyncio
@patch("app.services.gemini_service.load_patient_data", return_value="raw patient data")
async def test_chat_success(mock_load_data, mock_gemini_client):
    mock_response = MagicMock()
    mock_response.text = "Hello patient."
    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 20
    mock_response.usage_metadata.total_token_count = 30
    mock_gemini_client.aio.models.generate_content.return_value = mock_response
    
    service = GeminiService()
    # Also patch extract_clinical_data to return a simple mock
    service.extract_clinical_data = AsyncMock(return_value=("[{'type': 'diagnosis'}]", ["Medical Diagnoses"]))
    
    result = await service.chat(
        message="Hello",
        patient_id="p1",
        emr_consent=True
    )
    
    assert result["response"] == "Hello patient."
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20
    assert result["total_tokens"] == 30
    assert "Medical Diagnoses" in result["emr_fields_used"]

@pytest.mark.asyncio
async def test_chat_compaction_triggered(mock_gemini_client):
    mock_response = MagicMock()
    mock_response.text = "Response."
    mock_gemini_client.aio.models.generate_content.return_value = mock_response
    
    service = GeminiService()
    service.compact_history = AsyncMock(return_value="[CONVERSATION SUMMARY] patient is well.")
    
    with patch("app.services.gemini_service.needs_compaction", return_value=True):
        result = await service.chat(
            message="Hi",
            history=[{"role": "user", "content": "hi"} for _ in range(10)]
        )
        assert result["was_compacted"] is True
        assert result["new_compacted_summary"] == "[CONVERSATION SUMMARY] patient is well."

def test_build_emr_system_prompt():
    prompt = build_emr_system_prompt("Clinical Data")
    assert "PATIENT EMR CONTEXT" in prompt
    assert "Clinical Data" in prompt
    assert "reads records" in prompt
