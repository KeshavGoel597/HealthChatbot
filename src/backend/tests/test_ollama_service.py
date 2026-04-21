import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.services.ollama_service import OllamaService
from app.services.llm_factory import get_llm_service

@pytest.fixture
def ollama_service():
    return OllamaService(model_name="test-model:latest")

# --- TEST 1: Factory Routing ---
def test_llm_factory_routing():
    """Ensures 'medgemma' and 'ollama' strings route to the OllamaService."""
    service_med = get_llm_service("medgemma")
    service_ollama = get_llm_service("ollama")
    service_hf = get_llm_service("huggingface")
    
    assert isinstance(service_med, OllamaService)
    assert isinstance(service_ollama, OllamaService)
    # Just to prove it routes differently for others
    assert not isinstance(service_hf, OllamaService)

# --- TEST 2: Graceful Connection Failure ---
@patch("app.services.ollama_service.ollama.list")
def test_ollama_connection_failure(mock_list, ollama_service):
    """Ensures a clear runtime error is thrown if Ollama isn't running."""
    # Simulate the Ollama server being offline
    mock_list.side_effect = Exception("Connection refused")
    
    with pytest.raises(RuntimeError) as exc_info:
        ollama_service._load_model()
        
    assert "Ollama service not detected" in str(exc_info.value)
    assert "brew services start ollama" in str(exc_info.value)

# --- TEST 3: Successful Generation and Token Math ---
@pytest.mark.asyncio
@patch("app.services.ollama_service.ollama.list")
@patch("app.services.ollama_service.ollama.chat")
async def test_ollama_chat_success_and_tokens(mock_chat, mock_list, ollama_service):
    """Tests the happy path: context assembly, generation, and token counting."""
    
    # 1. Setup Mocks
    mock_list.return_value = {"models": [{"name": "test-model:latest"}]}
    mock_chat.return_value = {
        "message": {"content": "This is a mocked response from Ollama."}
    }
    
    # 2. Execute
    response = await ollama_service.chat(
        message="I have a headache.",
        patient_id="patient101",
        emr_consent=False,
        system_prompt="Test Prompt"
    )
    
    # 3. Assertions for Generation
    assert response["response"] == "This is a mocked response from Ollama."
    assert response["model_name"] == "test-model:latest"
    mock_chat.assert_called_once()
    
    # 4. Assertions for Token Math (len(words) * 2)
    # "This is a mocked response from Ollama." is 7 words. 7 * 2 = 14 output tokens.
    assert response["output_tokens"] == 14
    
    # Verify input token math is populated (exact number depends on your GDPR text length)
    assert response["input_tokens"] > 0 
    assert response["total_tokens"] == response["input_tokens"] + response["output_tokens"]

# --- TEST 4: EMR and Consent Injection ---
@pytest.mark.asyncio
@patch("app.services.ollama_service.ollama.list")
@patch("app.services.ollama_service.ollama.chat")
@patch("app.services.ollama_service.OllamaService._build_emr_section")
async def test_ollama_emr_consent_injection(mock_build_emr, mock_chat, mock_list, ollama_service):
    """Ensures EMR context is retrieved when consent is True."""
    
    mock_list.return_value = {}
    mock_chat.return_value = {"message": {"content": "Mocked"}}
    mock_build_emr.return_value = ("Mocked EMR Section: Patient has History of Migraines", ["Condition"])
    
    await ollama_service.chat(
        message="What is my history?",
        patient_id="patient101",
        emr_consent=True
    )
    
    # Verify the build EMR function was called securely with the right ID and consent flag
    mock_build_emr.assert_called_once_with(
        system_prompt="",
        emr_consent=True,
        patient_id="patient101",
        consent_prefix="PATIENT MEDICAL RECORDS (consented, read-only):\n"
    )