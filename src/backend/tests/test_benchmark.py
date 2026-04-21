# backend/tests/test_benchmark.py
import pytest
from unittest.mock import MagicMock, AsyncMock
from scripts.benchmark import _call_gemini

@pytest.mark.asyncio
async def test_call_gemini():
    mock_gemini = MagicMock()
    mock_gemini.chat = AsyncMock(return_value={
        "input_tokens": 10,
        "output_tokens": 20,
        "total_tokens": 30,
        "response": "Hello"
    })
    
    result = await _call_gemini(mock_gemini, "query", "prompt", "p1", True)
    assert result["input_tokens"] == 10
    assert result["output_tokens"] == 20
    assert result["total_tokens"] == 30
    assert result["response"] == "Hello"
    assert "latency_ms" in result
