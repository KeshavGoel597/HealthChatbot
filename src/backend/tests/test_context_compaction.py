# backend/tests/test_context_compaction.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.context_compaction import (
    estimate_tokens,
    compact_deterministic,
    compact_with_gemini,
    build_llm_history,
    needs_compaction,
    split_for_compaction,
    MAX_TOKENS_BEFORE_COMPACT,
    RECENT_TURNS_TO_KEEP
)

def test_estimate_tokens():
    messages = [{"content": "abcd efg hi"}, {"content": "1234"}]
    assert estimate_tokens(messages) == 15 // 4

def test_compact_deterministic():
    messages = [
        {"role": "user", "content": "I have been experiencing a bad headache and fever."},
        {"role": "assistant", "content": "I see. Let's look into that."},
        {"role": "user", "content": "Also taking ibuprofen and metformin."},
        {"role": "user", "content": "My diabetes is acting up."}
    ]
    summary = compact_deterministic(messages)
    assert "CONVERSATION SUMMARY" in summary
    assert "headache" in summary.lower() or "fever" in summary.lower()
    assert "ibuprofen" in summary.lower() or "metformin" in summary.lower()
    assert "diabetes" in summary.lower()
    assert "I see" in summary

def test_compact_deterministic_no_keywords():
    messages = [{"role": "user", "content": "Hello there."}]
    summary = compact_deterministic(messages)
    assert "no clinical keywords detected" in summary

@pytest.mark.asyncio
async def test_compact_with_gemini_success():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Patient has a headache."
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    messages = [{"role": "user", "content": "I have a headache"}]
    summary = await compact_with_gemini(messages, mock_client, "gemini-model")
    assert "[CONVERSATION SUMMARY]" in summary
    assert "Patient has a headache." in summary

@pytest.mark.asyncio
async def test_compact_with_gemini_fallback():
    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=Exception("API error"))
    
    messages = [{"role": "user", "content": "I have a fever."}]
    summary = await compact_with_gemini(messages, mock_client, "gemini-model")
    assert "CONVERSATION SUMMARY" in summary
    assert "fever" in summary.lower()

def test_build_llm_history():
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
    
    history = build_llm_history(messages)
    assert len(history) == min(10, RECENT_TURNS_TO_KEEP)
    assert history[-1]["content"] == "msg 9"
    
    history_with_summary = build_llm_history(messages, "My Summary")
    assert len(history_with_summary) == min(10, RECENT_TURNS_TO_KEEP) + 1
    assert history_with_summary[0]["role"] == "assistant"
    assert history_with_summary[0]["content"] == "My Summary"

def test_needs_compaction():
    short_msg = [{"content": "a"}] * (RECENT_TURNS_TO_KEEP + 1)
    assert not needs_compaction(short_msg)
    
    long_msg = [{"content": "a" * (MAX_TOKENS_BEFORE_COMPACT * 5)}] * (RECENT_TURNS_TO_KEEP + 1)
    assert needs_compaction(long_msg)

def test_split_for_compaction():
    messages = [{"content": str(i)} for i in range(10)]
    old, recent = split_for_compaction(messages)
    
    expected_recent_len = min(10, RECENT_TURNS_TO_KEEP)
    assert len(recent) == expected_recent_len
    assert len(old) == 10 - expected_recent_len
    if RECENT_TURNS_TO_KEEP == 6:
        assert old[-1]["content"] == "3"
        assert recent[0]["content"] == "4"
