# backend/tests/test_safety.py
import pytest
from app.services.safety import check_safety, CRISIS_RESPONSE


@pytest.mark.parametrize("message", [
    "I want to die",
    "I want to kill myself",
    "I want to hurt myself",
    "i want to hurt myself badly",
    "I am suicidal",
    "I want to end my life",
    "I'm thinking about self-harm",
    "I want to cut myself",
    "I want to take my own life",
    "I have no reason to live",
    "everyone would be better off without me",
    "I don't want to be alive anymore",
    "I am thinking about overdosing",
    "WANT TO DIE",
    "Want To Kill Myself",
])
def test_self_harm_triggers_guardrail(message):
    result = check_safety(message)
    assert result.triggered is True
    assert result.response == CRISIS_RESPONSE


@pytest.mark.parametrize("message", [
    "What medications am I taking?",
    "I have a headache",
    "What are my lab results?",
    "My blood pressure is high",
    "I feel tired all the time",
    "Can you explain my diagnosis?",
    "I want to understand my condition better",
    "I want to live a healthier life",
    "My plant is dying",
])
def test_safe_messages_do_not_trigger(message):
    result = check_safety(message)
    assert result.triggered is False
    assert result.response is None


def test_crisis_response_contains_helpline_numbers():
    result = check_safety("I want to die")
    assert "9152987821" in result.response
    assert "1860-2662-345" in result.response
    assert "9820466627" in result.response
