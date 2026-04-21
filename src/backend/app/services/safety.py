# backend/app/services/safety.py
import re
from dataclasses import dataclass

CRISIS_RESPONSE = (
    "I'm very concerned about what you've shared. "
    "Please reach out to a crisis helpline immediately:\n\n"
    "• iCall (India): 9152987821\n"
    "• Vandrevala Foundation: 1860-2662-345 (24/7)\n"
    "• AASRA: 9820466627\n\n"
    "You are not alone and help is available right now. "
    "Please contact a mental health professional or go to your nearest emergency room "
    "if you are in immediate danger.\n\n"
    "— I am Robert, an AI assistant. This is not a substitute for professional mental health support."
)

_PATTERNS = [
    r"\bwant\s+to\s+die\b",
    r"\bwant\s+to\s+kill\s+(my)?self\b",
    r"\bwant\s+to\s+hurt\s+(my)?self\b",
    r"\bsuicid(e|al)\b",
    r"\bend\s+(my\s+)?life\b",
    r"\bself[\s\-]?harm\b",
    r"\bcut\s+(my)?self\b",
    r"\bkill\s+(my)?self\b",
    r"\btake\s+my\s+(own\s+)?life\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bbetter\s+off\s+(dead|without\s+me)\b",
    r"\bdon'?t\s+want\s+to\s+(be\s+)?alive\b",
    r"\boverdos(e|ing)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


@dataclass
class SafetyResult:
    triggered: bool
    response: str | None  # Non-None only when triggered


def check_safety(message: str) -> SafetyResult:
    """Return a crisis response if message contains self-harm indicators.

    Must be called before any LLM invocation so the model never
    processes self-harm content.
    """
    for pattern in _COMPILED:
        if pattern.search(message):
            print("[SAFETY] Self-harm guardrail triggered.")
            return SafetyResult(triggered=True, response=CRISIS_RESPONSE)
    return SafetyResult(triggered=False, response=None)
