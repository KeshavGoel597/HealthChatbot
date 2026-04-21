"""
Context Compaction Service
==========================
GDPR Article 5(1)(c) — Data Minimisation:
  Only the clinically relevant slice of conversation history is sent to the LLM
  on each turn, not the entire raw history.

Two compaction strategies:
  1. compact_with_gemini   — Uses Gemini to produce a clinically faithful summary
                             (preferred; used when the Gemini service is active)
  2. compact_deterministic — Rule-based keyword extractor for local HF/MedGemma models
                             (reliable, zero extra API cost)

The `build_llm_history` function assembles the final list of message dicts
to be sent to any LLM backend:
    [synthetic summary block (if exists)] + [last RECENT_TURNS_TO_KEEP raw messages]
"""

import re
from typing import List, Optional

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
MAX_TOKENS_BEFORE_COMPACT = 3000   # Trigger compaction above this estimate
RECENT_TURNS_TO_KEEP = 6          # Always preserve the last N messages verbatim
RETENTION_DAYS = 30               # GDPR Art. 5(1)(e): chat log auto-deletion window


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------
def estimate_tokens(messages: List[dict]) -> int:
    """
    Rough token count: 1 token ≈ 4 characters.
    Good enough to decide whether compaction is needed.
    """
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return total_chars // 4


# ---------------------------------------------------------------------------
# Deterministic compaction (for local HF / MedGemma models)
# ---------------------------------------------------------------------------
def compact_deterministic(old_messages: List[dict]) -> str:
    """
    Extracts clinically significant facts from old conversation turns using
    regex pattern matching. No LLM call required.

    Captures:
      - Symptoms the patient reported
      - Medications or treatments mentioned
      - Specific diagnoses or conditions brought up
      - Lab values or numbers mentioned by the patient
      - Questions that were already answered (to avoid re-answering)

    GDPR Note: Only conversation content is summarised — no EMR data is
    copied into the summary. The EMR is re-loaded fresh from the source on
    each turn.
    """
    user_texts = [m["content"] for m in old_messages if m.get("role") == "user"]
    assistant_texts = [m["content"] for m in old_messages if m.get("role") == "assistant"]

    summary_lines = []

    # ---- Patient-reported symptoms ----
    symptom_keywords = [
        "pain", "ache", "fever", "fatigue", "tired", "nausea", "vomit", "cough",
        "breathless", "dizzy", "headache", "swelling", "rash", "bleed", "weak",
        "sore", "burning", "itching", "chills", "loss of", "difficulty"
    ]
    reported_symptoms = set()
    for text in user_texts:
        lower = text.lower()
        for kw in symptom_keywords:
            if kw in lower:
                # Extract a short phrase around the keyword
                idx = lower.find(kw)
                snippet = text[max(0, idx - 10):idx + 40].strip()
                reported_symptoms.add(snippet)
    if reported_symptoms:
        summary_lines.append(
            "Patient-reported symptoms in prior conversation: "
            + "; ".join(list(reported_symptoms)[:8])
        )

    # ---- Medications / treatments the patient mentioned ----
    med_pattern = re.compile(
        r'\b(metformin|insulin|aspirin|paracetamol|ibuprofen|amoxicillin|'
        r'lisinopril|atorvastatin|omeprazole|warfarin|prednisolone|'
        r'injection|tablet|capsule|syrup|inhaler|dose|mg|medication|medicine|drug)\b',
        re.IGNORECASE
    )
    all_user_text = " ".join(user_texts)
    med_matches = set(med_pattern.findall(all_user_text))
    if med_matches:
        summary_lines.append(
            "Medications / treatments patient mentioned: " + ", ".join(list(med_matches)[:10])
        )

    # ---- Diagnoses / conditions the patient mentioned ----
    condition_pattern = re.compile(
        r'\b(diabetes|hypertension|cancer|infection|anaemia|anemia|asthma|'
        r'arthritis|depression|anxiety|thyroid|kidney|liver|heart|'
        r'stroke|cholesterol|blood pressure|sugar|glucose|bp)\b',
        re.IGNORECASE
    )
    condition_matches = set(condition_pattern.findall(all_user_text))
    if condition_matches:
        summary_lines.append(
            "Conditions / terms patient mentioned: " + ", ".join(list(condition_matches)[:8])
        )

    # ---- Topics already addressed by the assistant (avoid repetition) ----
    if assistant_texts:
        # Take first sentence of each assistant reply as a summary of topic
        topics = []
        for text in assistant_texts[:5]:
            first_sentence = re.split(r'[.!?]', text)[0].strip()
            if first_sentence:
                topics.append(first_sentence[:80])
        if topics:
            summary_lines.append(
                "Topics already addressed: " + " | ".join(topics)
            )

    if not summary_lines:
        return "Earlier conversation covered general patient questions (no clinical keywords detected)."

    return (
        "[CONVERSATION SUMMARY — earlier turns compacted for efficiency]\n"
        + "\n".join(summary_lines)
    )


# ---------------------------------------------------------------------------
# Gemini-based compaction (async — for use with GeminiService)
# ---------------------------------------------------------------------------
async def compact_with_gemini(old_messages: List[dict], gemini_client, model_name: str) -> str:
    """
    Uses Gemini to produce a clinically faithful, concise summary of the
    older conversation turns that are being compacted out of the context window.

    The prompt explicitly instructs the model to:
      - Retain ALL clinically relevant facts (symptoms, medications, dates, concerns)
      - NOT infer or add anything not stated by the patient
      - NOT copy EMR data (that is loaded fresh each turn)
      - Format as a compact clinical note, NOT a full transcript

    GDPR Note: Only what the patient *said* during the chat is summarised —
    no additional personal data is stored or inferred.
    """
    if not old_messages:
        return ""

    # Format old messages as a readable transcript for the summarisation prompt
    transcript_lines = []
    for m in old_messages:
        role_label = "Patient" if m.get("role") == "user" else "Robert (AI)"
        transcript_lines.append(f"{role_label}: {m.get('content', '')}")
    transcript = "\n".join(transcript_lines)

    compaction_prompt = f"""You are a clinical documentation assistant.
Below is a partial transcript of a medical chatbot conversation between a patient and Robert (an AI medical assistant).
Your task is to produce a concise clinical summary of this transcript for use as context in the ongoing conversation.

STRICT RULES:
1. Include ONLY information explicitly stated in the transcript. Do NOT infer, assume, or hallucinate.
2. Preserve ALL: symptoms reported, medications mentioned, dates/timeframes, specific concerns or questions raised, and any advice given.
3. Do NOT include: EMR data, diagnoses, or lab values (those are loaded separately each turn).
4. Format: A short paragraph or bullet list. Maximum 200 words.
5. Start with: "[CONVERSATION SUMMARY]"

TRANSCRIPT TO SUMMARISE:
{transcript}

CLINICAL SUMMARY:"""

    try:
        from google.genai import types
        response = await gemini_client.aio.models.generate_content(
            model=model_name,
            contents=compaction_prompt,
            config=types.GenerateContentConfig(temperature=0.1)  # Low temp for factual fidelity
        )
        summary = response.text.strip()
        # Ensure prefix is present
        if not summary.startswith("[CONVERSATION SUMMARY]"):
            summary = "[CONVERSATION SUMMARY]\n" + summary
        return summary
    except Exception as e:
        print(f"[COMPACTION] Gemini compaction failed, falling back to deterministic: {e}")
        return compact_deterministic(old_messages)


# ---------------------------------------------------------------------------
# History builder — used by all LLM services
# ---------------------------------------------------------------------------
def build_llm_history(
    all_messages: List[dict],
    compacted_summary: Optional[str] = None
) -> List[dict]:
    """
    Assembles the list of message dicts to be passed to the LLM on each turn.

    Structure:
      1. If a compacted_summary exists: inject it as a synthetic "assistant" message
         at the start (acts as a memory block the model treats as prior context)
      2. Append the last RECENT_TURNS_TO_KEEP raw messages verbatim

    The calling service appends the current user message on top of the result.
    """
    recent = all_messages[-RECENT_TURNS_TO_KEEP:] if len(all_messages) > RECENT_TURNS_TO_KEEP else all_messages

    if compacted_summary:
        # Inject summary as a synthetic prior-context block
        summary_block = {
            "role": "assistant",
            "content": compacted_summary
        }
        return [summary_block] + recent
    else:
        return list(recent)


# ---------------------------------------------------------------------------
# Compaction trigger check
# ---------------------------------------------------------------------------
def needs_compaction(messages: List[dict]) -> bool:
    """Returns True if the history is large enough to warrant compaction."""
    return estimate_tokens(messages) > MAX_TOKENS_BEFORE_COMPACT and len(messages) > RECENT_TURNS_TO_KEEP


def split_for_compaction(messages: List[dict]):
    """
    Splits messages into (old_turns_to_compact, recent_turns_to_keep).
    The old turns will be summarised; recent turns are kept verbatim.
    """
    if len(messages) <= RECENT_TURNS_TO_KEEP:
        return [], messages
    old = messages[:-RECENT_TURNS_TO_KEEP]
    recent = messages[-RECENT_TURNS_TO_KEEP:]
    return old, recent
