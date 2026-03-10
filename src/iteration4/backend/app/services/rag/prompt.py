"""
Phase 4: Assemble a structured prompt from matched EMR sections.

Takes the query and matched sections from Phase 3 and formats them
into a clean system prompt for the LLM. Groups sections by category
so the model sees organized clinical context.

No torch/faiss/embedding dependencies — pure string formatting.
"""

from __future__ import annotations

from app.services.rag.emr_match import MatchedSection


# ── Category display order and labels ─────────────────────────────────

_CATEGORY_ORDER = [
    "diagnosis",
    "symptom",
    "comorbidity",
    "lab",
    "vitals",
    "medicine",
    "history",
    "comment",
    "recommended_labs",
    "demographics",
    "discharge",
]

_CATEGORY_LABELS = {
    "diagnosis": "Diagnoses",
    "symptom": "Symptoms",
    "comorbidity": "Comorbidities",
    "lab": "Lab Results",
    "vitals": "Vitals",
    "medicine": "Medications",
    "history": "Patient History",
    "comment": "Clinical Notes",
    "recommended_labs": "Recommended Labs",
    "demographics": "Demographics",
    "discharge": "Discharge Summary",
}


# ── Prompt assembly ───────────────────────────────────────────────────

def _format_section(match: MatchedSection) -> str:
    """Format a single matched section as a concise line."""
    s = match.section
    parts = [f"- {s.text}"]

    if s.value and s.category in ("lab", "vitals"):
        parts.append(f"({s.value})")
    if s.date:
        parts.append(f"[{s.date}]")

    return " ".join(parts)


def assemble_context(matches: list[MatchedSection]) -> str:
    """Format matched EMR sections into grouped clinical context.

    Groups sections by category in clinical priority order,
    producing a clean block of text suitable for embedding in
    a system prompt.

    Args:
        matches: Matched sections from Phase 3 (emr_match).

    Returns:
        Formatted string of clinical context, or empty string
        if no matches.
    """
    if not matches:
        return ""

    # Group by category
    groups: dict[str, list[MatchedSection]] = {}
    for m in matches:
        cat = m.section.category
        groups.setdefault(cat, []).append(m)

    # Build output in priority order
    lines: list[str] = []
    for cat in _CATEGORY_ORDER:
        if cat not in groups:
            continue
        label = _CATEGORY_LABELS.get(cat, cat.title())
        lines.append(f"{label}:")
        for m in groups[cat]:
            lines.append(_format_section(m))
        lines.append("")  # blank line between groups

    return "\n".join(lines).strip()


def assemble_prompt(
    query: str,
    matches: list[MatchedSection],
    patient_id: str = "",
    context: str | None = None,
) -> str:
    """Build the full system instruction for the LLM.

    Combines the assistant persona with RAG-retrieved clinical
    context. If no sections matched, falls back to a generic
    instruction without clinical data.

    Args:
        query: The user's original question.
        matches: Matched EMR sections from Phase 3.
        patient_id: Optional patient identifier for the prompt.
        context: Pre-built clinical context string. If None,
                 assembled from matches.

    Returns:
        Complete system instruction string.
    """
    if context is None:
        context = assemble_context(matches)

    base = (
        "You are Robert, a helpful AI medical assistant for patients. "
        "Your goal is to explain their medical records in simple, "
        "easy-to-understand language. Avoid medical jargon where possible, "
        "or explain it clearly. Always be empathetic and accurate."
    )

    if not context:
        return (
            f"{base}\n\n"
            "No specific clinical records matched this query. "
            "Answer based on general medical knowledge and advise "
            "the patient to consult their doctor for specifics."
        )

    patient_label = f" for {patient_id}" if patient_id else ""

    return (
        f"{base}\n\n"
        f"=== PATIENT CLINICAL RECORD{patient_label.upper()} ===\n"
        f"{context}\n"
        f"=== END CLINICAL RECORD ===\n\n"
        "IMPORTANT: The clinical record above is the patient's actual medical data. "
        "Always ground your response in these records first — reference specific "
        "diagnoses, medications, lab values, or symptoms from the record when they "
        "are relevant to the question. "
        "If the records contain information that answers the patient's question, "
        "lead with that information and explain it clearly.\n\n"
        "If the clinical record does not cover the patient's question, you must"
        "supplement with general medical knowledge. THIS IS VERY IMPORTANT. You must at least answer the question"
        "but clearly indicate when you"
        "are doing so and advise the patient to consult their doctor for specifics."
    )
