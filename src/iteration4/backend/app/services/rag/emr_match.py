"""
Phase 3: Match expanded CUIs against EMR sections.

For each EMR section, encodes its text with SapBERT and finds
the nearest CUIs in the embedding index. If any of those CUIs
overlap with the expanded CUI set, the section is considered relevant.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.emr import EMRSection


@dataclass
class MatchedSection:
    """An EMR section that matched one or more expanded CUIs."""

    section: EMRSection
    matched_cuis: list[str]  # CUIs from the expanded set that matched
    best_score: float  # Highest cosine similarity among matches


def _clean_for_encoding(text: str, category: str) -> str:
    """Clean EMR section text for SapBERT encoding.

    Handles:
        - Parenthetical content: "DIABETES (SUGAR)" → "diabetes sugar"
        - Dose/strength patterns: "METFORMIN 500MG" → "metformin"
        - Special chars: "Creatinine- Serum" → "creatinine serum"
    """
    # Strip parentheses but keep content
    text = re.sub(r"[()]", " ", text)

    # For medicines, strip dose patterns (numbers + units)
    if category == "medicine":
        text = re.sub(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?\s*(?:MG|ML|MCG|GM|IU|%)\b", "", text, flags=re.IGNORECASE)

    # Collapse multiple spaces, strip leading/trailing
    text = re.sub(r"[-/]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_sections(
    sections: list[EMRSection],
    expanded_cuis: set[str],
    index: EmbeddingIndex,
    top_k: int = 20,
    threshold: float = 0.5,
) -> list[MatchedSection]:
    """Match EMR sections against an expanded CUI set.

    For each section:
      1. Encode section text with SapBERT
      2. Search FAISS index for top-k nearest CUIs
      3. Keep CUIs above threshold that are also in expanded_cuis
      4. If any match, include the section in results

    Args:
        sections: EMR sections to search through.
        expanded_cuis: Set of CUI strings from Phase 2 expansion.
        index: Loaded EmbeddingIndex (SapBERT + FAISS).
        top_k: FAISS candidates per section (wider = more recall).
        threshold: Min cosine similarity for a CUI match.

    Returns:
        List of MatchedSection, sorted by best_score descending.
    """
    results: list[MatchedSection] = []

    # Demographics are too short / non-medical to match via embeddings;
    # include them as-is so they reach prompt assembly.
    passthrough_categories = {"demographics"}

    for section in sections:
        if section.category in passthrough_categories:
            results.append(
                MatchedSection(section=section, matched_cuis=[], best_score=1.0)
            )
            continue

        text = _clean_for_encoding(section.text, section.category)
        if not text or len(text) < 2:
            continue

        # Encode cleaned section text → search FAISS
        vec = index.encode(text)
        candidates = index.search(vec, top_k)

        # Intersect with expanded CUI set
        matched = []
        best = 0.0
        for cui, score in candidates:
            if score < threshold:
                break  # sorted descending, no need to continue
            if cui in expanded_cuis:
                matched.append(cui)
                best = max(best, score)

        if matched:
            results.append(
                MatchedSection(
                    section=section,
                    matched_cuis=matched,
                    best_score=best,
                )
            )

    # Sort by best match score
    results.sort(key=lambda m: m.best_score, reverse=True)
    return results
