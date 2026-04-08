"""
CUI search: natural language query → relevant UMLS CUIs.

Uses SapBERT semantic embeddings to map free-text queries
(e.g. "my head hurts") to UMLS Concept Unique Identifiers
(e.g. C0018681 = Headache).

When a TermExtractor is provided, the raw query is first split
into normalized medical terms (via Qwen LLM), then each term is
searched independently and results are merged.  This dramatically
improves recall for conversational queries.
"""

from __future__ import annotations

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.term_extractor import TermExtractor


def find_cuis(
    query: str,
    index: EmbeddingIndex,
    top_k: int = 10,
    threshold: float = 0.7,
    *,
    extractor: TermExtractor,
) -> list[dict]:
    """Find CUIs semantically matching a natural language query.

    The query is first decomposed into normalized medical terms
    via the TermExtractor (Qwen LLM), e.g. "I have a headache,
    what should I do?" → ["headache"].  Each term is searched
    individually and results are merged by CUI, keeping the
    highest score per CUI.

    Args:
        query: Free-text input (e.g. "my head hurts")
        index: Loaded EmbeddingIndex instance
        top_k: Max candidates to retrieve from FAISS
        threshold: Minimum cosine similarity to include
        extractor: TermExtractor for pre-processing queries

    Returns:
        List of dicts: [{"cui": "C0018681", "name": "Headache", "score": 0.856}, ...]
        Sorted by score descending. Empty list if nothing above threshold.
    """
    terms = extractor.extract(query)

    best: dict[str, dict] = {}  # cui → result dict (keep highest score)

    for term in terms:
        query_vec = index.encode(term)
        results = index.search(query_vec, top_k)

        for cui, score in results:
            if score < threshold:
                continue
            if cui not in best or score > best[cui]["score"]:
                best[cui] = {
                    "cui": cui,
                    "name": index.get_name(cui),
                    "score": round(score, 4),
                }
    return sorted(best.values(), key=lambda r: r["score"], reverse=True)
