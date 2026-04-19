"""
CUI search: medical terms → relevant UMLS CUIs.

Uses SapBERT semantic embeddings to map a list of normalized medical terms
(e.g. ["headache"]) to UMLS Concept Unique Identifiers.

The caller (pipeline.py) extracts terms from the query before calling this
function. Keeping extraction separate lets the pipeline route broad queries
to a category filter without touching CUI search at all.
"""

from __future__ import annotations

from app.services.rag.embeddings import EmbeddingIndex


def find_cuis(
    terms: list[str],
    index: EmbeddingIndex,
    top_k: int = 10,
    threshold: float = 0.7,
) -> list[dict]:
    """Find CUIs semantically matching a list of medical terms.

    Each term is encoded and searched independently; results are merged
    by CUI, keeping the highest score per CUI.

    Args:
        terms: Normalized medical terms (e.g. ["headache", "nausea"]).
               Pass an empty list to get an empty result without touching FAISS.
        index: Loaded EmbeddingIndex instance (SapBERT + FAISS).
        top_k: Max FAISS candidates to retrieve per term.
        threshold: Minimum cosine similarity to include.

    Returns:
        List of dicts: [{"cui": "C0018681", "name": "Headache", "score": 0.856}, ...]
        Sorted by score descending. Empty list if nothing above threshold.
    """
    best: dict[str, dict] = {}

    for term in terms:
        query_vec = index.encode(term)
        results = index.search(query_vec, top_k)

        for cui, score in results:
            if score < threshold:
                break  # sorted descending, no need to continue
            if cui not in best or score > best[cui]["score"]:
                best[cui] = {
                    "cui": cui,
                    "name": index.get_name(cui),
                    "score": round(score, 4),
                }

    return sorted(best.values(), key=lambda r: r["score"], reverse=True)
