"""
Phase 2: Knowledge graph expansion via BFS.

Given seed CUIs (from Phase 1), expand outward through the SNOMED
knowledge graph to discover related medical concepts.

Uses a relation allowlist to keep only diagnostically relevant edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.rag.graph import KnowledgeGraph


# ── Diagnostically relevant relation types ────────────────────────────
# 18 out of 108 physician-curated relations. Excludes noise like
# 'isa' (923K edges), 'self' (407K), drug formulation, measurement metadata.

DIAGNOSTIC_RELATIONS: set[str] = {
    # Causal / etiological
    "cause of",
    "due to",
    "causative agent of",
    "has causative agent",
    # Clinical findings
    "has associated finding",
    "associated finding of",
    "has definitional manifestation",
    "definitional manifestation of",
    # Anatomical
    "has finding site",
    "finding site of",
    # Pathological
    "has pathological process",
    "pathological process of",
    "has associated morphology",
    "associated morphology of",
    # Temporal / clinical course
    "has clinical course",
    "clinical course of",
    "occurs after",
    "occurs before",
}


@dataclass
class ExpandedCUI:
    """A CUI discovered during graph expansion."""

    cui: str
    hop: int  # 0 = seed, 1 = hop-1, 2 = hop-2, ...
    relation: str = ""  # edge label that reached this CUI ("" for seeds)
    source: str = ""  # CUI we came from ("" for seeds)


def expand_cuis(
    seed_cuis: list[str],
    graph: KnowledgeGraph,
    depth: int = 2,
    allowed_relations: set[str] | None = DIAGNOSTIC_RELATIONS,
    max_per_hop: int | None = None,
) -> list[ExpandedCUI]:
    """BFS expansion over the knowledge graph.

    Args:
        seed_cuis: Starting CUI codes (output of Phase 1).
        graph: Loaded KnowledgeGraph instance.
        depth: Number of hops to expand (default 2).
        allowed_relations: Relation types to traverse. None = all.
                           Defaults to DIAGNOSTIC_RELATIONS.
        max_per_hop: If set, cap neighbors per source node per hop.
                     Prevents hub nodes from dominating.

    Returns:
        List of ExpandedCUI, ordered by (hop, insertion order).
        Each CUI appears once at its minimum hop distance.
    """
    visited: set[str] = set()
    results: list[ExpandedCUI] = []

    # ── Hop 0: seeds ──
    for cui in seed_cuis:
        if cui not in visited:
            visited.add(cui)
            results.append(ExpandedCUI(cui=cui, hop=0))

    # ── Hop 1..depth ──
    frontier = [r.cui for r in results]  # CUIs to expand from

    for hop in range(1, depth + 1):
        next_frontier: list[str] = []

        for src in frontier:
            neighbors = graph.neighbors(src, allowed_relations)

            count = 0
            for neighbor_cui, relation in neighbors:
                if neighbor_cui in visited:
                    continue
                visited.add(neighbor_cui)
                results.append(
                    ExpandedCUI(
                        cui=neighbor_cui,
                        hop=hop,
                        relation=relation,
                        source=src,
                    )
                )
                next_frontier.append(neighbor_cui)
                count += 1

                if max_per_hop is not None and count >= max_per_hop:
                    break

        frontier = next_frontier

    return results


def group_by_hop(results: list[ExpandedCUI]) -> dict[int, list[ExpandedCUI]]:
    """Group expansion results by hop level."""
    groups: dict[int, list[ExpandedCUI]] = {}
    for r in results:
        groups.setdefault(r.hop, []).append(r)
    return groups
