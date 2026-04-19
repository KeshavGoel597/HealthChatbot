"""
RAG pipeline orchestrator.

Single entry point that runs all 4 phases:
  Query → Seed CUIs → Graph Expansion → EMR Matching → Prompt

Designed for use by the chat endpoint. All heavy resources
(EmbeddingIndex, KnowledgeGraph) are loaded once at startup
and passed in — this module owns no global state.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.services.rag.embeddings import EmbeddingIndex
from app.services.rag.graph import KnowledgeGraph
from app.services.rag.cui_search import find_cuis
from app.services.rag.term_extractor import ExtractionResult, TermExtractor
from app.services.rag.graph_expand import DIAGNOSTIC_RELATIONS, expand_cuis
from app.services.rag.emr import (
    parse_emr_file,
    extract_sections,
    deduplicate_sections,
    EMRSection,
)
from app.services.rag.emr_match import match_sections, MatchedSection
from app.services.rag.prompt import assemble_prompt, assemble_context


def _sections_for_categories(
    categories: list[str], sections: list[EMRSection]
) -> list[MatchedSection]:
    """Return all EMR sections whose category is in the given list.

    Used for broad-intent queries where the LLM identifies which EMR
    record types the user wants (e.g. ["medicine", "lab"]). No CUI
    search or embedding lookup is needed — we filter directly by the
    section.category string set by the EMR parser.
    """
    target = set(categories)
    return [
        MatchedSection(section=s, matched_cuis=[], best_score=1.0)
        for s in sections
        if s.category in target
    ]


@dataclass
class PipelineResult:
    """Output of a full RAG pipeline run."""

    # Phase 1
    seed_cuis: list[dict]  # [{"cui", "name", "score"}, ...]

    # Phase 2
    expanded_cui_count: int
    expanded_cuis: list  # raw ExpandedCUI list from expand_cuis()

    # Phase 3
    matches: list[MatchedSection]

    # Phase 4
    system_prompt: str
    context_text: str  # Just the clinical context block (no persona)

    # Sections parsed from EMR (for debugging / logging)
    total_sections: int

    # Term extractor output
    extraction: ExtractionResult


def run_pipeline(
    query: str,
    emr_path: str,
    index: EmbeddingIndex,
    graph: KnowledgeGraph,
    *,
    patient_id: str = "",
    extractor: TermExtractor | None = None,
    seed_top_k: int = 10,
    seed_threshold: float = 0.7,
    graph_depth: int = 2,
    allowed_relations: set[str] | None = DIAGNOSTIC_RELATIONS,
    match_top_k: int = 20,
    match_threshold: float = 0.5,
    dedup: bool = True,
) -> PipelineResult:
    """Run the full 4-phase RAG pipeline.

    Args:
        query: Natural language user question.
        emr_path: Path to patient EMR JSON file.
        index: Pre-loaded EmbeddingIndex.
        graph: Pre-loaded KnowledgeGraph.
        patient_id: Patient identifier for the prompt.
        extractor: TermExtractor (Qwen LLM) for query decomposition.
                   Improves CUI recall on conversational queries.
                   If None, falls back to using the raw query as the
                   seed term (skips find_cuis entirely).
        seed_top_k: Phase 1 FAISS candidates.
        seed_threshold: Phase 1 minimum cosine similarity.
        graph_depth: Phase 2 BFS depth.
        allowed_relations: Phase 2 relation filter (None = all).
        match_top_k: Phase 3 FAISS candidates per EMR section.
        match_threshold: Phase 3 minimum cosine similarity.
        dedup: Deduplicate EMR sections before matching.

    Returns:
        PipelineResult with all phase outputs.
    """
    # ── Degraded path: no LLM extractor available ──
    if extractor is None:
        context_text = assemble_context([])
        system_prompt = assemble_prompt(query, [], patient_id=patient_id, context=context_text)
        return PipelineResult(
            seed_cuis=[], expanded_cui_count=0, expanded_cuis=[],
            matches=[], system_prompt=system_prompt, context_text=context_text,
            total_sections=0,
            extraction=ExtractionResult(intent="specific", categories=[], terms=[]),
        )

    # ── Classify query ──
    extracted = extractor.extract(query)

    # ── Phase 1: Seed CUI extraction (specific clinical terms only) ──
    if extracted.intent in ("specific", "mixed") and extracted.terms:
        seeds = find_cuis(
            extracted.terms, index, top_k=seed_top_k, threshold=seed_threshold,
        )
        seed_cuis = [s["cui"] for s in seeds]
    else:
        seeds = []
        seed_cuis = []

    # ── Phase 2: Graph expansion ──
    if seed_cuis:
        expanded = expand_cuis(
            seed_cuis, graph, depth=graph_depth, allowed_relations=allowed_relations,
        )
        expanded_set = {e.cui for e in expanded}
    else:
        expanded = []
        expanded_set = set()

    # ── Phase 3: EMR matching ──
    emr = parse_emr_file(emr_path)
    sections = extract_sections(emr)
    if dedup:
        sections = deduplicate_sections(sections)

    # CUI-based matches (specific/mixed path)
    cui_matches = (
        match_sections(
            sections, expanded_set, index,
            top_k=match_top_k, threshold=match_threshold,
        )
        if expanded_set else []
    )

    # Category-based matches (broad/mixed path)
    category_matches = (
        _sections_for_categories(extracted.categories, sections)
        if extracted.intent in ("broad", "mixed") and extracted.categories else []
    )

    # Merge: CUI matches first, then category matches not already covered
    seen_texts = {m.section.text for m in cui_matches}
    matches = list(cui_matches)
    for m in category_matches:
        if m.section.text not in seen_texts:
            matches.append(m)
            seen_texts.add(m.section.text)

    # ── Phase 4: Prompt assembly ──
    context_text = assemble_context(matches)
    system_prompt = assemble_prompt(query, matches, patient_id=patient_id, context=context_text)

    return PipelineResult(
        seed_cuis=seeds,
        expanded_cui_count=len(expanded_set),
        expanded_cuis=expanded,
        matches=matches,
        system_prompt=system_prompt,
        context_text=context_text,
        total_sections=len(sections),
        extraction=extracted,
    )
