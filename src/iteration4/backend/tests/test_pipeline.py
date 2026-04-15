"""
Pipeline test suite with per-phase metrics.

Tests EMR parsing, full pipeline correctness (with mocked ML resources),
and per-phase timing + token count reporting.

Run with:
    pytest tests/test_pipeline.py -v          # all tests
    pytest tests/test_pipeline.py -v -s       # shows timing + token output
    pytest tests/test_pipeline.py -v -s -k metrics  # metrics tests only
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from unittest.mock import MagicMock

from app.services.rag.emr import (
    parse_emr_file,
    extract_sections,
    deduplicate_sections,
)
from app.services.rag.cui_search import find_cuis
from app.services.rag.graph_expand import expand_cuis, DIAGNOSTIC_RELATIONS
from app.services.rag.emr_match import match_sections
from app.services.rag.prompt import assemble_prompt
from app.services.rag.pipeline import run_pipeline, PipelineResult


# ── Synthetic EMR ──────────────────────────────────────────────────────

SYNTHETIC_EMR = {
    "age": "45 Yrs",
    "sex": "M",
    "lab_data": [
        {"name": "HbA1c", "value": "7.2", "date": "2024-01-15"},
        {"name": "RBS", "value": "210", "date": "2024-02-01"},
    ],
    "prescriptions": [
        {"name": "Diagnosis", "value": "Type 2 Diabetes Mellitus", "date": "2024-01-01"},
        {"name": "Symptoms", "value": "Increased thirst", "date": "2024-01-01"},
        {"medicine": "Metformin 500mg"},
        {"medicine": "Glipizide 5mg"},
    ],
    "discharge_summary": [],
    "assessments_data": [],
}


# ── Fixtures ───────────────────────────────────────────────────────────

@pytest.fixture
def emr_file(tmp_path):
    """Write synthetic EMR to a temp file; return its path as a string."""
    path = tmp_path / "patient_test.json"
    path.write_text(json.dumps(SYNTHETIC_EMR))
    return str(path)


@pytest.fixture
def matching_index():
    """EmbeddingIndex mock that always returns two matching CUIs."""
    m = MagicMock()
    m.encode.return_value = np.ones(768, dtype=np.float32)
    m.search.return_value = [("C0011849", 0.85), ("C0005823", 0.78)]

    name_map = {
        "C0011849": "Diabetes Mellitus",
        "C0005823": "Blood Glucose",
        "C0041696": "Hyperglycemia",
    }
    m.get_name.side_effect = lambda cui: name_map.get(cui, cui)
    return m


@pytest.fixture
def matching_graph():
    """KnowledgeGraph mock with one neighbor from C0011849."""
    m = MagicMock()
    neighbor_map = {
        "C0011849": [("C0041696", "has associated finding")],
        "C0005823": [],
        "C0041696": [],
    }
    m.neighbors.side_effect = lambda cui, allowed_relations=None: neighbor_map.get(cui, [])
    return m


@pytest.fixture
def matching_extractor():
    """TermExtractor mock that returns ['diabetes'] for any query."""
    m = MagicMock()
    m.extract.return_value = ["diabetes"]
    return m


@pytest.fixture
def no_match_extractor():
    """TermExtractor mock that returns no terms → no seeds → no matches."""
    m = MagicMock()
    m.extract.return_value = []
    return m


# ── TestEMRParsing ─────────────────────────────────────────────────────

class TestEMRParsing:
    """Pure EMR parsing functions — no ML dependencies."""

    def test_sections_extracted(self):
        sections = extract_sections(SYNTHETIC_EMR)
        assert len(sections) > 0

    def test_lab_sections_present(self):
        sections = extract_sections(SYNTHETIC_EMR)
        lab_sections = [s for s in sections if s.category == "lab"]
        assert len(lab_sections) == 2
        assert "HbA1c" in [s.text for s in lab_sections]

    def test_medicine_sections_present(self):
        sections = extract_sections(SYNTHETIC_EMR)
        medicine_sections = [s for s in sections if s.category == "medicine"]
        assert len(medicine_sections) >= 2

    def test_demographics_extracted(self):
        sections = extract_sections(SYNTHETIC_EMR)
        demo_sections = [s for s in sections if s.category == "demographics"]
        assert len(demo_sections) > 0

    def test_dedup_removes_duplicates(self):
        """Duplicate lab entry deduplicates to a single section."""
        emr_with_dupe = {
            "lab_data": [
                {"name": "HbA1c", "value": "7.2", "date": "2024-01-15"},
                {"name": "HbA1c", "value": "7.2", "date": "2024-01-15"},  # exact dupe
            ],
        }
        sections = extract_sections(emr_with_dupe)
        assert len([s for s in sections if s.category == "lab"]) == 2  # before dedup

        deduped = deduplicate_sections(sections)
        assert len([s for s in deduped if s.category == "lab"]) == 1


# ── TestPipelineCorrectness ────────────────────────────────────────────

class TestPipelineCorrectness:
    """Full run_pipeline() with mocked ML resources."""

    _QUERY = "What is my diabetes status?"

    def _run(self, emr_file, index, graph, extractor):
        return run_pipeline(
            query=self._QUERY,
            emr_path=emr_file,
            index=index,
            graph=graph,
            extractor=extractor,
        )

    def test_pipeline_returns_result(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        result = self._run(emr_file, matching_index, matching_graph, matching_extractor)
        assert isinstance(result, PipelineResult)

    def test_matching_query_produces_seed_cuis(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        result = self._run(emr_file, matching_index, matching_graph, matching_extractor)
        assert len(result.seed_cuis) > 0

    def test_matching_query_expands_cuis(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        result = self._run(emr_file, matching_index, matching_graph, matching_extractor)
        assert result.expanded_cui_count > 0

    def test_no_seed_cuis_produces_empty_matches(
        self, emr_file, matching_index, matching_graph, no_match_extractor
    ):
        """When extractor returns no terms, seeds are empty → no EMR matches."""
        result = self._run(emr_file, matching_index, matching_graph, no_match_extractor)
        assert result.matches == []

    def test_system_prompt_contains_persona(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        result = self._run(emr_file, matching_index, matching_graph, matching_extractor)
        assert "Robert" in result.system_prompt

    def test_no_match_prompt_has_fallback(
        self, emr_file, matching_index, matching_graph, no_match_extractor
    ):
        result = self._run(emr_file, matching_index, matching_graph, no_match_extractor)
        assert "No specific clinical records" in result.system_prompt


# ── TestPipelineMetrics ────────────────────────────────────────────────

class TestPipelineMetrics:
    """Per-phase timing and prompt token count, visible with pytest -s."""

    _QUERY = "What is my diabetes status?"

    def test_prompt_token_count_reported(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        result = run_pipeline(
            query=self._QUERY,
            emr_path=emr_file,
            index=matching_index,
            graph=matching_graph,
            extractor=matching_extractor,
        )
        chars = len(result.system_prompt)
        words = len(result.system_prompt.split())
        token_est = chars // 4

        print(
            f"\n[token count]\n"
            f"  Prompt chars:        {chars}\n"
            f"  Prompt words:        {words}\n"
            f"  Prompt tokens (est): {token_est}"
        )
        assert token_est > 0

    def test_per_phase_timing(
        self, emr_file, matching_index, matching_graph, matching_extractor
    ):
        """Call each phase function individually and report timing."""
        # Phase 1: seed CUI extraction
        t0 = time.perf_counter()
        seeds = find_cuis(
            self._QUERY, matching_index,
            top_k=10, threshold=0.7,
            extractor=matching_extractor,
        )
        phase1_ms = (time.perf_counter() - t0) * 1000

        seed_cuis = [s["cui"] for s in seeds]

        # Phase 2: graph expansion
        t0 = time.perf_counter()
        expanded = expand_cuis(
            seed_cuis, matching_graph,
            depth=2, allowed_relations=DIAGNOSTIC_RELATIONS,
        )
        phase2_ms = (time.perf_counter() - t0) * 1000
        expanded_set = {e.cui for e in expanded}

        # Parse EMR (setup for Phase 3 — not timed as a pipeline phase)
        emr = parse_emr_file(emr_file)
        sections = deduplicate_sections(extract_sections(emr))

        # Phase 3: EMR section matching
        t0 = time.perf_counter()
        matches = match_sections(
            sections, expanded_set, matching_index,
            top_k=20, threshold=0.5,
        )
        phase3_ms = (time.perf_counter() - t0) * 1000

        # Phase 4: prompt assembly
        t0 = time.perf_counter()
        prompt = assemble_prompt(self._QUERY, matches, patient_id="test-patient")
        phase4_ms = (time.perf_counter() - t0) * 1000

        total_ms = phase1_ms + phase2_ms + phase3_ms + phase4_ms
        chars = len(prompt)
        words = len(prompt.split())
        token_est = chars // 4

        print(
            f"\n[phase timing]\n"
            f"  Phase 1 (seed CUIs):     {phase1_ms:.2f}ms\n"
            f"  Phase 2 (graph expand):  {phase2_ms:.2f}ms\n"
            f"  Phase 3 (EMR match):     {phase3_ms:.2f}ms\n"
            f"  Phase 4 (prompt):        {phase4_ms:.2f}ms\n"
            f"  Total:                   {total_ms:.2f}ms\n"
            f"  Prompt chars:            {chars}\n"
            f"  Prompt words:            {words}\n"
            f"  Prompt tokens (est):     {token_est}"
        )

        assert phase1_ms < 5000, f"Phase 1 took {phase1_ms:.2f}ms (> 5s)"
        assert phase2_ms < 5000, f"Phase 2 took {phase2_ms:.2f}ms (> 5s)"
        assert phase3_ms < 5000, f"Phase 3 took {phase3_ms:.2f}ms (> 5s)"
        assert phase4_ms < 5000, f"Phase 4 took {phase4_ms:.2f}ms (> 5s)"
        assert token_est > 0
