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
from app.services.rag.term_extractor import ExtractionResult


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
    """TermExtractor mock that returns specific intent with ['diabetes']."""
    m = MagicMock()
    m.extract.return_value = ExtractionResult(
        intent="specific", categories=[], terms=["diabetes"]
    )
    return m


@pytest.fixture
def no_match_extractor():
    """TermExtractor mock that returns no terms → no seeds → no matches."""
    m = MagicMock()
    m.extract.return_value = ExtractionResult(
        intent="specific", categories=[], terms=[]
    )
    return m


def _build_long_emr() -> dict:
    """Generate a long EMR with many noisy sections and one nuanced relevant note."""
    prescriptions = [
        {
            "name": "Comments",
            "value": f"Routine follow-up note {i}: sleep and appetite stable.",
            "date": f"2024-03-{(i % 28) + 1:02d}",
        }
        for i in range(140)
    ]
    prescriptions.extend(
        [
            {
                "name": "Patient History",
                "value": "Long-standing diabetes with occasional pedal swelling after long days.",
                "date": "2024-03-20",
            },
            {
                "name": "RecommendedLabs",
                "value": "Urine albumin creatinine ratio advised for early renal risk stratification.",
                "date": "2024-03-21",
            },
            {"medicine": "Metformin 500mg"},
            {"medicine": "Telmisartan 40mg"},
        ]
    )

    return {
        "age": "59 Yrs",
        "sex": "F",
        "lab_data": [
            {"name": "HbA1c", "value": "8.1", "date": "2024-03-01"},
            {"name": "Creatinine", "value": "1.1", "date": "2024-03-01"},
            *[
                {
                    "name": f"Routine Lab {i}",
                    "value": str(100 + i),
                    "date": f"2024-02-{(i % 28) + 1:02d}",
                }
                for i in range(55)
            ],
        ],
        "prescriptions": prescriptions,
        "discharge_summary": [],
        "assessments_data": [],
    }


@pytest.fixture
def distinct_case_specs():
    """Distinct EMR/query scenarios with expected nuanced extraction targets."""
    return [
        {
            "name": "vascular_indirect",
            "query": "Why does my leg pain start after walking and settle when I stand still?",
            "extract_terms": ["intermittent claudication"],
            "seed": [("C1704436", 0.89)],
            "neighbors": {"C1704436": [("C0007222", "associated_with")]},
            "emr": {
                "age": "63 Yrs",
                "sex": "M",
                "lab_data": [
                    {"name": "LDL", "value": "165", "date": "2024-01-10"},
                ],
                "prescriptions": [
                    {
                        "name": "Comments",
                        "value": "Calf tightness appears after 8-10 minutes of walking and eases after brief rest.",
                        "date": "2024-01-11",
                    },
                    {"medicine": "Atorvastatin 20mg"},
                ],
                "discharge_summary": [],
                "assessments_data": [],
            },
            "section_matchers": [
                ("calf tightness", [("C1704436", 0.87)]),
            ],
            "expected_text": "Calf tightness appears after 8-10 minutes of walking",
            "min_total_sections": 4,
        },
        {
            "name": "thyroid_pattern",
            "query": "Can my tiredness and feeling cold be linked to one issue?",
            "extract_terms": ["hypothyroidism"],
            "seed": [("C0020676", 0.9)],
            "neighbors": {"C0020676": []},
            "emr": {
                "age": "38 Yrs",
                "sex": "F",
                "lab_data": [
                    {"name": "TSH", "value": "11.2", "date": "2024-02-04"},
                ],
                "prescriptions": [
                    {
                        "name": "Patient History",
                        "value": "Progressive fatigue, dry skin, and reduced exercise tolerance over 6 months.",
                        "date": "2024-02-03",
                    },
                    {"medicine": "Levothyroxine 50mcg"},
                ],
                "discharge_summary": [],
                "assessments_data": [],
            },
            "section_matchers": [
                ("fatigue", [("C0020676", 0.84)]),
                ("TSH", [("C0020676", 0.82)]),
            ],
            "expected_text": "Progressive fatigue",
            "min_total_sections": 4,
        },
        {
            "name": "long_emr_kidney_screening",
            "query": "How do we catch early kidney stress from long-term sugar disease?",
            "extract_terms": ["diabetic nephropathy screening"],
            "seed": [("C0011881", 0.88)],
            "neighbors": {"C0011881": [("C0520679", "has_associated_finding")]},
            "emr": _build_long_emr(),
            "section_matchers": [
                ("albumin creatinine ratio", [("C0011881", 0.86)]),
                ("pedal swelling", [("C0520679", 0.79)]),
            ],
            "expected_text": "Urine albumin creatinine ratio advised",
            "min_total_sections": 190,
        },
    ]


def _mock_pipeline_resources(case: dict):
    """Create index/graph/extractor mocks tailored to one scenario."""
    extractor = MagicMock()
    extractor.extract.return_value = ExtractionResult(
        intent="specific", categories=[], terms=case["extract_terms"]
    )

    index = MagicMock()
    index.encode.side_effect = lambda text: text
    index.get_name.side_effect = lambda cui: cui

    section_matchers = [
        (needle.lower(), results)
        for needle, results in case["section_matchers"]
    ]

    def _search(encoded, top_k):
        text = str(encoded).lower()
        for term in case["extract_terms"]:
            if term.lower() in text:
                return case["seed"]
        for needle, results in section_matchers:
            if needle in text:
                return results
        return [("C9999999", 0.19)]

    index.search.side_effect = _search

    graph = MagicMock()
    neighbor_map = case["neighbors"]
    graph.neighbors.side_effect = lambda cui, allowed_relations=None: neighbor_map.get(cui, [])

    return index, graph, extractor


@pytest.fixture
def emr_case_file(tmp_path):
    """Write one case EMR payload to a temp file."""
    def _write(case: dict) -> str:
        path = tmp_path / f"{case['name']}.json"
        path.write_text(json.dumps(case["emr"]))
        return str(path)

    return _write


def _find_case(cases: list[dict], name: str) -> dict:
    """Return one named test scenario from distinct_case_specs."""
    return next(c for c in cases if c["name"] == name)


def _assert_case_relevance(case: dict, result: PipelineResult):
    """Shared assertions for scenario coverage and nuanced relevance."""
    matched_texts = [m.section.text for m in result.matches]

    assert len(result.seed_cuis) > 0
    assert result.expanded_cui_count > 0
    assert result.total_sections >= case["min_total_sections"]
    assert any(case["expected_text"] in text for text in matched_texts)


def _measure_case_phases(case: dict, emr_path: str, index, graph, extractor):
    """Measure and print per-phase timing for one scenario."""
    t0 = time.perf_counter()
    extracted = extractor.extract(case["query"])
    seeds = find_cuis(extracted.terms, index, top_k=10, threshold=0.7)
    phase1_ms = (time.perf_counter() - t0) * 1000
    seed_cuis = [s["cui"] for s in seeds]

    t0 = time.perf_counter()
    expanded = expand_cuis(
        seed_cuis, graph,
        depth=2, allowed_relations=DIAGNOSTIC_RELATIONS,
    )
    phase2_ms = (time.perf_counter() - t0) * 1000
    expanded_set = {e.cui for e in expanded}

    emr = parse_emr_file(emr_path)
    sections = deduplicate_sections(extract_sections(emr))

    t0 = time.perf_counter()
    matches = match_sections(
        sections, expanded_set, index,
        top_k=20, threshold=0.5,
    )
    phase3_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    prompt = assemble_prompt(case["query"], matches, patient_id=case["name"])
    phase4_ms = (time.perf_counter() - t0) * 1000

    total_ms = phase1_ms + phase2_ms + phase3_ms + phase4_ms
    chars = len(prompt)
    words = len(prompt.split())
    token_est = chars // 4

    print(
        f"\n[phase timing] case={case['name']}\n"
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


# ── TestFindCuisSignature ──────────────────────────────────────────────

class TestFindCuisSignature:
    """find_cuis accepts list[str] directly — no extractor param."""

    def test_accepts_terms_list(self, matching_index):
        from app.services.rag.cui_search import find_cuis
        results = find_cuis(["diabetes"], matching_index, top_k=10, threshold=0.7)
        assert isinstance(results, list)

    def test_empty_terms_returns_empty(self, matching_index):
        from app.services.rag.cui_search import find_cuis
        results = find_cuis([], matching_index, top_k=10, threshold=0.7)
        assert results == []
        matching_index.encode.assert_not_called()


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
        extracted = matching_extractor.extract(self._QUERY)
        seeds = find_cuis(extracted.terms, matching_index, top_k=10, threshold=0.7)
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


class TestDistinctEMRScenarios:
    """Distinct EMRs + distinct queries to improve retrieval coverage."""

    def test_vascular_indirect_extracts_relevant_section(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "vascular_indirect")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)

        result = run_pipeline(
            query=case["query"],
            emr_path=emr_path,
            index=index,
            graph=graph,
            extractor=extractor,
        )
        _assert_case_relevance(case, result)

    def test_thyroid_pattern_extracts_relevant_section(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "thyroid_pattern")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)

        result = run_pipeline(
            query=case["query"],
            emr_path=emr_path,
            index=index,
            graph=graph,
            extractor=extractor,
        )
        _assert_case_relevance(case, result)

    def test_long_emr_kidney_screening_extracts_relevant_section(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "long_emr_kidney_screening")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)

        result = run_pipeline(
            query=case["query"],
            emr_path=emr_path,
            index=index,
            graph=graph,
            extractor=extractor,
        )
        _assert_case_relevance(case, result)

    def test_long_emr_prompt_contains_nuanced_relevant_section(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "long_emr_kidney_screening")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)

        result = run_pipeline(
            query=case["query"],
            emr_path=emr_path,
            index=index,
            graph=graph,
            extractor=extractor,
        )

        assert result.total_sections >= 190
        assert case["expected_text"] in result.context_text


class TestDistinctEMRMetrics:
    """Per-case timing output so each scenario has its own metrics block."""

    def test_phase_timing_vascular_indirect(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "vascular_indirect")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)
        _measure_case_phases(case, emr_path, index, graph, extractor)

    def test_phase_timing_thyroid_pattern(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "thyroid_pattern")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)
        _measure_case_phases(case, emr_path, index, graph, extractor)

    def test_phase_timing_long_emr_kidney_screening(
        self,
        distinct_case_specs,
        emr_case_file,
    ):
        case = _find_case(distinct_case_specs, "long_emr_kidney_screening")
        emr_path = emr_case_file(case)
        index, graph, extractor = _mock_pipeline_resources(case)
        _measure_case_phases(case, emr_path, index, graph, extractor)
