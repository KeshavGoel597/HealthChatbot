# backend/tests/test_rag_pure.py
"""Unit tests for pure RAG functions (no torch/faiss/embedding needed)."""
import pytest
from app.services.rag.emr import EMRSection
from app.services.rag.emr_match import MatchedSection, _clean_for_encoding
from app.services.rag.prompt import assemble_context, assemble_prompt


# ── helpers ───────────────────────────────────────────────────────────

def _section(text, category="diagnosis", date="", value=""):
    return EMRSection(category=category, text=text, date=date, value=value)


def _match(text, category="diagnosis", score=0.9):
    return MatchedSection(
        section=_section(text, category),
        matched_cuis=["C0000001"],
        best_score=score,
    )


# ── _clean_for_encoding ───────────────────────────────────────────────

def test_clean_strips_parentheses():
    result = _clean_for_encoding("DIABETES (SUGAR)", "diagnosis")
    assert "SUGAR" in result.upper()
    assert "(" not in result


def test_clean_removes_medicine_dose():
    result = _clean_for_encoding("METFORMIN 500MG", "medicine")
    assert "500" not in result
    assert "metformin" in result.lower()


def test_clean_replaces_hyphens_with_spaces():
    result = _clean_for_encoding("Creatinine- Serum", "lab")
    assert "-" not in result


def test_clean_collapses_whitespace():
    result = _clean_for_encoding("  a   b  ", "diagnosis")
    assert result == "a b"


# ── assemble_context ──────────────────────────────────────────────────

def test_assemble_context_empty_returns_empty_string():
    assert assemble_context([]) == ""


def test_assemble_context_groups_by_category():
    matches = [
        _match("Type 2 Diabetes", "diagnosis"),
        _match("Headache", "symptom"),
    ]
    result = assemble_context(matches)
    assert "Diagnoses:" in result
    assert "Symptoms:" in result
    assert "Type 2 Diabetes" in result
    assert "Headache" in result


def test_assemble_context_respects_category_order():
    # diagnosis should appear before symptom in output
    matches = [
        _match("Headache", "symptom"),
        _match("Diabetes", "diagnosis"),
    ]
    result = assemble_context(matches)
    assert result.index("Diagnoses:") < result.index("Symptoms:")


def test_assemble_context_includes_date_when_present():
    m = MatchedSection(
        section=EMRSection(category="lab", text="HbA1c", date="2024-01-15", value="7.2"),
        matched_cuis=["C0001"],
        best_score=0.8,
    )
    result = assemble_context([m])
    assert "2024-01-15" in result


# ── assemble_prompt ───────────────────────────────────────────────────

def test_assemble_prompt_no_matches_returns_general_prompt():
    result = assemble_prompt("How are you?", [])
    assert "Robert" in result
    assert "No specific clinical records" in result


def test_assemble_prompt_with_matches_contains_clinical_block():
    matches = [_match("Type 2 Diabetes")]
    result = assemble_prompt("What is my diagnosis?", matches, patient_id="p101")
    assert "CLINICAL RECORD" in result
    assert "Type 2 Diabetes" in result


def test_assemble_prompt_no_missing_spaces():
    """Regression: string concat bug that caused words to be joined without spaces."""
    matches = [_match("Hypertension")]
    result = assemble_prompt("Do I have high blood pressure?", matches)
    # No two words should be directly joined (no lowercase letter immediately followed by uppercase with no space)
    import re
    assert "mustsupplement" not in result
    assert "questionyou" not in result
    # Verify the key sentence has proper spacing
    assert "you must " in result


from app.services.rag.term_extractor import _parse_categories, _parse_terms, _compute_intent


class TestParseCategories:
    def test_valid_category_returned(self):
        assert _parse_categories('{"categories":["medicine"]}') == ["medicine"]

    def test_invalid_categories_dropped(self):
        result = _parse_categories('{"categories":["medicine","laboratory","bogus"]}')
        assert result == ["medicine"]

    def test_multiple_valid_categories(self):
        result = _parse_categories('{"categories":["diagnosis","comorbidity"]}')
        assert set(result) == {"diagnosis", "comorbidity"}

    def test_empty_array_returns_empty(self):
        assert _parse_categories('{"categories":[]}') == []

    def test_invalid_json_returns_empty(self):
        assert _parse_categories("not json") == []

    def test_non_dict_json_returns_empty(self):
        assert _parse_categories('["medicine"]') == []


class TestParseTerms:
    def test_terms_returned(self):
        assert _parse_terms('{"terms":["headache"]}') == ["headache"]

    def test_blank_terms_stripped(self):
        result = _parse_terms('{"terms":["  ","headache"]}')
        assert result == ["headache"]

    def test_multiple_terms(self):
        result = _parse_terms('{"terms":["headache","nausea"]}')
        assert result == ["headache", "nausea"]

    def test_empty_array_returns_empty(self):
        assert _parse_terms('{"terms":[]}') == []

    def test_invalid_json_returns_empty(self):
        assert _parse_terms("not json") == []


class TestComputeIntent:
    def test_terms_only_is_specific(self):
        assert _compute_intent([], ["headache"]) == "specific"

    def test_categories_only_is_broad(self):
        assert _compute_intent(["medicine"], []) == "broad"

    def test_both_is_mixed(self):
        assert _compute_intent(["medicine"], ["diabetes"]) == "mixed"

    def test_both_empty_is_specific(self):
        assert _compute_intent([], []) == "specific"


from app.services.rag.pipeline import _sections_for_categories


class TestSectionsForCategories:
    def _section(self, category, text):
        return EMRSection(category=category, text=text)

    def test_medicine_category_selects_medicine_sections(self):
        sections = [
            self._section("medicine", "Metformin 500mg"),
            self._section("medicine", "Glipizide 5mg"),
            self._section("lab", "HbA1c"),
        ]
        matches = _sections_for_categories(["medicine"], sections)
        assert len(matches) == 2
        assert all(m.section.category == "medicine" for m in matches)

    def test_matched_sections_have_score_1_and_empty_cuis(self):
        sections = [self._section("medicine", "Metformin")]
        matches = _sections_for_categories(["medicine"], sections)
        assert matches[0].best_score == 1.0
        assert matches[0].matched_cuis == []

    def test_unknown_category_returns_empty(self):
        sections = [self._section("medicine", "Metformin")]
        matches = _sections_for_categories(["nonexistent"], sections)
        assert matches == []

    def test_empty_categories_returns_empty(self):
        sections = [self._section("medicine", "Metformin")]
        matches = _sections_for_categories([], sections)
        assert matches == []

    def test_lab_category_selects_lab_sections(self):
        sections = [
            self._section("lab", "HbA1c"),
            self._section("lab", "RBS"),
            self._section("medicine", "Metformin"),
        ]
        matches = _sections_for_categories(["lab"], sections)
        assert len(matches) == 2
        assert all(m.section.category == "lab" for m in matches)

    def test_multiple_categories_selects_from_all(self):
        sections = [
            self._section("medicine", "Metformin"),
            self._section("lab", "HbA1c"),
            self._section("diagnosis", "Type 2 Diabetes"),
        ]
        matches = _sections_for_categories(["medicine", "lab"], sections)
        assert len(matches) == 2
        assert {m.section.category for m in matches} == {"medicine", "lab"}
