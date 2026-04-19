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


from app.services.rag.term_extractor import ExtractionResult, _parse_extraction_response


class TestParseExtractionResponse:
    def test_specific_intent_has_terms_no_categories(self):
        r = _parse_extraction_response(
            '{"intent":"specific","categories":[],"terms":["headache"]}', "q"
        )
        assert r.intent == "specific"
        assert r.terms == ["headache"]
        assert r.categories == []

    def test_broad_intent_has_categories_no_terms(self):
        r = _parse_extraction_response(
            '{"intent":"broad","categories":["medicine"],"terms":[]}', "q"
        )
        assert r.intent == "broad"
        assert r.categories == ["medicine"]
        assert r.terms == []

    def test_mixed_intent_has_both(self):
        r = _parse_extraction_response(
            '{"intent":"mixed","categories":["medicine"],"terms":["diabetes"]}', "q"
        )
        assert r.intent == "mixed"
        assert r.categories == ["medicine"]
        assert r.terms == ["diabetes"]

    def test_invalid_category_strings_are_dropped(self):
        r = _parse_extraction_response(
            '{"intent":"broad","categories":["medicine","laboratory","bogus"],"terms":[]}', "q"
        )
        assert r.categories == ["medicine"]  # "laboratory" and "bogus" are not valid

    def test_unknown_intent_defaults_to_specific(self):
        r = _parse_extraction_response(
            '{"intent":"whatever","categories":[],"terms":["x"]}', "q"
        )
        assert r.intent == "specific"

    def test_invalid_json_falls_back_to_query(self):
        r = _parse_extraction_response("not json at all", "my query")
        assert r.intent == "specific"
        assert r.terms == ["my query"]
        assert r.categories == []

    def test_flat_list_response_treated_as_specific_terms(self):
        r = _parse_extraction_response('["headache","nausea"]', "q")
        assert r.intent == "specific"
        assert r.terms == ["headache", "nausea"]
        assert r.categories == []

    def test_blank_terms_stripped(self):
        r = _parse_extraction_response(
            '{"intent":"specific","categories":[],"terms":["  ","headache"]}', "q"
        )
        assert r.terms == ["headache"]

    def test_multiple_valid_categories(self):
        r = _parse_extraction_response(
            '{"intent":"broad","categories":["diagnosis","comorbidity"],"terms":[]}', "q"
        )
        assert set(r.categories) == {"diagnosis", "comorbidity"}
