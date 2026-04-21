# backend/tests/test_emr_summary.py
import pytest
from app.services.emr_summary import summarize_emr_context

def test_summarize_emr_context_empty():
    summary, fields = summarize_emr_context("")
    assert summary == ""
    assert fields == []

def test_summarize_emr_context_full():
    raw_data = '''
    age: "45"
    sex: "Male"
    "diag" => "Hypertension"
    "diag" => "@10"
    "sym" => "Headache"
    "sym" => "FCU"
    "medicine" => "Lisinopril"
    "name" => "Hemoglobin", "value" => "14.2", "date" => "2023-01-01"
    "name" => "Hemoglobin", "value" => "14.5", "date" => "2023-06-01"
    '''
    summary, fields = summarize_emr_context(raw_data)
    assert "PATIENT: Age 45, Sex Male" in summary
    assert "DIAGNOSES: Hypertension" in summary
    assert "SYMPTOMS: Headache" in summary
    assert "MEDICATIONS: Lisinopril" in summary
    assert "RECENT LABS: Hemoglobin: 14.5 (2023-06-01)" in summary
    assert set(fields) == {
        "Patient Demographics",
        "Medical Diagnoses",
        "Recorded Symptoms",
        "Prescribed Medications",
        "Laboratory Results"
    }

def test_summarize_emr_context_partial():
    raw_data = 'age: "30"'
    summary, fields = summarize_emr_context(raw_data)
    assert summary == "PATIENT: Age 30, Sex ?"
    assert fields == ["Patient Demographics"]
