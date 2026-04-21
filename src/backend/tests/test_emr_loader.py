# backend/tests/test_emr_loader.py
import pytest
from unittest.mock import patch
from app.services.emr_loader import load_patient_data

def test_load_patient_data_success():
    mock_json = '{"name": "John Doe", "age": "40"}'
    with patch("pathlib.Path.read_text", return_value=mock_json):
        result = load_patient_data("patient_101")
        assert result == mock_json

def test_load_patient_data_not_found():
    with patch("pathlib.Path.read_text", side_effect=FileNotFoundError):
        result = load_patient_data("non_existent_patient")
        assert result == "{}"

def test_load_patient_data_exception():
    with patch("pathlib.Path.read_text", side_effect=Exception("Read error")):
        result = load_patient_data("error_patient")
        assert result == "{}"
