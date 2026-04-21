"""Shared EMR file loading helpers."""

from pathlib import Path


def _data_dir() -> Path:
    # app/services -> app -> backend
    backend_dir = Path(__file__).resolve().parents[2]
    return backend_dir / "data"


def load_patient_data(patient_id: str) -> str:
    """Load raw patient EMR JSON as text, with a safe empty fallback."""
    data_path = _data_dir() / f"{patient_id}.json"
    try:
        return data_path.read_text()
    except FileNotFoundError:
        return "{}"
    except Exception as e:
        print(f"Error reading patient data: {e}")
        return "{}"
