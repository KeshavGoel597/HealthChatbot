# backend/tests/conftest.py
import numpy as np
import os
import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from app.services.rag.term_extractor import ExtractionResult


@pytest.fixture
def mock_index():
    """EmbeddingIndex stub — returns zero vectors and empty search results."""
    m = MagicMock()
    m.encode.return_value = np.zeros(768, dtype=np.float32)
    m.search.return_value = []
    m.get_name.return_value = "MockCUI"
    return m


@pytest.fixture
def mock_graph():
    """KnowledgeGraph stub — returns no neighbours."""
    m = MagicMock()
    m.neighbors.return_value = []
    return m


@pytest.fixture
def mock_extractor():
    """TermExtractor stub — returns a single generic specific term."""
    m = MagicMock()
    m.extract.return_value = ExtractionResult(
        intent="specific", categories=[], terms=["mock term"]
    )
    return m


@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient with mocked ML resources and isolated session storage."""
    # Redirect session storage to temp dir
    sessions_dir = str(tmp_path / "sessions")
    os.makedirs(sessions_dir)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-tests")

    import app.services.session_store as ss
    monkeypatch.setattr(ss, "SESSIONS_DIR", sessions_dir)
    import app.routers.sessions as sr
    monkeypatch.setattr(sr, "SESSIONS_DIR", sessions_dir)
    import app.routers.gdpr_router as gr
    monkeypatch.setattr(gr, "SESSIONS_DIR", sessions_dir)

    # Patch heavy ML constructors so lifespan does not try to load .pkl files
    monkeypatch.setattr(
        "app.main.EmbeddingIndex",
        lambda *a, **kw: MagicMock(
            encode=lambda t: np.zeros(768, dtype=np.float32),
            search=lambda v, k: [],
            get_name=lambda c: c,
        ),
    )
    monkeypatch.setattr("app.main.KnowledgeGraph", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "app.main.TermExtractor",
        lambda *a, **kw: MagicMock(
            extract=lambda q: ExtractionResult(intent="specific", categories=[], terms=[])
        ),
    )
    monkeypatch.setattr("app.routers.sessions.run_retention_cleanup", lambda: None)
    monkeypatch.setattr("app.routers.sessions.sarvam_service", MagicMock(
        translate=lambda *a, **kw: None,
        text_to_speech=lambda *a, **kw: None,
    ))

    # Patch shared orchestration function imported by routers.
    async def _fake_run_llm_turn(*a, **kw):
        return {
            "response": "Mocked LLM response.",
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "model_name": "mock",
            "emr_fields_used": [],
            "was_compacted": False,
            "new_compacted_summary": None,
        }

    monkeypatch.setattr("app.routers.chat.run_llm_turn", _fake_run_llm_turn)
    monkeypatch.setattr("app.routers.sessions.run_llm_turn", _fake_run_llm_turn)

    from app.main import app
    with TestClient(app) as c:
        yield c
