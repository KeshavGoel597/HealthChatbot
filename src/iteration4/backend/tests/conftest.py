# backend/tests/conftest.py
import numpy as np
import os
import pytest
from unittest.mock import MagicMock, AsyncMock
from fastapi.testclient import TestClient


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
    """TermExtractor stub — returns single generic term."""
    m = MagicMock()
    m.extract.return_value = ["mock term"]
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
        lambda *a, **kw: MagicMock(extract=lambda q: []),
    )
    monkeypatch.setattr("app.routers.sessions.run_retention_cleanup", lambda: None)
    monkeypatch.setattr("app.routers.sessions.sarvam_service", MagicMock(
        translate=lambda *a, **kw: None,
        text_to_speech=lambda *a, **kw: None,
    ))

    # Patch GeminiService.chat so no real API calls are made
    async def _fake_chat(*a, **kw):
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

    monkeypatch.setattr("app.routers.chat.gemini_service.chat", _fake_chat)
    monkeypatch.setattr("app.routers.sessions.gemini_service.chat", _fake_chat)

    # Patch run_pipeline to avoid needing real EMR files
    fake_pipeline = MagicMock(system_prompt="")
    monkeypatch.setattr(
        "app.routers.sessions.run_pipeline", lambda **kw: fake_pipeline
    )
    monkeypatch.setattr(
        "app.routers.chat.run_pipeline", lambda **kw: fake_pipeline
    )

    from app.main import app
    with TestClient(app) as c:
        yield c
