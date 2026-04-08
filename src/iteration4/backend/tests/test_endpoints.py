# backend/tests/test_endpoints.py
"""Integration tests for /sessions, /chat, and /gdpr endpoints."""
import pytest


# ── Health ────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── Session lifecycle ─────────────────────────────────────────────────

def test_create_session(client):
    r = client.post("/sessions/patient101", json={
        "emr_consent": True,
        "store_history_consent": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["patient_id"] == "patient101"
    assert body["emr_consent"] is True
    assert "id" in body
    assert "expires_at" in body


def test_list_sessions_empty(client):
    r = client.get("/sessions/patient101")
    assert r.status_code == 200
    assert r.json() == []


def test_list_sessions_after_create(client):
    client.post("/sessions/patient101", json={})
    client.post("/sessions/patient101", json={})
    r = client.get("/sessions/patient101")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_get_session(client):
    created = client.post("/sessions/patient101", json={}).json()
    session_id = created["id"]
    r = client.get(f"/sessions/{session_id}/messages")
    assert r.status_code == 200
    assert r.json()["id"] == session_id


def test_get_session_not_found(client):
    r = client.get("/sessions/nonexistent-id/messages")
    assert r.status_code == 404


# ── Messaging ─────────────────────────────────────────────────────────

def test_send_message_returns_response(client):
    session_id = client.post("/sessions/patient101", json={
        "store_history_consent": True,
    }).json()["id"]

    r = client.post(f"/sessions/{session_id}/message", json={
        "message": "What medications am I on?",
        "model": "gemini",
    })
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert isinstance(body["input_tokens"], int)
    assert isinstance(body["total_tokens"], int)


def test_send_message_session_not_found(client):
    r = client.post("/sessions/bad-id/message", json={"message": "hello"})
    assert r.status_code == 404


def test_send_message_saves_to_history_when_consented(client):
    session_id = client.post("/sessions/patient101", json={
        "store_history_consent": True,
    }).json()["id"]

    client.post(f"/sessions/{session_id}/message", json={
        "message": "Tell me about my health",
        "model": "gemini",
        "store_history_consent": True,
    })

    session = client.get(f"/sessions/{session_id}/messages").json()
    roles = [m["role"] for m in session["messages"]]
    assert "user" in roles
    assert "assistant" in roles


def test_send_message_does_not_save_without_consent(client):
    session_id = client.post("/sessions/patient101", json={
        "store_history_consent": False,
    }).json()["id"]

    client.post(f"/sessions/{session_id}/message", json={
        "message": "Tell me about my health",
        "model": "gemini",
    })

    session = client.get(f"/sessions/{session_id}/messages").json()
    assert session["messages"] == []


# ── Safety guardrail via endpoint ─────────────────────────────────────

def test_self_harm_message_returns_crisis_response(client):
    session_id = client.post("/sessions/patient101", json={}).json()["id"]
    r = client.post(f"/sessions/{session_id}/message", json={
        "message": "I want to kill myself",
        "model": "gemini",
    })
    assert r.status_code == 200
    body = r.json()
    assert "9152987821" in body["response"]
    assert body["model_name"] == "safety-guardrail"
    assert body["total_tokens"] == 0


# ── GDPR endpoints ────────────────────────────────────────────────────

def test_gdpr_delete_session(client):
    session_id = client.post("/sessions/patient101", json={}).json()["id"]
    r = client.delete(f"/gdpr/sessions/{session_id}")
    assert r.status_code == 200
    assert r.json()["gdpr_action"] == "session_deleted"
    assert client.get(f"/sessions/{session_id}/messages").status_code == 404


def test_gdpr_delete_session_not_found(client):
    r = client.delete("/gdpr/sessions/nonexistent")
    assert r.status_code == 404


def test_gdpr_delete_all_patient_sessions(client):
    client.post("/sessions/patient101", json={})
    client.post("/sessions/patient101", json={})
    r = client.delete("/gdpr/patient/patient101")
    assert r.status_code == 200
    assert r.json()["sessions_deleted"] == 2
    assert client.get("/sessions/patient101").json() == []


def test_gdpr_update_consent(client):
    session_id = client.post("/sessions/patient101", json={
        "emr_consent": False,
        "store_history_consent": False,
    }).json()["id"]

    r = client.post(f"/gdpr/consent/{session_id}", json={
        "emr_consent": True,
        "store_history_consent": True,
    })
    assert r.status_code == 200
    assert r.json()["emr_consent"] is True
