"""Smoke tests for `GET /internal/health`."""

from tests.conftest import TEST_TOKEN


def test_health_rejects_missing_token(client):
    response = client.get("/internal/health")

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["data"] is None
    assert body["error"]["code"] == "MISSING_TOKEN"
    assert response.headers.get("WWW-Authenticate") == "Bearer"


def test_health_rejects_wrong_token(client):
    response = client.get(
        "/internal/health",
        headers={"Authorization": "Bearer wrong-token"},
    )

    assert response.status_code == 401
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "INVALID_TOKEN"


def test_health_returns_envelope_on_success(client):
    response = client.get(
        "/internal/health",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["error"] is None

    data = body["data"]
    assert data["status"] in {"ok", "degraded", "starting"}
    assert data["version"] == "0.1.0"
    assert data["gpu_enabled"] is False
    assert data["db"] == "mock"
    assert data["sessions"] == 0
    assert set(data["models"].keys()) == {"llm", "tts", "ditto"}
    assert all(v == "not_loaded" for v in data["models"].values())
    assert data["uptime_seconds"] >= 0
