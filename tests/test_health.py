"""`GET /internal/health` 스모크 테스트."""

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
    # USE_DB_MOCK=true 환경에서는 lifespan 의 db_ping 이 즉시 ok 반환.
    assert data["db"] == "ok"
    assert data["sessions"] == 0
    assert set(data["models"].keys()) == {"llm", "tts", "ditto"}
    # GPU_ENABLED=false 모드에서 모든 모델이 더미 모드로 'loaded' 표시.
    # Ditto 의 'loaded' 는 "경로 검증 통과, render 호출 가능" 의 의미.
    assert data["models"]["llm"] == "loaded"
    assert data["models"]["tts"] == "loaded"
    assert data["models"]["ditto"] == "loaded"
    assert data["uptime_seconds"] >= 0
