"""테스트 공용 픽스처."""

import pytest
from fastapi.testclient import TestClient

TEST_TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    """모든 테스트가 알려진 내부 토큰·개발 친화 토글로 동작하도록 강제.

    `get_settings`가 `lru_cache`라 패치된 환경 변수를 반영하려면 캐시를
    매 테스트 전후로 비워야 한다.
    """
    monkeypatch.setenv("INTERNAL_TOKEN", TEST_TOKEN)
    monkeypatch.setenv("USE_DB_MOCK", "true")
    monkeypatch.setenv("GPU_ENABLED", "false")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from app.config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client(_settings_env):
    """lifespan 시작/종료까지 트리거하는 TestClient.

    `_settings_env`를 명시 의존으로 두어 픽스처 scope가 바뀌어도 환경 변수
    세팅이 보장되도록 한다.
    """
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
