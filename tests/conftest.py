"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

TEST_TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def _settings_env(monkeypatch):
    """Force every test to run with a known internal token and dev-friendly
    toggles. `get_settings` is `lru_cache`-d so we clear the cache around
    each test to pick up the patched environment.
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
    """TestClient with lifespan startup/shutdown triggered.

    Declares `_settings_env` as an explicit dependency so test environment
    setup is guaranteed regardless of fixture scope changes in the future.
    """
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
