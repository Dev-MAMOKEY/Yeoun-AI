"""테스트 공용 픽스처."""

import shutil

import pytest
from fastapi.testclient import TestClient

TEST_TOKEN = "test-internal-token"


@pytest.fixture(autouse=True)
def _stub_ffmpeg_trim(monkeypatch):
    """`_prepare_voice_ref` 의 ffmpeg subprocess 호출을 단순 복사로 우회.

    호스트 venv 에 ffmpeg 미설치 환경에서도 process_persona 흐름 테스트가 통과
    하도록 격리. 운영(도커) 환경의 실제 ffmpeg trim 동작은 라이브 검증 책임 — 본
    fixture 는 테스트 격리 목적이라 trim 의미(첫 8초 자르기) 자체는 검증하지 않음.
    """
    from app.pipeline import persona_creation

    async def _fake_prepare(voice_path, ref_audio_path):
        if ref_audio_path.exists():
            return
        ref_audio_path.parent.mkdir(parents=True, exist_ok=True)
        await _async_copy(voice_path, ref_audio_path)

    async def _async_copy(src, dst):
        import asyncio as _asyncio
        await _asyncio.to_thread(shutil.copy2, src, dst)

    monkeypatch.setattr(persona_creation, "_prepare_voice_ref", _fake_prepare)


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
    from app.db.engine import get_engine, get_sessionmaker

    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()
    yield
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


@pytest.fixture
def client(_settings_env):
    """lifespan 시작/종료까지 트리거하는 TestClient.

    `_settings_env`를 명시 의존으로 두어 픽스처 scope가 바뀌어도 환경 변수
    세팅이 보장되도록 한다.
    """
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
