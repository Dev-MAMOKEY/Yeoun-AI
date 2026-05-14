"""쓰기 함수의 `sessionmaker is None` 분기 단위 테스트.

`USE_DB_MOCK=false` 환경에서 부팅 결함으로 sessionmaker 가 초기화 안 된 케이스
— 이전엔 silent return 이었으나 #56 (delete) 와 본 #59 (전반 정리) 에서 명시적
RuntimeError 로 전환. 본 스위트는 각 쓰기 함수가 실제로 예외를 raise 하는지
검증 (#56 의 dual review code-reviewer P1 conf 85 패턴).

읽기 함수(`get_persona`·`get_persona_interviews`) 는 silent (None/[] 반환) 유지가
graceful degradation 정상 흐름이라 본 스위트의 대상 아님.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from app.config import get_settings
from app.db import repository
from app.db.repository import _raise_db_uninitialized


@pytest.fixture
def db_uninitialized(monkeypatch):
    """USE_DB_MOCK=false + get_sessionmaker None 강제 — 실 DB 분기의 silent-return-차단 경로 활성."""
    monkeypatch.setenv("USE_DB_MOCK", "false")
    get_settings.cache_clear()
    monkeypatch.setattr("app.db.repository.get_sessionmaker", lambda: None)
    yield
    get_settings.cache_clear()


def test_raise_db_uninitialized_helper_raises_runtime_error():
    """헬퍼 단위 — RuntimeError + 메시지에 함수명 포함 + sessionmaker 키워드 노출."""
    with pytest.raises(RuntimeError, match="sessionmaker"):
        _raise_db_uninitialized("test_func", persona_id=uuid4())


def test_raise_db_uninitialized_redacts_pii_keys(caplog):
    """PII 키(user_id·persona_id 등) 가 ERROR 로그에 평문 노출되지 않고 redact 되는지.

    명세서 「안전 정책: 원본 메시지·비식별 패턴 모니터링 전용」 정합성 회귀 방지.
    """
    import logging

    user_id = uuid4()
    persona_id = uuid4()
    with caplog.at_level(logging.ERROR, logger="yeoun"):
        with pytest.raises(RuntimeError):
            _raise_db_uninitialized(
                "insert_safety_log",
                user_id=user_id,
                persona_id=persona_id,
                event_type="crisis_keyword",
            )
    log_text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert str(user_id) not in log_text, "user_id 가 평문 노출됨"
    assert str(persona_id) not in log_text, "persona_id 가 평문 노출됨"
    # 운영 진단 메타는 보존돼야 함.
    assert "crisis_keyword" in log_text
    assert "<redacted>" in log_text


async def test_update_persona_status_raises_when_db_uninitialized(db_uninitialized):
    with pytest.raises(RuntimeError, match="sessionmaker"):
        await repository.update_persona_status(uuid4(), "PROCESSING")


async def test_insert_persona_photo_asset_raises_when_db_uninitialized(db_uninitialized):
    with pytest.raises(RuntimeError, match="sessionmaker"):
        await repository.insert_persona_photo_asset(
            photo_asset_id=uuid4(),
            persona_id=uuid4(),
            filesystem_path="/var/persona/x/photo/x.jpg",
        )


async def test_insert_persona_voice_asset_raises_when_db_uninitialized(db_uninitialized):
    with pytest.raises(RuntimeError, match="sessionmaker"):
        await repository.insert_persona_voice_asset(
            voice_asset_id=uuid4(),
            persona_id=uuid4(),
            original_name="ref.wav",
            filesystem_path="/var/persona/x/voice/ref.wav",
        )


async def test_insert_persona_idle_clip_raises_when_db_uninitialized(db_uninitialized):
    with pytest.raises(RuntimeError, match="sessionmaker"):
        await repository.insert_persona_idle_clip(
            clip_id=uuid4(),
            persona_id=uuid4(),
            sequence_order=0,
            filesystem_path="/var/persona/x/idle/0.mp4",
        )


async def test_insert_safety_log_raises_when_db_uninitialized(db_uninitialized):
    with pytest.raises(RuntimeError, match="sessionmaker"):
        await repository.insert_safety_log(
            logs_id=uuid4(),
            user_id=uuid4(),
            event_type="crisis_keyword",
            action_katen="block",
            deceted_at=datetime.now(timezone.utc),
        )
