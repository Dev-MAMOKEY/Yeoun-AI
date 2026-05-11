"""SessionStore 스모크 테스트.

이슈 #4 「단위 테스트 자리」 — TTL sweeper·CRUD 핵심만 검증. 본격 단위 테스트
는 이슈 #13 에서 확장한다.
"""

import asyncio
import time
from uuid import uuid4

import pytest

from app.sessions.schemas import MessageRecord, SessionState
from app.sessions.store import SessionStore


def _make_session(now: float | None = None) -> SessionState:
    now = now or time.time()
    return SessionState(
        session_id=uuid4(),
        user_id=uuid4(),
        persona_id=uuid4(),
        system_prompt="테스트용 시스템 프롬프트",
        started_at=now,
        last_activity_at=now,
    )


def test_create_and_get():
    store = SessionStore()
    session = _make_session()

    store.create(session)

    assert store.get(session.session_id) is session
    assert store.count() == 1


def test_get_missing_returns_none():
    store = SessionStore()
    assert store.get(uuid4()) is None


def test_touch_updates_last_activity():
    store = SessionStore()
    session = _make_session(now=time.time() - 100)
    store.create(session)

    assert store.touch(session.session_id) is True
    assert session.last_activity_at >= time.time() - 1


def test_touch_missing_session_returns_false():
    store = SessionStore()
    assert store.touch(uuid4()) is False


def test_add_message_appends_history():
    store = SessionStore()
    session = _make_session()
    store.create(session)

    msg = MessageRecord(role="user", text="안녕하세요", created_at=time.time())
    assert store.add_message(session.session_id, msg) is True
    assert session.history == [msg]


def test_end_removes_session():
    store = SessionStore()
    session = _make_session()
    store.create(session)

    removed = store.end(session.session_id)
    assert removed is session
    assert store.count() == 0
    assert store.get(session.session_id) is None


def test_sweep_once_removes_only_expired():
    # TTL 60 초로 좁혀 만료 1건 + 살아있는 1건이 섞이도록 구성.
    store = SessionStore(ttl_seconds=60, sweep_interval=999)
    now = time.time()
    expired = _make_session(now=now - 120)
    fresh = _make_session(now=now)
    store.create(expired)
    store.create(fresh)

    removed_count = store._sweep_once()

    assert removed_count == 1
    assert store.get(expired.session_id) is None
    assert store.get(fresh.session_id) is fresh


@pytest.mark.asyncio
async def test_sweeper_lifecycle_start_and_stop():
    # 매우 짧은 sweep_interval 로 백그라운드 태스크가 실제 도는지만 확인.
    store = SessionStore(ttl_seconds=0.1, sweep_interval=0.05)
    expired = _make_session(now=time.time() - 10)
    store.create(expired)

    await store.start()
    # 한 사이클 이상 sweep 이 돌도록 대기.
    await asyncio.sleep(0.2)
    await store.stop()

    assert store.get(expired.session_id) is None
    assert store.count() == 0


@pytest.mark.asyncio
async def test_start_is_idempotent():
    store = SessionStore(sweep_interval=999)
    await store.start()
    first_task = store._sweeper_task
    await store.start()
    assert store._sweeper_task is first_task
    await store.stop()
