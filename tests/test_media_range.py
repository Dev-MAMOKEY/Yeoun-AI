"""미디어 스트리밍 라우터 + Range helper 스모크.

`parse_range` / `safe_resolve` 단위 + `/internal/personas/{id}/idle-clips/{idx}` ·
`/internal/sessions/{id}/messages/{msg}/media` 통합. PERSONA_DIR 은 tmp_path 로
fixture override 해 산출물 시드.
"""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.db import repository
from app.db.models import PersonaRecord
from app.sessions.schemas import SessionState
from app.storage.filesystem import (
    RangeNotSatisfiable,
    parse_range,
    safe_resolve,
)
from tests.conftest import TEST_TOKEN


def _make_persona(persona_id: UUID, status: str = "READY") -> PersonaRecord:
    return PersonaRecord(
        personas_id=persona_id,
        owner_user_id=uuid4(),
        name="고인",
        nickname="할아버지",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- parse_range -------------------------------------------------------------


def test_parse_range_none_when_header_absent():
    assert parse_range(None, 1000) is None
    assert parse_range("", 1000) is None


def test_parse_range_full_form():
    assert parse_range("bytes=0-99", 1000) == (0, 99)


def test_parse_range_open_end_uses_file_end():
    assert parse_range("bytes=500-", 1000) == (500, 999)


def test_parse_range_suffix_form():
    # 마지막 200 바이트.
    assert parse_range("bytes=-200", 1000) == (800, 999)


def test_parse_range_rejects_out_of_bounds():
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=2000-3000", 1000)


def test_parse_range_rejects_multi_range():
    with pytest.raises(RangeNotSatisfiable):
        parse_range("bytes=0-100,200-300", 1000)


# --- safe_resolve ------------------------------------------------------------


def test_safe_resolve_keeps_inside_root(tmp_path: Path):
    target = tmp_path / "x" / "y.txt"
    resolved = safe_resolve(tmp_path, "x", "y.txt")
    assert resolved == target.resolve()


def test_safe_resolve_blocks_dotdot_escape(tmp_path: Path):
    with pytest.raises(PermissionError):
        safe_resolve(tmp_path, "..", "..", "etc", "passwd")


# --- 라우터 통합 -------------------------------------------------------------


@pytest.fixture
def media_client(_settings_env, tmp_path, monkeypatch):
    """PERSONA_DIR 을 tmp_path 로 override 한 TestClient."""
    monkeypatch.setenv("PERSONA_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client, tmp_path


def _seed_idle(persona_dir: Path, persona_id: UUID) -> bytes:
    root = persona_dir / str(persona_id) / "idle"
    root.mkdir(parents=True)
    payload = b"\x00" * 1000 + b"END"
    (root / "0.mp4").write_bytes(payload)
    return payload


def test_get_idle_clip_full_200(media_client):
    client, persona_dir = media_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    payload = _seed_idle(persona_dir, pid)

    res = client.get(
        f"/internal/personas/{pid}/idle-clips/0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert res.headers["accept-ranges"] == "bytes"
    assert res.content == payload


def test_get_idle_clip_range_206(media_client):
    client, persona_dir = media_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    payload = _seed_idle(persona_dir, pid)

    res = client.get(
        f"/internal/personas/{pid}/idle-clips/0",
        headers={
            "Authorization": f"Bearer {TEST_TOKEN}",
            "Range": "bytes=10-19",
        },
    )
    assert res.status_code == 206
    assert res.headers["content-range"] == f"bytes 10-19/{len(payload)}"
    assert res.content == payload[10:20]


def test_get_idle_clip_404_missing_persona(media_client):
    client, _ = media_client
    _run(repository._reset_mock())
    res = client.get(
        f"/internal/personas/{uuid4()}/idle-clips/0",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404


def test_get_idle_clip_416_out_of_range(media_client):
    client, persona_dir = media_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    payload = _seed_idle(persona_dir, pid)

    res = client.get(
        f"/internal/personas/{pid}/idle-clips/0",
        headers={
            "Authorization": f"Bearer {TEST_TOKEN}",
            "Range": "bytes=99999-",
        },
    )
    assert res.status_code == 416
    # RFC 9110 §15.5.17 — 416 응답은 `bytes */N` 형식으로 자원 크기 노출.
    assert res.headers["content-range"] == f"bytes */{len(payload)}"


def test_get_idle_clip_invalid_index_404(media_client):
    client, _ = media_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    res = client.get(
        f"/internal/personas/{pid}/idle-clips/9",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404


# --- session media 라우트 --------------------------------------------------


def _seed_session(client, persona_dir: Path, persona_id: UUID, session_id: UUID, message_id: UUID, *, video: bool = True) -> bytes:
    """speak/{sid}/{mid}.{mp4|wav} 시드 + 라우터에서 store.get 통과하도록 세션 등록."""
    suffix = "mp4" if video else "wav"
    speak_dir = persona_dir / str(persona_id) / "speak" / str(session_id)
    speak_dir.mkdir(parents=True)
    payload = b"\x11" * 500 + b"VIDEO"
    (speak_dir / f"{message_id}.{suffix}").write_bytes(payload)
    # SessionStore 에 등록 — TestClient lifespan 안에서 app.state.session_store 존재.
    store = client.app.state.session_store
    store.create(
        SessionState(
            session_id=session_id,
            user_id=uuid4(),
            persona_id=persona_id,
            system_prompt="prompt",
            started_at=time.time(),
            last_activity_at=time.time(),
        )
    )
    return payload


def test_get_session_media_full_200(media_client):
    client, persona_dir = media_client
    pid, sid, mid = uuid4(), uuid4(), uuid4()
    payload = _seed_session(client, persona_dir, pid, sid, mid, video=True)

    res = client.get(
        f"/internal/sessions/{sid}/messages/{mid}/media",
        params={"kind": "video"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert res.headers["accept-ranges"] == "bytes"
    assert res.content == payload


def test_get_session_media_range_206(media_client):
    client, persona_dir = media_client
    pid, sid, mid = uuid4(), uuid4(), uuid4()
    payload = _seed_session(client, persona_dir, pid, sid, mid, video=True)

    res = client.get(
        f"/internal/sessions/{sid}/messages/{mid}/media",
        params={"kind": "video"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}", "Range": "bytes=0-49"},
    )
    assert res.status_code == 206
    assert res.headers["content-range"] == f"bytes 0-49/{len(payload)}"
    assert res.content == payload[:50]


def test_get_session_media_audio_kind(media_client):
    client, persona_dir = media_client
    pid, sid, mid = uuid4(), uuid4(), uuid4()
    payload = _seed_session(client, persona_dir, pid, sid, mid, video=False)

    res = client.get(
        f"/internal/sessions/{sid}/messages/{mid}/media",
        params={"kind": "audio"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("audio/")
    assert res.content == payload


def test_get_session_media_404_unknown_session(media_client):
    client, _ = media_client
    res = client.get(
        f"/internal/sessions/{uuid4()}/messages/{uuid4()}/media",
        params={"kind": "video"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404


def test_get_session_media_invalid_kind_422(media_client):
    client, _ = media_client
    res = client.get(
        f"/internal/sessions/{uuid4()}/messages/{uuid4()}/media",
        params={"kind": "subtitle"},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 422
