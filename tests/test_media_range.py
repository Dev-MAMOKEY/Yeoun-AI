"""미디어 스트리밍 라우터 + Range helper 스모크 (이슈 #11).

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


def _make_persona(persona_id: UUID, status: str = "ready") -> PersonaRecord:
    return PersonaRecord(
        personas_id=persona_id,
        owner_user_id=uuid4(),
        id=None,
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
    _seed_idle(persona_dir, pid)

    res = client.get(
        f"/internal/personas/{pid}/idle-clips/0",
        headers={
            "Authorization": f"Bearer {TEST_TOKEN}",
            "Range": "bytes=99999-",
        },
    )
    assert res.status_code == 416
    assert "content-range" in res.headers


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
