"""DELETE /internal/personas/{id} + safe_rmtree 스모크."""

import asyncio
import shutil
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.db import repository
from app.db.models import PersonaRecord
from app.storage.filesystem import safe_rmtree
from tests.conftest import TEST_TOKEN


def _make_persona(persona_id: UUID) -> PersonaRecord:
    return PersonaRecord(
        personas_id=persona_id,
        owner_user_id=uuid4(),
        name="고인",
        nickname="할아버지",
        status="ready",
        created_at=datetime.now(timezone.utc),
    )


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# --- safe_rmtree --------------------------------------------------------------


async def test_safe_rmtree_deletes_existing(tmp_path: Path):
    target = tmp_path / "persona-1"
    (target / "idle").mkdir(parents=True)
    (target / "idle" / "0.mp4").write_bytes(b"x")

    deleted = await safe_rmtree(tmp_path, "persona-1")
    assert deleted is True
    assert not target.exists()


async def test_safe_rmtree_returns_false_when_missing(tmp_path: Path):
    deleted = await safe_rmtree(tmp_path, "ghost")
    assert deleted is False


async def test_safe_rmtree_blocks_dotdot_escape(tmp_path: Path):
    with pytest.raises(PermissionError):
        await safe_rmtree(tmp_path, "..", "..", "etc")


# --- DELETE persona 라우터 ----------------------------------------------------


@pytest.fixture
def delete_client(_settings_env, tmp_path, monkeypatch):
    """PERSONA_DIR 을 tmp_path 로 override 한 TestClient."""
    monkeypatch.setenv("PERSONA_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client, tmp_path


def _seed_persona(persona_dir: Path, persona_id: UUID) -> Path:
    root = persona_dir / str(persona_id)
    (root / "voice").mkdir(parents=True)
    (root / "voice" / "ref.wav").write_bytes(b"RIFF")
    (root / "idle").mkdir()
    (root / "idle" / "0.mp4").write_bytes(b"\x00" * 100)
    return root


def test_delete_persona_removes_db_and_fs(delete_client):
    client, persona_dir = delete_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    root = _seed_persona(persona_dir, pid)

    res = client.delete(
        f"/internal/personas/{pid}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert res.json()["success"] is True
    # DB 행 제거
    assert _run(repository.get_persona(pid)) is None
    # FS 삭제
    assert not root.exists()


def test_delete_persona_404_when_missing(delete_client):
    client, _ = delete_client
    _run(repository._reset_mock())
    res = client.delete(
        f"/internal/personas/{uuid4()}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404


def test_delete_persona_no_fs_dir_still_succeeds(delete_client):
    """FS 디렉토리가 없어도 (이미 정리된 상태) DB 행만 제거 후 200."""
    client, _ = delete_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    # 디렉토리 시드 안 함.

    res = client.delete(
        f"/internal/personas/{pid}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert _run(repository.get_persona(pid)) is None


def test_delete_persona_processing_409(delete_client):
    """processing 상태 페르소나는 삭제 차단 — race 방지."""
    client, _ = delete_client
    pid = uuid4()
    _run(repository._reset_mock())
    rec = _make_persona(pid)
    # status 만 processing 으로 덮어쓰기.
    rec = rec.model_copy(update={"status": "processing"})
    _run(repository.mock_seed_persona(rec))

    res = client.delete(
        f"/internal/personas/{pid}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "CONFLICT"
    # DB 행 보존
    assert _run(repository.get_persona(pid)) is not None


def test_delete_persona_fs_failure_leaves_db(delete_client, monkeypatch):
    """`safe_rmtree` 가 OSError 면 500 + DB 행 그대로 — 고아 없음 보장."""
    client, persona_dir = delete_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    _seed_persona(persona_dir, pid)

    async def fake_rmtree(root, *parts):  # noqa: ARG001
        raise OSError("simulated FS failure")

    monkeypatch.setattr("app.routers.personas.safe_rmtree", fake_rmtree)
    res = client.delete(
        f"/internal/personas/{pid}",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 500
    assert res.json()["error"]["code"] == "DELETE_FAILED"
    # DB 행 보존 (롤백 효과)
    assert _run(repository.get_persona(pid)) is not None
