"""mock 기반 핵심 흐름 통합 E2E.

GPU_ENABLED=false + USE_DB_MOCK=true 에서 명세서 흐름 A(페르소나 생성) → 세션
start/end → 영구 삭제까지 라우터 시퀀스로 검증한다. 개별 단위 테스트가 각
모듈을 커버하고, 본 스위트는 라우터 간 결합·BackgroundTask 동작·DB+FS 정합성을
한 번에 묶어 확인.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.db import repository
from app.db.models import PersonaRecord
from tests.conftest import TEST_TOKEN


def _make_persona(persona_id: UUID, status: str = "created") -> PersonaRecord:
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


@pytest.fixture
def e2e_client(_settings_env, tmp_path, monkeypatch):
    """PERSONA_DIR 을 tmp_path 로 override 한 TestClient."""
    monkeypatch.setenv("PERSONA_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client, tmp_path


def _seed_assets(persona_dir: Path, persona_id: UUID) -> Path:
    root = persona_dir / str(persona_id)
    (root / "voice").mkdir(parents=True)
    (root / "voice" / "ref.wav").write_bytes(b"RIFF....WAVE")
    (root / "photo").mkdir()
    (root / "photo" / "face.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    return root


def test_e2e_persona_session_delete_flow(e2e_client):
    """명세 흐름 A → B → 삭제 한 줄짜리 시퀀스 검증.

    1. 페르소나 시드 (status='created') + voice/photo 자원 시드
    2. POST /process → 202, TestClient 가 BackgroundTask 를 응답 후 동기 실행
    3. GET /status → status='ready', step=READY
    4. GET /idle-clips → 더미 placeholder mp4 2개
    5. POST /sessions/start → session_id
    6. POST /sessions/{id}/end → cleaned_messages=0
    7. DELETE /personas/{id} → 200, DB+FS 모두 제거
    8. GET /status → 404
    """
    client, persona_dir = e2e_client
    pid = uuid4()
    auth = {"Authorization": f"Bearer {TEST_TOKEN}"}

    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))
    root = _seed_assets(persona_dir, pid)

    # 2. process
    res = client.post(f"/internal/personas/{pid}/process", headers=auth)
    assert res.status_code == 202

    # 3. status — TestClient 가 BackgroundTask 를 response 직후 동기 실행하므로
    #    여기서 status 가 'ready' 여야 함.
    res = client.get(f"/internal/personas/{pid}/status", headers=auth)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["status"] == "ready", (
        f"BackgroundTask 완료 후 status 불일치: status={data['status']!r}, "
        f"step={data.get('step')!r}, error_reason={data.get('error_reason')!r}"
    )
    assert data["step"] == "ready", f"step 불일치: {data.get('step')!r}"

    # 4. idle-clips
    res = client.get(f"/internal/personas/{pid}/idle-clips", headers=auth)
    assert res.status_code == 200
    clips = res.json()["data"]["clips"]
    assert len(clips) == 2
    assert {c["index"] for c in clips} == {0, 1}

    # 5. 세션 start
    res = client.post(
        "/internal/sessions/start",
        json={"user_id": str(uuid4()), "persona_id": str(pid)},
        headers=auth,
    )
    assert res.status_code == 200
    sid = res.json()["data"]["session_id"]

    # 6. 세션 end
    res = client.post(f"/internal/sessions/{sid}/end", headers=auth)
    assert res.status_code == 200
    assert res.json()["data"]["cleaned_messages"] == 0

    # 7. 페르소나 삭제
    res = client.delete(f"/internal/personas/{pid}", headers=auth)
    assert res.status_code == 200
    assert not root.exists()

    # 8. /status 404 — DB 행 사라짐
    res = client.get(f"/internal/personas/{pid}/status", headers=auth)
    assert res.status_code == 404
