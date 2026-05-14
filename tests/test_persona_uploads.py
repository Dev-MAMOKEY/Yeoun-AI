"""페르소나 사진·음성 업로드 라우트 스모크 테스트.

multipart 업로드 → 200·404·409·413·415 분기 검증. PERSONA_DIR 은 tmp_path 로
fixture override.
"""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from app.db import repository
from app.db.models import PersonaRecord
from tests.conftest import TEST_TOKEN


def _make_persona(persona_id: UUID, status: str = "DRAFT") -> PersonaRecord:
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
def upload_client(_settings_env, tmp_path, monkeypatch):
    """PERSONA_DIR override + 작은 max_upload_bytes 로 413 테스트 가능."""
    monkeypatch.setenv("PERSONA_DIR", str(tmp_path))
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "1024")  # 테스트용 1KiB cap
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        yield client, tmp_path


# --- 정상 흐름 ---------------------------------------------------------------


def test_upload_photo_200_writes_file(upload_client):
    client, persona_dir = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))

    payload = b"\xff\xd8\xff\xe0" + b"\x00" * 100
    res = client.post(
        f"/internal/personas/{pid}/photo",
        files={"file": ("face.jpg", payload, "image/jpeg")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["path"] == f"{pid}/photo/face.jpg"
    assert data["size_bytes"] == len(payload)
    assert (persona_dir / str(pid) / "photo" / "face.jpg").read_bytes() == payload
    # persona_photo_assets 메타 행 1건 INSERT.
    assets = _run(repository.mock_get_photo_assets())
    assert len(assets) == 1
    assert assets[0]["persona_id"] == pid


def test_upload_voice_200_writes_file(upload_client):
    client, persona_dir = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))

    # WAV 파일 RIFF header 로 시작 — magic byte 검사 통과.
    payload = b"RIFF" + b"\x00" * 100 + b"WAVE"
    res = client.post(
        f"/internal/personas/{pid}/voice",
        files={"file": ("ref.wav", payload, "audio/wav")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    assert (persona_dir / str(pid) / "voice" / "ref.wav").read_bytes() == payload
    # persona_voice_assets 메타 행 1건 INSERT — original_name 보존 검증.
    assets = _run(repository.mock_get_voice_assets())
    assert len(assets) == 1
    assert assets[0]["original_name"] == "ref.wav"


# --- 에러 분기 ---------------------------------------------------------------


def test_upload_404_when_persona_missing(upload_client):
    client, _ = upload_client
    _run(repository._reset_mock())
    res = client.post(
        f"/internal/personas/{uuid4()}/photo",
        files={"file": ("face.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "NOT_FOUND"


def test_upload_409_when_processing(upload_client):
    client, _ = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    rec = _make_persona(pid).model_copy(update={"status": "PROCESSING"})
    _run(repository.mock_seed_persona(rec))

    res = client.post(
        f"/internal/personas/{pid}/photo",
        files={"file": ("face.jpg", b"\xff\xd8\xff\xe0", "image/jpeg")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "CONFLICT"


def test_upload_413_when_too_large(upload_client):
    client, _ = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))

    # fixture 에서 MAX_UPLOAD_BYTES=1024 — 2KiB 페이로드는 초과. JPEG 매직 prefix 로
    # magic byte 검사는 통과시키고 크기에서 차단되도록 한다.
    payload = b"\xff\xd8\xff\xe0" + b"\x00" * 2048
    res = client.post(
        f"/internal/personas/{pid}/photo",
        files={"file": ("big.jpg", payload, "image/jpeg")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


def test_upload_415_when_unsupported_type(upload_client):
    client, _ = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))

    res = client.post(
        f"/internal/personas/{pid}/photo",
        files={"file": ("not-image.txt", b"hello", "text/plain")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_upload_415_when_magic_byte_mismatch(upload_client):
    """content-type 은 image/jpeg 라 호출했지만 실제 바이트는 텍스트 — 매직 가드가 차단."""
    client, _ = upload_client
    pid = uuid4()
    _run(repository._reset_mock())
    _run(repository.mock_seed_persona(_make_persona(pid)))

    res = client.post(
        f"/internal/personas/{pid}/photo",
        files={"file": ("forged.jpg", b"%PDF-1.4 not really an image", "image/jpeg")},
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
