"""페르소나 생성 백그라운드 파이프라인 (이슈 #9) 더미 모드 스모크 테스트.

GPU_ENABLED=false + USE_DB_MOCK=true 에서 `process_persona` 가 voice/photo 자원을
받아 ref_text·ref_audio·idle 클립을 생성하고 status='ready' 까지 도달하는지,
재호출 시 멱등 스킵이 동작하는지, 자원 누락 시 status='failed' + error_reason 이
채워지는지 검증한다.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.config import get_settings
from app.db import repository
from app.db.models import PersonaRecord
from app.models.registry import ModelRegistry
from app.pipeline.persona_creation import (
    PersonaProcessingStore,
    process_persona,
)
from app.schemas.persona import ProcessingStep


def _make_persona(persona_id: UUID) -> PersonaRecord:
    return PersonaRecord(
        personas_id=persona_id,
        owner_user_id=uuid4(),
        id=None,
        name="고인 이름",
        nickname="할아버지",
        status="created",
        created_at=datetime.now(timezone.utc),
    )


def _seed_files(persona_dir: Path, persona_id: UUID) -> None:
    """voice + photo 디렉토리에 더미 파일 시드."""
    root = persona_dir / str(persona_id)
    (root / "voice").mkdir(parents=True)
    (root / "voice" / "ref.wav").write_bytes(b"RIFF....WAVE")
    (root / "photo").mkdir(parents=True)
    (root / "photo" / "face.jpg").write_bytes(b"\xff\xd8\xff\xe0")


async def test_persona_processing_store_set_get():
    store = PersonaProcessingStore()
    pid = uuid4()
    await store.set_step(pid, ProcessingStep.TRANSCRIBING)
    state = store.get(pid)
    assert state is not None
    assert state.step is ProcessingStep.TRANSCRIBING
    assert state.error_reason is None
    assert store.count() == 1


async def test_persona_processing_store_failed_carries_reason():
    store = PersonaProcessingStore()
    pid = uuid4()
    await store.set_step(pid, ProcessingStep.FAILED, error_reason="OOM")
    state = store.get(pid)
    assert state is not None
    assert state.step is ProcessingStep.FAILED
    assert state.error_reason == "OOM"


async def test_process_persona_dummy_mode_full_flow(tmp_path: Path):
    settings = get_settings()
    assert settings.gpu_enabled is False
    assert settings.use_db_mock is True

    await repository._reset_mock()
    persona_id = uuid4()
    await repository.mock_seed_persona(_make_persona(persona_id))
    _seed_files(tmp_path, persona_id)

    registry = ModelRegistry(settings)
    await registry.start()
    store = PersonaProcessingStore()
    try:
        await process_persona(
            persona_id,
            registry=registry,
            persona_dir=str(tmp_path),
            store=store,
        )
    finally:
        await registry.stop()

    # status + step
    record = await repository.get_persona(persona_id)
    assert record is not None
    assert record.status == "ready"
    state = store.get(persona_id)
    assert state is not None
    assert state.step is ProcessingStep.READY
    assert state.error_reason is None

    # 산출물
    root = tmp_path / str(persona_id)
    assert (root / "voice_ref" / "ref_text.txt").read_text(encoding="utf-8")
    assert any((root / "voice_ref").glob("ref_audio.*"))
    assert (root / "idle" / "0.mp4").exists()
    assert (root / "idle" / "1.mp4").exists()


async def test_process_persona_idempotent_skip(tmp_path: Path):
    settings = get_settings()

    await repository._reset_mock()
    persona_id = uuid4()
    await repository.mock_seed_persona(_make_persona(persona_id))
    _seed_files(tmp_path, persona_id)

    registry = ModelRegistry(settings)
    await registry.start()
    store = PersonaProcessingStore()
    try:
        # 1회차 — 산출물 생성
        await process_persona(
            persona_id, registry=registry, persona_dir=str(tmp_path), store=store
        )
        idle_0 = tmp_path / str(persona_id) / "idle" / "0.mp4"
        mtime_first = idle_0.stat().st_mtime

        # 2회차 — 멱등 스킵 (mtime 동일)
        await process_persona(
            persona_id, registry=registry, persona_dir=str(tmp_path), store=store
        )
    finally:
        await registry.stop()

    assert idle_0.stat().st_mtime == mtime_first


async def test_process_persona_failure_marks_failed(tmp_path: Path):
    settings = get_settings()

    await repository._reset_mock()
    persona_id = uuid4()
    await repository.mock_seed_persona(_make_persona(persona_id))
    # voice/photo 디렉토리 일부러 미생성 → _pick_voice_file 이 FileNotFoundError

    registry = ModelRegistry(settings)
    await registry.start()
    store = PersonaProcessingStore()
    try:
        await process_persona(
            persona_id, registry=registry, persona_dir=str(tmp_path), store=store
        )
    finally:
        await registry.stop()

    record = await repository.get_persona(persona_id)
    assert record is not None
    assert record.status == "failed"
    state = store.get(persona_id)
    assert state is not None
    assert state.step is ProcessingStep.FAILED
    assert state.error_reason is not None
    assert "FileNotFoundError" in state.error_reason
