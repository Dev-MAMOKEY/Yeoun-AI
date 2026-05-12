"""DittoTalkingHead 더미 모드(GPU_ENABLED=false) 스모크 테스트.

실 GPU 렌더는 운영자가 vendor clone + checkpoints 다운로드 후 수행. 본 테스트는
더미 분기가 placeholder mp4 를 정상 경로에 쓰고 라이프사이클 상태가 약속대로
전이되는지만 확인한다.
"""

from pathlib import Path

from app.config import get_settings
from app.models.registry import ModelRegistry
from app.models.talkinghead_ditto import DittoTalkingHead


async def test_ditto_dummy_load_then_unload():
    ditto = DittoTalkingHead(
        vendor_dir=None,
        data_root=None,
        cfg_pkl=None,
        gpu_enabled=False,
    )

    assert ditto.status == "not_loaded"
    await ditto.load()
    assert ditto.status == "loaded"

    await ditto.unload()
    assert ditto.status == "not_loaded"


async def test_ditto_dummy_render_speak_writes_placeholder_mp4(tmp_path: Path):
    ditto = DittoTalkingHead(
        vendor_dir=None, data_root=None, cfg_pkl=None, gpu_enabled=False
    )
    await ditto.load()

    target = tmp_path / "persona" / "session" / "msg.mp4"
    result = await ditto.render_speak(
        image_path="/tmp/nonexistent_image.png",  # 더미 모드라 실 존재 불필요
        audio_path="/tmp/nonexistent_audio.wav",
        output_path=target,
    )

    assert result == target
    assert target.exists()
    # placeholder 가 mp4 의 ftyp box 로 시작해야 한다.
    head = target.read_bytes()[:12]
    assert b"ftyp" in head


async def test_ditto_dummy_render_idle_creates_output(tmp_path: Path):
    ditto = DittoTalkingHead(
        vendor_dir=None, data_root=None, cfg_pkl=None, gpu_enabled=False
    )
    await ditto.load()

    target = tmp_path / "idle.mp4"
    result = await ditto.render_idle(
        image_path="/tmp/nonexistent_image.png",
        output_path=target,
        duration_seconds=3.0,
    )

    assert result == target
    assert target.exists()


async def test_registry_lifecycle_includes_ditto_in_dummy_mode():
    settings = get_settings()
    assert settings.gpu_enabled is False

    registry = ModelRegistry(settings)
    await registry.start()

    assert registry.ditto is not None
    assert registry.ditto.status == "loaded"
    # Ditto 전용 세마포어가 별도로 1건 허용.
    assert registry.ditto_semaphore is not registry.gpu_semaphore

    await registry.stop()
    assert registry.ditto is None
