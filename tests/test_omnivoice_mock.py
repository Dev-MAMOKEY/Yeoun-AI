"""OmniVoiceTTS 더미 모드(GPU_ENABLED=false) 스모크 테스트.

실 GPU 합성은 운영자가 수행. 본 테스트는 더미 분기가 무음 wav 를 정상 출력
경로에 쓰는지, 라이프사이클 상태가 약속대로 전이되는지만 확인한다.
"""

from pathlib import Path

import soundfile as sf

from app.config import get_settings
from app.models.registry import ModelRegistry
from app.models.tts_omnivoice import OmniVoiceTTS


async def test_omnivoice_dummy_load_then_unload():
    tts = OmniVoiceTTS(model_path=None, gpu_enabled=False)

    assert tts.status == "not_loaded"
    await tts.load()
    assert tts.status == "loaded"

    await tts.unload()
    assert tts.status == "not_loaded"


async def test_omnivoice_dummy_synthesize_writes_silent_wav(tmp_path: Path):
    tts = OmniVoiceTTS(model_path=None, gpu_enabled=False)
    await tts.load()

    # 호출자가 깊은 경로를 줘도 parents=True 로 자동 생성하는지 확인.
    target = tmp_path / "persona" / "session" / "msg.wav"
    result = await tts.synthesize(
        text="안녕하세요",
        ref_audio_path="/tmp/nonexistent_ref.wav",  # 더미 모드라 ref 파일 실 존재 불필요
        ref_text="안녕하세요",
        output_path=target,
    )

    assert result == target
    assert target.exists()
    data, rate = sf.read(str(target))
    assert rate == 24_000
    # 1초 분량 무음 (오차 허용).
    assert len(data) == 24_000
    assert float(abs(data).max()) == 0.0


async def test_registry_lifecycle_includes_tts_in_dummy_mode():
    settings = get_settings()
    assert settings.gpu_enabled is False

    registry = ModelRegistry(settings)
    await registry.start()

    assert registry.tts is not None
    assert registry.tts.status == "loaded"

    await registry.stop()
    assert registry.tts is None
