"""GemmaLLM·ModelRegistry 더미 모드(GPU_ENABLED=false) 스모크 테스트.

실 GPU 추론 검증은 GPU 인스턴스에서 운영자가 수행. 본 테스트는 더미 분기가
안전하게 통과하고 약속된 더미 응답을 반환하는지만 확인한다.
"""

from app.config import get_settings
from app.models.llm_gemma import GemmaLLM
from app.models.registry import ModelRegistry


async def test_gemma_dummy_load_then_unload():
    llm = GemmaLLM(awq_path=None, bf16_path=None, gpu_enabled=False)

    assert llm.status == "not_loaded"
    await llm.load()
    assert llm.status == "loaded"
    # 더미 모드는 가중치 variant 를 설정하지 않는다.
    assert llm.variant is None

    await llm.unload()
    assert llm.status == "not_loaded"


async def test_gemma_dummy_transcribe_returns_filename_hint():
    llm = GemmaLLM(awq_path=None, bf16_path=None, gpu_enabled=False)
    await llm.load()

    result = await llm.transcribe("/tmp/sample.wav")

    assert "더미 전사" in result
    assert "sample.wav" in result


async def test_gemma_dummy_stream_yields_tokens():
    llm = GemmaLLM(awq_path=None, bf16_path=None, gpu_enabled=False)
    await llm.load()

    collected: list[str] = []
    async for token in llm.stream_response(
        system_prompt="시스템 프롬프트 (더미)",
        history=[],
        user_audio_path="/tmp/a.wav",
    ):
        collected.append(token)

    assert len(collected) > 0
    joined = "".join(collected)
    assert "더미 응답" in joined
    assert "GPU_ENABLED=false" in joined


async def test_registry_lifecycle_in_dummy_mode():
    # `_settings_env` autouse 가 GPU_ENABLED=false 를 박아둠.
    settings = get_settings()
    assert settings.gpu_enabled is False

    registry = ModelRegistry(settings)
    await registry.start()

    assert registry.llm is not None
    assert registry.llm.status == "loaded"
    # GPU 세마포어가 1건만 허용하도록 초기화돼 있다.
    assert registry.gpu_semaphore._value == 1  # type: ignore[attr-defined]

    await registry.stop()
    assert registry.llm is None
