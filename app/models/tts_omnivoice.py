"""OmniVoiceTTS — `k2-fsa/OmniVoice` zero-shot 음성 클로닝 TTS 래퍼.

- 부팅 시 가중치를 GPU 에 적재해 상주한다 (Qwen3-0.6B 베이스 + TTS 모듈, ~2GB).
- `generate(text, ref_audio, ref_text)` API 로 24kHz wav 를 합성한다.
- `GPU_ENABLED=false` 모드는 무음 wav (1초 0배열) 를 출력 경로에 쓰고 반환.
- GPU 직렬화는 호출자(#9·#10) 가 `ModelRegistry.gpu_semaphore` 로 보호.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal

logger = logging.getLogger("yeoun")

# OmniVoice 가 반환하는 wav 의 표본화 주파수.
_SAMPLE_RATE = 24_000
# 더미 모드에서 만들 무음 wav 길이 — 1초.
_DUMMY_DURATION_SECONDS = 1.0

TtsStatus = Literal["not_loaded", "loading", "loaded", "error"]


class OmniVoiceTTS:
    """OmniVoice TTS 래퍼."""

    def __init__(
        self,
        *,
        model_path: str | None,
        gpu_enabled: bool,
    ) -> None:
        self._model_path = model_path
        self._gpu_enabled = gpu_enabled
        self._status: TtsStatus = "not_loaded"
        self._model = None

    # --- 라이프사이클 -------------------------------------------------------
    async def load(self) -> None:
        """가중치를 GPU 에 적재. 더미 모드는 즉시 `loaded`."""
        if not self._gpu_enabled:
            logger.info("GPU_ENABLED=false → OmniVoiceTTS 더미 모드")
            self._status = "loaded"
            return

        self._status = "loading"
        if not self._model_path:
            self._status = "error"
            raise RuntimeError(
                "TTS_MODEL_PATH 가 비어 있습니다. 가중치 디렉토리 또는 HF repo ID 를 "
                "설정하거나 GPU_ENABLED=false 로 더미 모드를 사용하십시오."
            )
        # 실 로드는 다음 커밋에서 채운다.
        raise NotImplementedError("OmniVoice 실 로드 분기는 다음 커밋에서 구현.")

    async def unload(self) -> None:
        """모델 자원 해제. 더미 모드면 no-op."""
        if not self._gpu_enabled:
            self._status = "not_loaded"
            return
        self._model = None
        self._status = "not_loaded"
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # --- 상태 --------------------------------------------------------------
    @property
    def status(self) -> TtsStatus:
        return self._status

    # --- 합성 ---------------------------------------------------------------
    async def synthesize(
        self,
        *,
        text: str,
        ref_audio_path: str | Path,
        ref_text: str,
        output_path: str | Path,
    ) -> Path:
        """텍스트 + 참조 음성/전사 → 24kHz wav 파일 생성, 출력 경로 반환.

        Args:
            text: 합성할 한국어(또는 다국어) 텍스트.
            ref_audio_path: 페르소나의 reference wav 파일 경로.
            ref_text: reference wav 의 한국어 전사 (Gemma 4 가 미리 생성).
            output_path: 합성 결과 wav 를 저장할 경로 (호출자가 결정).
        """
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)

        if not self._gpu_enabled:
            await asyncio.to_thread(_write_silent_wav, target)
            return target

        if self._model is None:
            raise RuntimeError("OmniVoiceTTS 가 로드되지 않았습니다.")
        # 실 합성 분기는 다음 커밋에서 구현.
        raise NotImplementedError("OmniVoice 실 합성 분기는 다음 커밋에서 구현.")


def _write_silent_wav(target: Path) -> None:
    """더미 모드용 무음 wav (1초, 24kHz, float32 zeros) 를 쓴다."""
    import numpy as np
    import soundfile as sf

    samples = int(_SAMPLE_RATE * _DUMMY_DURATION_SECONDS)
    sf.write(str(target), np.zeros(samples, dtype="float32"), _SAMPLE_RATE)
