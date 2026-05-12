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

        try:
            await asyncio.to_thread(self._load_model, self._model_path)
            self._status = "loaded"
        except Exception:
            logger.exception("OmniVoice 로드 실패")
            self._status = "error"
            raise

    def _load_model(self, path: str) -> None:
        """동기 로드 — `asyncio.to_thread` 안에서 호출."""
        import torch
        from omnivoice import OmniVoice  # type: ignore[import-not-found]

        logger.info("OmniVoice 로드 시작: %s", path)
        # Gemma 와 동일한 `device_map="auto"` 정책으로 단일 GPU 환경에서도 0번
        # 디바이스에 올라가고, 다중 GPU 환경에선 accelerate 가 알아서 분산.
        self._model = OmniVoice.from_pretrained(
            path,
            device_map="auto",
            dtype=torch.float16,
        )
        logger.info("OmniVoice 로드 완료")

    async def unload(self) -> None:
        """모델 자원 해제. 더미 모드면 no-op."""
        if not self._gpu_enabled:
            self._status = "not_loaded"
            return
        # 상태 전이를 모델 해제보다 먼저 — 호출 사이에 await 가 끼어들어도 외부
        # observer 가 `loaded` 인데 `_model is None` 인 중간 상태를 보지 못하게 한다.
        self._status = "not_loaded"
        self._model = None
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

        if not self._gpu_enabled:
            # mkdir + 파일 쓰기를 helper 안에서 to_thread 로 한 번에 처리 — async
            # 본문이 동기 disk I/O 로 event loop 를 점유하지 않게 한다.
            await asyncio.to_thread(_write_silent_wav, target)
            return target

        if self._model is None:
            raise RuntimeError("OmniVoiceTTS 가 로드되지 않았습니다.")

        return await asyncio.to_thread(
            self._run_synthesize, text, str(ref_audio_path), ref_text, target
        )

    def _run_synthesize(
        self,
        text: str,
        ref_audio_path: str,
        ref_text: str,
        target: Path,
    ) -> Path:
        """동기 합성 — `asyncio.to_thread` 안에서 호출.

        `OmniVoice.generate` 는 `list[np.ndarray]` 를 반환하므로 첫 항목을 24kHz
        wav 로 직렬화한다. ref embedding 캐시는 OmniVoice 내부 API 가 노출되면
        후속 최적화 영역 — 현재는 매 호출 ref_audio/ref_text 그대로 전달.
        """
        import soundfile as sf  # type: ignore[import-not-found]

        target.parent.mkdir(parents=True, exist_ok=True)
        audio = self._model.generate(  # type: ignore[union-attr]
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
        sf.write(str(target), audio[0], _SAMPLE_RATE)
        return target


def _write_silent_wav(target: Path) -> None:
    """더미 모드용 무음 wav (1초, 24kHz, float32 zeros) 를 쓴다."""
    import numpy as np
    import soundfile as sf

    target.parent.mkdir(parents=True, exist_ok=True)
    samples = int(_SAMPLE_RATE * _DUMMY_DURATION_SECONDS)
    sf.write(str(target), np.zeros(samples, dtype="float32"), _SAMPLE_RATE)
