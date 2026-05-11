"""GemmaLLM — `google/gemma-4-E4B-it` 멀티모달 로더 + 추론.

- 부팅 시 AWQ INT4 가중치 우선 로드. Blackwell(sm_120) 등에서 AWQ 커널이
  미지원이면 BF16 으로 폴백한다.
- 음성 입력은 멀티모달 audio-in 으로 Gemma 가 직접 이해해 한국어 응답을 만든다.
- 토큰은 transformers `TextIteratorStreamer` 를 별도 스레드에 띄워 async 큐로
  yield (단일 워커 + asyncio 단일 루프 전제).
- `GPU_ENABLED=false` 모드는 모든 호출을 더미 텍스트/토큰 반환으로 단락 처리.

이슈 #6 범위는 로더·전사·스트림 API 까지. 시스템 프롬프트 조립과 conversation
파이프라인은 #10 에서 본 클래스를 호출한다.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal

logger = logging.getLogger("yeoun")

# model card 권장 샘플링 — 모든 추론 호출 공통.
_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}

# 명세서 결정 — KV 캐시 절약 위해 컨텍스트 8K 로 제한 (모델 자체는 256K 까지 지원).
_MAX_CONTEXT_TOKENS = 8192

_DUMMY_RESPONSE_TOKENS = (
    "[",
    "더미",
    " 응답",
    "]",
    " GPU_ENABLED=false",
    " 모드",
    " — ",
    "이",
    " 메시지는",
    " 실제",
    " 모델",
    " 출력이",
    " 아닙니다",
    ".",
)

LlmStatus = Literal["not_loaded", "loading", "loaded", "error"]


class GemmaLLM:
    """Gemma 4 E4B-it 멀티모달 LLM 래퍼."""

    def __init__(
        self,
        *,
        awq_path: str | None,
        bf16_path: str | None,
        gpu_enabled: bool,
    ) -> None:
        self._awq_path = awq_path
        self._bf16_path = bf16_path
        self._gpu_enabled = gpu_enabled
        self._status: LlmStatus = "not_loaded"
        self._model = None
        self._processor = None
        self._loaded_variant: Literal["awq", "bf16"] | None = None

    # --- 라이프사이클 -------------------------------------------------------
    async def load(self) -> None:
        """가중치를 GPU 에 적재. 더미 모드면 즉시 'loaded' 로 표시."""
        if not self._gpu_enabled:
            logger.info("GPU_ENABLED=false → GemmaLLM 더미 모드")
            self._status = "loaded"
            return

        # 실 GPU 로드 — AWQ 우선, 실패 시 BF16 폴백.
        # 실제 transformers 호출은 다음 커밋에서 채운다.
        raise NotImplementedError(
            "GPU 실 로드 분기는 다음 커밋(AWQ + BF16 폴백) 에서 구현."
        )

    async def unload(self) -> None:
        """모델 자원을 해제. 더미 모드면 no-op."""
        if not self._gpu_enabled:
            self._status = "not_loaded"
            return
        # GPU 해제는 후속 커밋에서.
        self._status = "not_loaded"
        self._model = None
        self._processor = None
        self._loaded_variant = None

    # --- 상태 --------------------------------------------------------------
    @property
    def status(self) -> LlmStatus:
        return self._status

    @property
    def variant(self) -> Literal["awq", "bf16"] | None:
        return self._loaded_variant

    # --- 추론 (더미) -------------------------------------------------------
    async def transcribe(self, audio_path: str | Path) -> str:
        """짧은 오디오 → 한국어 전사. 더미 모드는 고정 문자열."""
        if not self._gpu_enabled:
            return f"[더미 전사] {Path(audio_path).name}"
        raise NotImplementedError("실 전사 분기는 다음 커밋에서 구현.")

    async def stream_response(
        self,
        *,
        system_prompt: str,
        history: list[dict],
        user_audio_path: str | Path,
    ) -> AsyncIterator[str]:
        """audio-in 응답을 토큰 단위로 yield. 더미 모드는 고정 토큰 시퀀스."""
        if not self._gpu_enabled:
            for token in _DUMMY_RESPONSE_TOKENS:
                # 실모드 흉내내기 위해 한 토큰당 짧은 await.
                await asyncio.sleep(0)
                yield token
            return
        raise NotImplementedError("실 스트림 분기는 다음 커밋에서 구현.")
