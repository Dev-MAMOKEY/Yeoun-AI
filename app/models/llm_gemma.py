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
        """가중치를 GPU 에 적재.

        AWQ INT4 우선 시도 → 실패 시(Blackwell 등에서 AWQ 커널 미지원) BF16 폴백.
        둘 다 실패하면 `error` 상태로 예외 raise. 더미 모드는 즉시 `loaded`.
        """
        if not self._gpu_enabled:
            logger.info("GPU_ENABLED=false → GemmaLLM 더미 모드")
            self._status = "loaded"
            return

        self._status = "loading"

        if self._awq_path:
            try:
                await asyncio.to_thread(self._load_variant, "awq", self._awq_path)
                self._loaded_variant = "awq"
                self._status = "loaded"
                return
            except Exception as exc:  # noqa: BLE001 — 어떤 실패든 BF16 폴백 시도
                logger.warning(
                    "AWQ 로드 실패 (Blackwell 커널 미지원 가능) — BF16 폴백 시도: %s",
                    exc,
                )

        if self._bf16_path:
            try:
                await asyncio.to_thread(self._load_variant, "bf16", self._bf16_path)
                self._loaded_variant = "bf16"
                self._status = "loaded"
                return
            except Exception:
                logger.exception("BF16 로드도 실패")
                self._status = "error"
                raise

        self._status = "error"
        raise RuntimeError(
            "LLM 가중치 경로가 모두 비어 있습니다. LLM_AWQ_PATH 또는 LLM_BF16_PATH 중 "
            "최소 하나를 설정하거나 GPU_ENABLED=false 로 더미 모드를 사용하십시오."
        )

    def _load_variant(self, variant: Literal["awq", "bf16"], path: str) -> None:
        """단일 변형 가중치를 동기 로드. `asyncio.to_thread` 안에서 호출된다."""
        # transformers >= 4.50 의 멀티모달 클래스. 일부 버전엔 이 이름이 없어
        # AutoModelForCausalLM 으로 폴백 — Gemma 4 model card 의 멀티모달 예시는
        # AutoModelForMultimodalLM 을 쓰지만 가용성이 환경마다 다르다.
        try:
            from transformers import AutoModelForMultimodalLM as _AutoModel  # type: ignore[attr-defined]
        except ImportError:
            from transformers import AutoModelForCausalLM as _AutoModel  # type: ignore[assignment]
        from transformers import AutoProcessor

        logger.info("Gemma %s 로드 시작: %s", variant.upper(), path)
        self._processor = AutoProcessor.from_pretrained(path)

        load_kwargs: dict[str, object] = {"device_map": "auto"}
        if variant == "awq":
            # AWQ 양자화 가중치는 dtype 을 그대로 따른다 ("auto").
            load_kwargs["dtype"] = "auto"
        else:
            import torch

            load_kwargs["dtype"] = torch.bfloat16

        self._model = _AutoModel.from_pretrained(path, **load_kwargs)
        logger.info("Gemma %s 로드 완료", variant.upper())

    async def unload(self) -> None:
        """모델 자원을 해제. 더미 모드면 no-op."""
        if not self._gpu_enabled:
            self._status = "not_loaded"
            return
        self._model = None
        self._processor = None
        self._loaded_variant = None
        self._status = "not_loaded"
        # GPU 메모리 회수 — torch 가 import 가능할 때만.
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

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
