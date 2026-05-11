"""ModelRegistry — GPU 모델 자원 매니저.

단일 uvicorn 워커 전제로 LLM·TTS·Ditto 모델 인스턴스를 한 곳에서 보유한다.
- Gemma LLM: 부팅 시 로드, 상주 (이슈 #6).
- OmniVoice TTS: #7 에서 등록 (작아서 상주 가능).
- Ditto Talking Head: #8 에서 등록 (온디맨드 로드/언로드).

`gpu_semaphore` 로 동시 GPU 진입을 1건으로 직렬화한다 — LLM/TTS/Ditto 가 같은
GPU 를 공유해 추론 사이 메모리 충돌을 방지하기 위함. 추후 LLM 과 Ditto 처럼
서로 다른 자원을 동시 사용해도 무방한 경우 별도 세마포어 분리 가능.

`start()` / `stop()` 는 FastAPI lifespan 에서 호출한다.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import Settings
from .llm_gemma import GemmaLLM

logger = logging.getLogger("yeoun")


# GPU 추론 진입 직렬화 폭 — 단일 워커 + GPU 1장 전제로 한 번에 1건만 허용.
GPU_CONCURRENCY: int = 1


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.llm: GemmaLLM | None = None
        # TTS / Ditto 는 이슈 #7 / #8 에서 추가.
        self._gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

    async def start(self) -> None:
        """상주 모델을 부팅 시점에 로드."""
        logger.info("ModelRegistry 시작 — Gemma LLM 로드 시도")
        self.llm = GemmaLLM(
            awq_path=self._settings.llm_awq_path,
            bf16_path=self._settings.llm_bf16_path,
            gpu_enabled=self._settings.gpu_enabled,
        )
        await self.llm.load()
        logger.info("Gemma LLM 상태=%s, variant=%s", self.llm.status, self.llm.variant)

    async def stop(self) -> None:
        """모든 모델 자원 정리."""
        logger.info("ModelRegistry 종료")
        if self.llm is not None:
            await self.llm.unload()
            self.llm = None

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        """GPU 추론 진입 직렬화용 세마포어 — 호출자가 `async with` 로 보호."""
        return self._gpu_semaphore

    @property
    def gpu_concurrency(self) -> int:
        """GPU 추론 동시 허용 개수 (테스트·모니터링용 공개 상수)."""
        return GPU_CONCURRENCY
