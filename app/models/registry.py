"""ModelRegistry — GPU 모델 자원 매니저.

단일 uvicorn 워커 전제로 LLM·TTS·Ditto 모델 인스턴스를 한 곳에서 보유한다.
- Gemma LLM: 부팅 시 로드, 상주.
- OmniVoice TTS: 부팅 시 로드, 상주 (작아서 상주 가능).
- Ditto Talking Head: 매 렌더 subprocess (온디맨드 로드/언로드).

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
from .talkinghead_ditto import DittoTalkingHead
from .tts_omnivoice import OmniVoiceTTS

logger = logging.getLogger("yeoun")


# GPU 추론 진입 직렬화 폭 — 단일 워커 + GPU 1장 전제로 한 번에 1건만 허용.
GPU_CONCURRENCY: int = 1


class ModelRegistry:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.llm: GemmaLLM | None = None
        self.tts: OmniVoiceTTS | None = None
        self.ditto: DittoTalkingHead | None = None
        self._gpu_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)
        # Ditto 는 매 호출 subprocess 라 Gemma/TTS 와 자원이 분리된다 — 별도 락 유지.
        # 통합하면 SSE 토큰 스트리밍(Gemma) 도중 Ditto 렌더가 차단되어 응답 지연.
        # 동시 점유 시 VRAM 합계는 LLM(BF16) ~8GB + TTS ~2GB + Ditto subprocess
        # ~2.2GB ≈ 12GB 로 24GB 안에서 안전 마진 확보. 추후 KV 캐시 증가나 동시 세션
        # 폭증으로 OOM 압력이 보이면 통합 또는 메모리 예산 재산정.
        self._ditto_semaphore = asyncio.Semaphore(GPU_CONCURRENCY)

    async def start(self) -> None:
        """상주 모델을 부팅 시점에 로드."""
        logger.info("ModelRegistry 시작 — Gemma LLM 로드 시도")
        self.llm = GemmaLLM(
            bf16_path=self._settings.llm_bf16_path,
            gpu_enabled=self._settings.gpu_enabled,
        )
        await self.llm.load()
        logger.info("Gemma LLM 상태=%s", self.llm.status)

        logger.info("ModelRegistry — OmniVoice TTS 로드 시도")
        self.tts = OmniVoiceTTS(
            model_path=self._settings.tts_model_path,
            gpu_enabled=self._settings.gpu_enabled,
        )
        await self.tts.load()
        logger.info("OmniVoice TTS 상태=%s", self.tts.status)

        logger.info("ModelRegistry — DittoTalkingHead 경로 검증")
        self.ditto = DittoTalkingHead(
            vendor_dir=self._settings.ditto_vendor_dir,
            data_root=self._settings.ditto_data_root,
            cfg_pkl=self._settings.ditto_cfg_pkl,
            gpu_enabled=self._settings.gpu_enabled,
        )
        await self.ditto.load()
        logger.info("DittoTalkingHead 상태=%s", self.ditto.status)

    async def stop(self) -> None:
        """모든 모델 자원 정리."""
        logger.info("ModelRegistry 종료")
        # Ditto·TTS 먼저 내려 GPU 메모리부터 회수.
        if self.ditto is not None:
            await self.ditto.unload()
            self.ditto = None
        if self.tts is not None:
            await self.tts.unload()
            self.tts = None
        if self.llm is not None:
            await self.llm.unload()
            self.llm = None

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        """GPU 추론 진입 직렬화용 세마포어 — 호출자가 `async with` 로 보호."""
        return self._gpu_semaphore

    @property
    def ditto_semaphore(self) -> asyncio.Semaphore:
        """Ditto 렌더 전용 세마포어 — subprocess 라 Gemma/TTS 와 자원 분리."""
        return self._ditto_semaphore

    @property
    def gpu_concurrency(self) -> int:
        """GPU 추론 동시 허용 개수 (테스트·모니터링용 공개 상수)."""
        return GPU_CONCURRENCY
