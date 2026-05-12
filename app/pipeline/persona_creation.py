"""페르소나 생성 백그라운드 파이프라인 (이슈 #9, 명세서 흐름 A).

업로드 완료된 페르소나(`personas.status='created'`) 에 대해:
1. `/var/persona/{id}/voice/*` 중 가장 새 파일을 선택 → Gemma audio-in 전사 →
   `/var/persona/{id}/voice_ref/ref_text.txt`
2. OmniVoice ref embedding 추출(가능 시) → `/var/persona/{id}/voice_ref/embedding.pt`
3. Ditto idle 클립 2개 렌더 → `/var/persona/{id}/idle/{0,1}.mp4`
4. `personas.status='ready'`. 실패 시 `'failed'` + `error_reason`.

GPU 자원: 1·2 단계는 `ModelRegistry.gpu_semaphore` (LLM/TTS 공유), 3 단계는
`ditto_semaphore` (Ditto 전용). 진행률은 `PersonaProcessingStore` 가 워커
메모리에 보관 — 워커 재시작 시 사라지지만 `process_persona` 가 멱등이라 재호출
시 이미 완료된 단계는 스킵.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID

from ..schemas.persona import ProcessingStep

logger = logging.getLogger("yeoun")


@dataclass
class _ProcessingState:
    """단일 페르소나의 진행률 스냅샷.

    `PersonaProcessingStore` 가 dict value 로 보관. step/error_reason 은 단계
    함수가 갱신, updated_at 은 자동 갱신.
    """

    persona_id: UUID
    step: ProcessingStep = ProcessingStep.PENDING
    error_reason: str | None = None
    updated_at: float = field(default_factory=time.time)


class PersonaProcessingStore:
    """인메모리 진행률 저장소 — SessionStore 와 동일한 워커 로컬 dict 패턴.

    `personas.status` 는 DB(또는 mock repository) 가 보유, `step` 은 본 스토어가
    보유. 둘은 의도적으로 분리 — DB 컬럼 미확정 상태에서 step 추가를 강요하지
    않으면서 운영자/Spring 이 폴링으로 세부 단계를 확인할 수 있게 한다.
    """

    def __init__(self) -> None:
        self._states: dict[UUID, _ProcessingState] = {}
        self._lock = asyncio.Lock()

    async def set_step(
        self,
        persona_id: UUID,
        step: ProcessingStep,
        *,
        error_reason: str | None = None,
    ) -> None:
        """진행률 단계 갱신. 신규 페르소나면 자동 생성."""
        async with self._lock:
            state = self._states.get(persona_id)
            if state is None:
                state = _ProcessingState(persona_id=persona_id)
                self._states[persona_id] = state
            state.step = step
            state.error_reason = error_reason
            state.updated_at = time.time()

    def get(self, persona_id: UUID) -> _ProcessingState | None:
        """스냅샷 조회. 없으면 None."""
        return self._states.get(persona_id)

    def count(self) -> int:
        """현재 추적 중인 페르소나 수 (테스트·모니터링용)."""
        return len(self._states)


def _persona_root(persona_dir: str, persona_id: UUID) -> Path:
    """페르소나별 자원 루트 — `{PERSONA_DIR}/{persona_id}/`."""
    return Path(persona_dir) / str(persona_id)


def _pick_voice_file(persona_dir: str, persona_id: UUID) -> Path:
    """`/var/persona/{id}/voice/*` 중 mtime 가장 새 파일을 선택.

    Spring 이 다중 업로드한 경우 사용자가 마지막에 올린 파일을 ref 음성으로
    사용한다는 가정. 디렉토리가 없거나 비어 있으면 `FileNotFoundError`.
    """
    voice_dir = _persona_root(persona_dir, persona_id) / "voice"
    if not voice_dir.exists():
        raise FileNotFoundError(f"voice 디렉토리 없음: {voice_dir}")
    candidates = [p for p in voice_dir.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"voice 디렉토리 비어 있음: {voice_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def process_persona(
    persona_id: UUID,
    *,
    registry,  # ModelRegistry — 순환 import 회피 위해 타입 힌트 생략
    persona_dir: str,
    store: PersonaProcessingStore,
) -> None:
    """페르소나 생성 백그라운드 파이프라인의 단일 진입점.

    호출자는 FastAPI `BackgroundTasks` 로 즉시 202 응답 후 비동기 실행한다.
    예외는 본 함수가 잡아 status='failed' 로 마감 — 호출자는 별도 처리 불필요.

    멱등성: 이미 산출물이 존재하는 단계는 스킵. 워커 재시작 후 재호출 시 유용.
    """
    # 이후 단계(전사·추출·렌더·status 갱신) 는 후속 커밋에서 점진적으로 추가.
    await store.set_step(persona_id, ProcessingStep.PENDING)
