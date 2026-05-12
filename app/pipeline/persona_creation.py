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
import shutil
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


def _pick_photo_file(persona_dir: str, persona_id: UUID) -> Path:
    """`/var/persona/{id}/photo/*` 중 mtime 가장 새 파일을 ref 사진으로 선택.

    Spring 이 다중 업로드한 경우 사용자가 마지막에 올린 사진을 페르소나
    대표 사진으로 사용한다는 가정. 디렉토리가 없거나 비어 있으면 `FileNotFoundError`.
    """
    photo_dir = _persona_root(persona_dir, persona_id) / "photo"
    if not photo_dir.exists():
        raise FileNotFoundError(f"photo 디렉토리 없음: {photo_dir}")
    candidates = [p for p in photo_dir.iterdir() if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"photo 디렉토리 비어 있음: {photo_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


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


async def _transcribe_voice(
    voice_path: Path,
    ref_text_path: Path,
    registry,
) -> str:
    """Gemma audio-in 으로 voice 파일을 전사하고 `ref_text.txt` 에 저장.

    멱등성: ref_text_path 가 이미 존재하고 비어 있지 않으면 그 내용을 그대로
    반환해 GPU 호출을 스킵한다. GPU 진입은 `gpu_semaphore` 로 직렬화.
    """
    if ref_text_path.exists() and ref_text_path.stat().st_size > 0:
        logger.info("ref_text 멱등 스킵: %s", ref_text_path)
        return ref_text_path.read_text(encoding="utf-8")

    async with registry.gpu_semaphore:
        text = await registry.llm.transcribe(voice_path)

    # atomic write — 부분 쓰기 후 죽으면 멱등 스킵이 깨져 GPU 전사를 재호출하게
    # 되므로 임시 파일에 먼저 쓰고 rename 으로 마감한다.
    ref_text_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = ref_text_path.with_suffix(ref_text_path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(ref_text_path)
    logger.info("ref_text 저장: %s (%d 자)", ref_text_path, len(text))
    return text


async def _prepare_voice_ref(voice_path: Path, ref_audio_path: Path) -> None:
    """OmniVoice ref 자원으로 voice 파일을 voice_ref/ref_audio.<ext> 로 복사.

    명세서는 "embedding 추출" 을 명시하지만 OmniVoice 가 embedding 추출 API 를
    공개하지 않으므로 본 1차 구현은 매 합성 시 ref_audio + ref_text 를 그대로
    재사용하는 방식. 본 helper 는 그 ref_audio 를 voice/ 의 후속 업로드와 분리된
    안정적 경로(`voice_ref/`)에 복사한다.

    멱등성: ref_audio_path 가 이미 존재하면 스킵.
    """
    if ref_audio_path.exists():
        logger.info("ref_audio 멱등 스킵: %s", ref_audio_path)
        return
    ref_audio_path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.copy2, voice_path, ref_audio_path)
    logger.info("ref_audio 복사: %s -> %s", voice_path, ref_audio_path)


async def _render_idle_clips(
    photo_path: Path,
    idle_dir: Path,
    registry,
    *,
    count: int = 2,
    duration_seconds: float = 7.0,
) -> list[Path]:
    """idle 클립 N 개를 `idle_dir/{0..N-1}.mp4` 로 렌더.

    클라이언트가 0↔1 번갈아 재생해 자연스러운 변주를 얻도록 기본 2 개. 각
    인덱스별로 멱등 — 이미 비어 있지 않은 파일이 있으면 GPU 호출 스킵. GPU
    진입은 `ditto_semaphore` 로 직렬화.
    """
    idle_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Path] = []
    for i in range(count):
        target = idle_dir / f"{i}.mp4"
        if target.exists() and target.stat().st_size > 0:
            logger.info("idle 클립 멱등 스킵: %s", target)
            rendered.append(target)
            continue
        async with registry.ditto_semaphore:
            await registry.ditto.render_idle(
                image_path=str(photo_path),
                output_path=str(target),
                duration_seconds=duration_seconds,
            )
        logger.info("idle 클립 렌더 완료: %s", target)
        rendered.append(target)
    return rendered


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

    멱등성: 각 단계 helper 가 이미 존재하는 산출물을 스킵. 워커 재시작 후 재호출
    시에도 안전.
    """
    # 순환 import 회피용 지연 import.
    from ..db import repository

    root = _persona_root(persona_dir, persona_id)
    voice_ref_dir = root / "voice_ref"
    idle_dir = root / "idle"
    ref_text_path = voice_ref_dir / "ref_text.txt"

    try:
        await store.set_step(persona_id, ProcessingStep.PENDING)
        await repository.update_persona_status(persona_id, "processing")

        # 1. voice 파일 선택
        await store.set_step(persona_id, ProcessingStep.SELECTING_VOICE)
        voice_path = _pick_voice_file(persona_dir, persona_id)

        # 2. Gemma audio-in 전사
        await store.set_step(persona_id, ProcessingStep.TRANSCRIBING)
        await _transcribe_voice(voice_path, ref_text_path, registry)

        # 3. ref_audio 보관 (OmniVoice embedding API 미공개로 ref_audio 복사)
        await store.set_step(persona_id, ProcessingStep.EXTRACTING_REF)
        ref_audio_path = voice_ref_dir / f"ref_audio{voice_path.suffix}"
        await _prepare_voice_ref(voice_path, ref_audio_path)

        # 4. Ditto idle 클립 렌더
        await store.set_step(persona_id, ProcessingStep.RENDERING_IDLE)
        photo_path = _pick_photo_file(persona_dir, persona_id)
        await _render_idle_clips(photo_path, idle_dir, registry)

        # 5. ready
        await store.set_step(persona_id, ProcessingStep.READY)
        await repository.update_persona_status(persona_id, "ready")
        logger.info("페르소나 생성 완료: %s", persona_id)
    except Exception as exc:  # noqa: BLE001 — 모든 실패를 failed 로 환원
        logger.exception("페르소나 생성 실패: %s", persona_id)
        # 클라이언트/운영자에 한국어 요약 노출, 500자 컷.
        reason = f"{type(exc).__name__}: {exc}"[:500]
        await store.set_step(persona_id, ProcessingStep.FAILED, error_reason=reason)
        try:
            await repository.update_persona_status(persona_id, "failed")
        except Exception:  # noqa: BLE001 — 폴백 실패도 로그만
            logger.exception("update_persona_status('failed') 폴백 실패")
