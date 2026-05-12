"""페르소나 생성·조회 라우터 (이슈 #9, 명세서 흐름 A).

엔드포인트:
- POST `/internal/personas/{id}/process` → 202 + BackgroundTask 로
  `process_persona` 비동기 실행
- GET `/internal/personas/{id}/status` (#9 후속 커밋)
- GET `/internal/personas/{id}/idle-clips` (#9 후속 커밋)
"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi import status as http_status

from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..db import repository
from ..models.registry import ModelRegistry
from ..pipeline.persona_creation import PersonaProcessingStore, process_persona
from ..schemas.common import Envelope, ok
from ..schemas.persona import IdleClipMeta, IdleClipsData, PersonaStatusData

router = APIRouter(prefix="/internal/personas", tags=["personas"])


@router.post(
    "/{persona_id}/process",
    response_model=Envelope[None],
    status_code=http_status.HTTP_202_ACCEPTED,
    summary="페르소나 생성 백그라운드 작업 시작",
    description=(
        "Spring 이 사진·음성 업로드를 끝낸 직후 호출. 202 를 즉시 반환하고 "
        "FastAPI BackgroundTask 가 비동기로 voice 전사·ref 보관·idle 렌더를 수행한다. "
        "진행률은 GET `/internal/personas/{id}/status`, 산출물은 GET `/internal/personas/{id}/idle-clips` 으로 폴링."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        202: {"description": "백그라운드 작업이 큐에 등록되었음."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        503: {"description": "ModelRegistry 또는 PersonaProcessingStore 가 부팅 전 (`SERVICE_UNAVAILABLE`)."},
    },
)
async def start_processing(
    persona_id: UUID,
    background_tasks: BackgroundTasks,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Envelope[None]:
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    store: PersonaProcessingStore | None = getattr(request.app.state, "persona_store", None)
    if registry is None or store is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "ModelRegistry 또는 PersonaProcessingStore 가 초기화되지 않았습니다.",
            },
        )
    background_tasks.add_task(
        process_persona,
        persona_id,
        registry=registry,
        persona_dir=settings.persona_dir,
        store=store,
    )
    return ok(None)


@router.get(
    "/{persona_id}/status",
    response_model=Envelope[PersonaStatusData],
    summary="페르소나 생성 진행 상태 조회",
    description=(
        "Spring 이 대기 화면에서 5초 주기로 폴링하는 엔드포인트. "
        "`status` 는 DB(또는 USE_DB_MOCK 인메모리 repository) 에서, "
        "`step` 은 PersonaProcessingStore 에서 가져온다. 워커 메모리라 재시작 시 step 사라짐 — "
        "그 경우 `step=null` 이지만 `status` 는 그대로 유지."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "상태 조회 성공."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
    },
)
async def get_status(
    persona_id: UUID,
    request: Request,
) -> Envelope[PersonaStatusData]:
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )
    store: PersonaProcessingStore | None = getattr(request.app.state, "persona_store", None)
    state = store.get(persona_id) if store is not None else None
    return ok(
        PersonaStatusData(
            persona_id=persona_id,
            status=record.status,
            step=state.step if state is not None else None,
            error_reason=state.error_reason if state is not None else None,
        )
    )


@router.get(
    "/{persona_id}/idle-clips",
    response_model=Envelope[IdleClipsData],
    summary="페르소나 idle 클립 메타 조회",
    description=(
        "`status='ready'` 이후 호출. `/var/persona/{id}/idle/{n}.mp4` 의 메타(인덱스·경로·크기)를 반환. "
        "실제 mp4 스트리밍은 이슈 #11 의 미디어 라우터가 담당."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "조회 성공 (status 가 ready 가 아니면 클립 목록이 비어 있을 수 있음)."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
    },
)
async def get_idle_clips(
    persona_id: UUID,
    settings: Settings = Depends(get_settings),
) -> Envelope[IdleClipsData]:
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )

    idle_dir = Path(settings.persona_dir) / str(persona_id) / "idle"
    clips: list[IdleClipMeta] = []
    if idle_dir.exists():
        for p in sorted(idle_dir.glob("*.mp4")):
            if not p.is_file():
                continue
            try:
                # 0.mp4, 1.mp4 만 유효 인덱스로 채택.
                idx = int(p.stem)
            except ValueError:
                continue
            clips.append(
                IdleClipMeta(index=idx, path=str(p), size_bytes=p.stat().st_size)
            )

    return ok(IdleClipsData(persona_id=persona_id, clips=clips))
