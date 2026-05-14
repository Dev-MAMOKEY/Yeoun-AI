"""페르소나 생성·조회·삭제 라우터.

엔드포인트는 파이프라인 흐름 순서로 정의:
- POST `/internal/personas/{id}/photo`        사진 업로드
- POST `/internal/personas/{id}/voice`        음성 업로드
- POST `/internal/personas/{id}/process`      202 + BackgroundTask
- GET  `/internal/personas/{id}/status`       진행률 폴링
- GET  `/internal/personas/{id}/idle-clips`   산출물 메타
- DELETE `/internal/personas/{id}`            DB + FS 영구 삭제
"""

import asyncio
import logging
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile
from fastapi import status as http_status

from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..db import repository
from ..models.registry import ModelRegistry
from ..pipeline.persona_creation import PersonaProcessingStore, process_persona
from ..schemas.common import Envelope, ok
from ..schemas.persona import (
    IdleClipMeta,
    IdleClipsData,
    PersonaStatusData,
    ProcessingStep,
    UploadResult,
)
from ..storage.filesystem import safe_resolve, safe_rmtree

# 허용 content-type — 흔한 모바일/웹 업로드 형식만. inference.py / Gemma audio-in
# 이 ffmpeg 로 자동 디코딩하므로 codec 별로 분기하지 않는다.
_PHOTO_CONTENT_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_VOICE_CONTENT_TYPES = frozenset(
    {
        "audio/wav", "audio/x-wav", "audio/wave",
        "audio/mpeg", "audio/mp3",
        "audio/mp4", "audio/m4a", "audio/x-m4a",
        "audio/webm", "audio/ogg",
    }
)

# 매직 바이트 prefix — content-type 위조 1차 방어. 첫 16바이트 검사.
# ffmpeg 가 컨테이너 안에서 다시 검증하므로 본 가드는 명백히 잘못된 파일만 거른다.
_PHOTO_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"\xff\xd8\xff",         # JPEG
    b"\x89PNG\r\n\x1a\n",    # PNG
    b"RIFF",                  # WebP (RIFF....WEBP)
)
_VOICE_MAGIC_PREFIXES: tuple[bytes, ...] = (
    b"RIFF",                  # WAV/WebM-RIFF
    b"ID3",                   # MP3 with ID3
    b"\xff\xfb", b"\xff\xf3", b"\xff\xf2",  # MP3 frame sync
    b"\x1aE\xdf\xa3",        # WebM/Matroska
    b"OggS",                  # OGG
    b"ftyp",                  # MP4 (ftyp box, offset 4 — 호출자가 [4:8] 비교)
)

logger = logging.getLogger("yeoun")

router = APIRouter(prefix="/internal/personas", tags=["personas"])


async def _ensure_writable_persona(persona_id: UUID) -> None:
    """업로드 라우트 공용 가드 — 페르소나 존재 + processing 차단."""
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )
    if record.status == "PROCESSING":
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "code": "CONFLICT",
                "message": "처리 중 페르소나에는 업로드할 수 없습니다 (status='PROCESSING').",
            },
        )


def _looks_like(prefix_set: tuple[bytes, ...], head: bytes) -> bool:
    """파일 첫 N 바이트가 매직 prefix 중 하나와 일치하는지. MP4 `ftyp` 는 offset 4 비교."""
    for prefix in prefix_set:
        if prefix == b"ftyp" and len(head) >= 8 and head[4:8] == b"ftyp":
            return True
        if head.startswith(prefix):
            return True
    return False


async def _save_upload(
    *,
    persona_id: UUID,
    file: UploadFile,
    subdir: str,
    allowed_content_types: frozenset[str],
    magic_prefixes: tuple[bytes, ...],
    settings: Settings,
) -> tuple[Path, int]:
    """업로드 파일을 검증 후 PERSONA_DIR 하위에 atomic 저장하고 경로·크기 반환.

    검증 순서: content-type → 매직 바이트 → 크기 cap(스트리밍) → 파일명 sanitize →
    `safe_resolve` 가드. `.tmp.{uuid}` 임시 파일로 먼저 쓰고 `replace` 로 마감해
    중단·동시 업로드 race 모두 차단. 같은 파일명 재업로드는 **덮어쓰기**.
    """
    try:
        if file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={
                    "code": "UNSUPPORTED_MEDIA_TYPE",
                    "message": f"지원하지 않는 content-type: {file.content_type!r}",
                },
            )

        # 스트리밍 read — 한도 초과 즉시 413. 첫 chunk 의 앞부분으로 매직 바이트 확인.
        chunk_size = 1024 * 1024
        buffer = bytearray()
        magic_checked = False
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            buffer.extend(chunk)
            if not magic_checked:
                head = bytes(buffer[:16])
                if len(head) >= 4 and not _looks_like(magic_prefixes, head):
                    raise HTTPException(
                        status_code=http_status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                        detail={
                            "code": "UNSUPPORTED_MEDIA_TYPE",
                            "message": "파일 매직 바이트가 허용 형식과 일치하지 않습니다.",
                        },
                    )
                if len(head) >= 4:
                    magic_checked = True
            if len(buffer) > settings.max_upload_bytes:
                raise HTTPException(
                    status_code=http_status.HTTP_413_CONTENT_TOO_LARGE,
                    detail={
                        "code": "PAYLOAD_TOO_LARGE",
                        "message": f"업로드 크기 한도 {settings.max_upload_bytes} 바이트를 초과했습니다.",
                    },
                )
    finally:
        # 415·413 raise 경로 포함해 항상 UploadFile 의 SpooledTemporaryFile 닫기.
        await file.close()

    # 파일명 sanitize — Path 한 단계로 path-traversal 차단, 위험 케이스(`.`·`..`·null-byte·빈
    # 이름)는 uuid 로 대체.
    raw_name = Path(file.filename or "").name
    if not raw_name or raw_name in (".", "..") or "\x00" in raw_name or "/" in raw_name:
        raw_name = ""
    filename = raw_name or f"upload-{uuid4().hex}"

    try:
        target = safe_resolve(settings.persona_dir, str(persona_id), subdir, filename)
    except PermissionError:
        raise HTTPException(
            status_code=http_status.HTTP_400_BAD_REQUEST,
            detail={"code": "BAD_REQUEST", "message": "허용되지 않는 파일명입니다."},
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    # uuid 섞은 임시 파일명 — 동시 업로드 시 `.tmp` 충돌 방지.
    tmp = Path(f"{target}.tmp.{uuid4().hex}")
    try:
        await asyncio.to_thread(tmp.write_bytes, bytes(buffer))
        tmp.replace(target)
    except Exception:
        # write·replace 사이 예외 — 잔류 .tmp 정리.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    logger.info("업로드 저장: persona=%s, path=%s, size=%d", persona_id, target, len(buffer))
    return target, len(buffer)


@router.post(
    "/{persona_id}/photo",
    response_model=Envelope[UploadResult],
    summary="페르소나 사진 업로드",
    description=(
        "사용자가 올린 사진을 `/var/persona/{id}/photo/{filename}` 으로 저장. "
        "허용 content-type: `image/jpeg`, `image/png`, `image/webp`. 한 파일 최대 `max_upload_bytes` "
        "(기본 50MB). 처리 중 페르소나에는 업로드 불가(409)."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "업로드 성공."},
        400: {"description": "허용되지 않는 파일명 (`BAD_REQUEST`)."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
        409: {"description": "처리 중 페르소나 (`CONFLICT`)."},
        413: {"description": "업로드 크기 한도 초과 (`PAYLOAD_TOO_LARGE`)."},
        415: {"description": "지원하지 않는 content-type (`UNSUPPORTED_MEDIA_TYPE`)."},
    },
)
async def upload_photo(
    persona_id: UUID,
    file: UploadFile = File(..., description="사진 파일 (jpeg/png/webp)."),
    settings: Settings = Depends(get_settings),
) -> Envelope[UploadResult]:
    await _ensure_writable_persona(persona_id)
    target, size = await _save_upload(
        persona_id=persona_id,
        file=file,
        subdir="photo",
        allowed_content_types=_PHOTO_CONTENT_TYPES,
        magic_prefixes=_PHOTO_MAGIC_PREFIXES,
        settings=settings,
    )
    # 자원 메타 등록 — Spring 이 DB 만 보고도 페르소나에 어떤 사진이 매핑됐는지 알 수 있게.
    await repository.insert_persona_photo_asset(
        photo_asset_id=uuid4(),
        persona_id=persona_id,
        filesystem_path=str(target),
    )
    return ok(UploadResult(path=f"{persona_id}/photo/{target.name}", size_bytes=size))


@router.post(
    "/{persona_id}/voice",
    response_model=Envelope[UploadResult],
    summary="페르소나 음성 업로드",
    description=(
        "사용자가 올린 음성(reference voice) 을 `/var/persona/{id}/voice/{filename}` 으로 저장. "
        "허용 content-type: `audio/wav`, `audio/mp3`, `audio/m4a`, `audio/webm`, `audio/ogg` 등. "
        "한 파일 최대 `max_upload_bytes` (기본 50MB). 처리 중 페르소나에는 업로드 불가(409)."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "업로드 성공."},
        400: {"description": "허용되지 않는 파일명 (`BAD_REQUEST`)."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
        409: {"description": "처리 중 페르소나 (`CONFLICT`)."},
        413: {"description": "업로드 크기 한도 초과 (`PAYLOAD_TOO_LARGE`)."},
        415: {"description": "지원하지 않는 content-type (`UNSUPPORTED_MEDIA_TYPE`)."},
    },
)
async def upload_voice(
    persona_id: UUID,
    file: UploadFile = File(..., description="음성 파일 (wav/mp3/m4a/webm/ogg)."),
    settings: Settings = Depends(get_settings),
) -> Envelope[UploadResult]:
    await _ensure_writable_persona(persona_id)
    original_name = file.filename or ""
    target, size = await _save_upload(
        persona_id=persona_id,
        file=file,
        subdir="voice",
        allowed_content_types=_VOICE_CONTENT_TYPES,
        magic_prefixes=_VOICE_MAGIC_PREFIXES,
        settings=settings,
    )
    await repository.insert_persona_voice_asset(
        voice_asset_id=uuid4(),
        persona_id=persona_id,
        original_name=original_name or target.name,
        filesystem_path=str(target),
    )
    return ok(UploadResult(path=f"{persona_id}/voice/{target.name}", size_bytes=size))


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
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
        409: {"description": "이미 처리 중이거나 완료된 페르소나 (`CONFLICT`)."},
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

    # 페르소나 존재 검증 — 없으면 404. 백그라운드로 넘긴 뒤 mock 이 silently
    # return 되는 일을 막아 Spring 이 즉시 오류를 인지하게 한다.
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )

    # 중복 호출 보호 — 진행 중 또는 ready 면 409. 같은 persona 두 작업 동시 실행
    # 시 디렉토리 race 와 status 역행(`ready→processing→ready`) 차단.
    state = store.get(persona_id)
    in_progress = state is not None and state.step not in (
        ProcessingStep.READY,
        ProcessingStep.FAILED,
    )
    if in_progress or record.status in ("PROCESSING", "READY"):
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "code": "CONFLICT",
                "message": f"이미 처리 중이거나 완료된 페르소나입니다 (status={record.status}).",
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
        "`status='READY'` 이후 호출. `/var/persona/{id}/idle/{n}.mp4` 의 메타(인덱스·경로·크기)를 반환. "
        "실제 mp4 스트리밍은 미디어 라우터(`/internal/personas/{id}/idle-clips/{idx}`) 가 담당."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "조회 성공 (status 가 READY 가 아니면 클립 목록이 비어 있을 수 있음)."},
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
            # 내부 절대경로 노출을 피하기 위해 PERSONA_DIR 기준 상대 경로로 변환.
            relative = f"{persona_id}/idle/{p.name}"
            clips.append(
                IdleClipMeta(index=idx, path=relative, size_bytes=p.stat().st_size)
            )

    return ok(IdleClipsData(persona_id=persona_id, clips=clips))


@router.delete(
    "/{persona_id}",
    response_model=Envelope[None],
    summary="페르소나 영구 삭제",
    description=(
        "DB 레코드 + `/var/persona/{id}/` 영구 파일을 함께 물리 삭제한다. "
        "FS 정리를 먼저 시도해 실패하면 DB 변경 없이 500 환원 — 고아 행 방지. "
        "삭제 순서를 바꿀 경우 DB rollback 이 어려운 mock repository 환경에서 일관성을 잃을 위험이 있어 본 순서를 고수."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "DB+FS 모두 삭제 완료."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
        500: {"description": "FS 정리 실패 — DB 변경 없음 (`DELETE_FAILED`)."},
    },
)
async def delete_persona(
    persona_id: UUID,
    settings: Settings = Depends(get_settings),
) -> Envelope[None]:
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )

    # 진행 중인 페르소나는 삭제 차단 — process_persona 가 voice_ref/idle 생성 중
    # rmtree 진입 시 ENOENT 후 재생성으로 FS/DB 불일치 발생. 운영자가 process
    # 완료(또는 failed) 까지 기다리거나 강제 cancel 인터페이스(후속) 호출 후 재시도.
    if record.status == "PROCESSING":
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "code": "CONFLICT",
                "message": "처리 중 페르소나는 삭제할 수 없습니다 (status='PROCESSING').",
            },
        )

    # 1) FS 정리 먼저 — 실패 시 DB 변경 안 함으로 고아 행 방지.
    try:
        deleted = await safe_rmtree(settings.persona_dir, str(persona_id))
        logger.info("페르소나 FS 정리: persona=%s, deleted=%s", persona_id, deleted)
    except PermissionError:
        # safe_resolve 가드가 raise — 내부 절대 경로가 메시지에 포함되어 응답으로 새지 않도록 sanitize.
        logger.exception("페르소나 FS 경로 가드 차단: persona=%s", persona_id)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DELETE_FAILED", "message": "내부 경로 검증 실패."},
        )
    except OSError as exc:
        logger.exception("FS 삭제 실패, DB 변경 보류: persona=%s", persona_id)
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "DELETE_FAILED", "message": f"파일 정리 실패: {type(exc).__name__}"},
        )

    # 2) DB 삭제 — mock 은 dict pop, 실 DB 는 CASCADE 트랜잭션.
    #    FS 는 이미 정리됐으므로 여기서 예외가 나오면 역고아 상태 (FS 없음 + DB 있음).
    #    운영자가 즉시 인지하도록 한국어 로그 + 500 응답으로 환원.
    try:
        await repository.delete_persona_tx(persona_id)
    except Exception as exc:
        logger.exception(
            "DB DELETE 실패 — FS 는 이미 정리됨, 수동 DB 정리 필요: persona_id=%s",
            persona_id,
        )
        raise HTTPException(
            status_code=http_status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "DELETE_FAILED",
                "message": (
                    "DB 삭제 실패 — 파일은 정리됐으나 DB 행이 남아 있습니다. "
                    f"운영자가 수동 정리해야 합니다 ({type(exc).__name__})."
                ),
            },
        )
    return ok(None)
