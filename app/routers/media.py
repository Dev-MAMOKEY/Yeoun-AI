"""미디어 스트리밍 라우터.

엔드포인트:
- GET `/internal/personas/{id}/idle-clips/{idx}` — `idle/{idx}.mp4` 스트리밍
- GET `/internal/sessions/{sessionId}/messages/{msgId}/media?kind=audio|video`
   — `speak/{sessionId}/{msgId}.{wav,mp4}` 스트리밍

HTTP Range 헤더를 지원해 영상 시킹·부분 다운로드가 가능하다. 외부에서 들어온
경로 인자는 `safe_resolve` 로 PERSONA_DIR 밖 탈출을 차단.
"""

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi import status as http_status
from fastapi.responses import StreamingResponse

from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..db import repository
from ..sessions.store import SessionStore
from ..storage.filesystem import (
    RangeNotSatisfiable,
    iter_file_range,
    parse_range,
    safe_resolve,
)

router = APIRouter(prefix="/internal", tags=["media"])

_VIDEO_MEDIA_TYPE = "video/mp4"
_AUDIO_MEDIA_TYPE = "audio/wav"
_FULL_CHUNK = 64 * 1024


def _stream_response(path: Path, range_header: str | None, media_type: str) -> StreamingResponse:
    """파일을 전체(200) 또는 Range(206) 응답으로 송출. 416 은 raise."""
    size = path.stat().st_size
    try:
        rng = parse_range(range_header, size)
    except RangeNotSatisfiable as exc:
        raise HTTPException(
            status_code=http_status.HTTP_416_RANGE_NOT_SATISFIABLE,
            headers={"Content-Range": f"bytes */{size}"},
            detail={"code": "RANGE_NOT_SATISFIABLE", "message": str(exc)},
        )

    if rng is None:
        async def full_iter():
            with path.open("rb") as f:
                while True:
                    data = f.read(_FULL_CHUNK)
                    if not data:
                        break
                    yield data

        return StreamingResponse(
            full_iter(),
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(size)},
        )

    start, end = rng
    return StreamingResponse(
        iter_file_range(path, start, end),
        status_code=http_status.HTTP_206_PARTIAL_CONTENT,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{size}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(end - start + 1),
        },
    )


@router.get(
    "/personas/{persona_id}/idle-clips/{idx}",
    summary="페르소나 idle 클립 스트리밍",
    description=(
        "페르소나의 idle 클립(0/1) mp4 를 송출한다. HTTP Range 헤더가 있으면 206 + Content-Range, "
        "없으면 200 전체. 파일이 없거나 페르소나가 없으면 404."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "전체 mp4 응답.", "content": {"video/mp4": {}}},
        206: {"description": "Range 부분 응답.", "content": {"video/mp4": {}}},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 또는 클립 없음 (`NOT_FOUND`)."},
        416: {"description": "Range 가 파일 크기 밖 (`RANGE_NOT_SATISFIABLE`)."},
    },
)
async def get_idle_clip(
    persona_id: UUID,
    idx: int,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if idx not in (0, 1):
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"유효하지 않은 idle 인덱스: {idx}"},
        )
    record = await repository.get_persona(persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {persona_id}"},
        )
    try:
        path = safe_resolve(settings.persona_dir, str(persona_id), "idle", f"{idx}.mp4")
    except PermissionError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(exc)},
        )
    if not path.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"idle 클립 파일이 없습니다: {idx}.mp4"},
        )
    return _stream_response(path, request.headers.get("range"), _VIDEO_MEDIA_TYPE)


@router.get(
    "/sessions/{session_id}/messages/{message_id}/media",
    summary="세션 메시지 미디어 스트리밍",
    description=(
        "세션 메시지의 TTS wav(`kind=audio`) 또는 Ditto mp4(`kind=video`) 를 송출한다. "
        "세션이 종료되어 메모리에서 사라지면 404. HTTP Range 지원."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        200: {"description": "전체 미디어 응답."},
        206: {"description": "Range 부분 응답."},
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "세션·메시지 없음 (`NOT_FOUND`)."},
        416: {"description": "Range 가 파일 크기 밖 (`RANGE_NOT_SATISFIABLE`)."},
        503: {"description": "SessionStore 가 부팅 전 (`SERVICE_UNAVAILABLE`)."},
    },
)
async def get_session_media(
    session_id: UUID,
    message_id: UUID,
    request: Request,
    kind: str = Query(..., pattern="^(audio|video)$", description="`audio` (wav) 또는 `video` (mp4)."),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    store: SessionStore | None = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "SERVICE_UNAVAILABLE", "message": "SessionStore 미초기화."},
        )
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"세션을 찾을 수 없습니다: {session_id}"},
        )

    suffix = "wav" if kind == "audio" else "mp4"
    media_type = _AUDIO_MEDIA_TYPE if kind == "audio" else _VIDEO_MEDIA_TYPE
    try:
        path = safe_resolve(
            settings.persona_dir,
            str(session.persona_id),
            "speak",
            str(session_id),
            f"{message_id}.{suffix}",
        )
    except PermissionError as exc:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": str(exc)},
        )
    if not path.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"미디어 파일이 없습니다: {message_id}.{suffix}"},
        )
    return _stream_response(path, request.headers.get("range"), media_type)
