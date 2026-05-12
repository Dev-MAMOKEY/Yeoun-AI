"""대화 세션 라우터 (이슈 #10, 명세서 흐름 B).

엔드포인트:
- POST `/internal/sessions/start` → sessionId 발급
- POST `/internal/sessions/{sessionId}/message` (multipart audio) → SSE
- POST `/internal/sessions/{sessionId}/end` → 세션 정리

큐 길이 초과 시 503 + Retry-After.
"""

import asyncio
import os
import tempfile
import time
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi import status as http_status
from sse_starlette.sse import EventSourceResponse

from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..db import repository
from ..models.registry import ModelRegistry
from ..pipeline.conversation import process_message
from ..prompts.system_prompt import build_system_prompt
from ..schemas.common import Envelope, ok
from ..schemas.session import SessionStartData, SessionStartRequest
from ..sessions.schemas import SessionState
from ..sessions.store import SessionStore

# 동시 SSE 메시지 처리 한도 — 초과 시 503 + Retry-After.
# 단일 워커·GPU 1 장 전제에서 한 번에 너무 많은 메시지가 큐잉되면 응답 지연이
# 클라이언트 timeout 을 넘어가기 시작하므로 보수적 상한을 둔다.
_MAX_CONCURRENT_MESSAGES = 5
_active_messages = 0
_message_counter_lock = asyncio.Lock()

router = APIRouter(prefix="/internal/sessions", tags=["sessions"])


def _service_state(request: Request) -> tuple[ModelRegistry, SessionStore]:
    """app.state 에서 registry·store 를 꺼내거나 503 raise."""
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    store: SessionStore | None = getattr(request.app.state, "session_store", None)
    if registry is None or store is None:
        raise HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "SERVICE_UNAVAILABLE",
                "message": "ModelRegistry 또는 SessionStore 가 초기화되지 않았습니다.",
            },
        )
    return registry, store


@router.post(
    "/start",
    response_model=Envelope[SessionStartData],
    summary="대화 세션 시작",
    description=(
        "페르소나 메타·인터뷰 답변 10개·응답 가이드라인을 시스템 프롬프트로 조립해 "
        "세션 메모리에 보관한다. status='ready' 아닌 페르소나는 409. 응답 sessionId 는 "
        "이후 `/message`·`/end` 호출의 키."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "페르소나 없음 (`NOT_FOUND`)."},
        409: {"description": "페르소나가 ready 상태가 아님 (`CONFLICT`)."},
        503: {"description": "ModelRegistry 또는 SessionStore 가 부팅 전 (`SERVICE_UNAVAILABLE`)."},
    },
)
async def start_session(
    body: SessionStartRequest,
    request: Request,
) -> Envelope[SessionStartData]:
    _registry, store = _service_state(request)

    record = await repository.get_persona(body.persona_id)
    if record is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"페르소나를 찾을 수 없습니다: {body.persona_id}"},
        )
    if record.status != "ready":
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail={
                "code": "CONFLICT",
                "message": f"페르소나가 ready 상태가 아닙니다 (status={record.status}).",
            },
        )

    interviews = await repository.get_persona_interviews(body.persona_id)
    system_prompt = build_system_prompt(record, interviews)

    now = time.time()
    session = SessionState(
        session_id=uuid4(),
        user_id=body.user_id,
        persona_id=body.persona_id,
        system_prompt=system_prompt,
        started_at=now,
        last_activity_at=now,
    )
    store.create(session)
    return ok(SessionStartData(session_id=session.session_id))


@router.post(
    "/{session_id}/message",
    summary="세션 메시지 처리 (SSE)",
    description=(
        "사용자 음성 wav/webm 를 multipart 로 받아 한 메시지의 전체 흐름을 SSE 로 송출:\n"
        "- `event: token` — Gemma 응답 부분 텍스트\n"
        "- `event: text_done` — 최종 텍스트 + message_id\n"
        "- `event: media_ready` — TTS+Ditto 합성 완료 mp4 경로\n"
        "- `event: crisis` — 위기 키워드 감지 시 즉시 종료\n"
        "- `event: error` — 처리 실패"
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
        404: {"description": "세션 없음 (`NOT_FOUND`)."},
        503: {"description": "동시 처리 한도 초과 (`TOO_MANY_REQUESTS`) 또는 부팅 전 (`SERVICE_UNAVAILABLE`)."},
    },
)
async def post_message(
    session_id: UUID,
    request: Request,
    audio: UploadFile = File(..., description="사용자 발화 wav/webm 파일"),
    settings: Settings = Depends(get_settings),
):
    global _active_messages
    registry, store = _service_state(request)
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail={"code": "NOT_FOUND", "message": f"세션을 찾을 수 없습니다: {session_id}"},
        )

    async with _message_counter_lock:
        if _active_messages >= _MAX_CONCURRENT_MESSAGES:
            raise HTTPException(
                status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "TOO_MANY_REQUESTS",
                    "message": f"동시 처리 한도 {_MAX_CONCURRENT_MESSAGES} 초과. Retry-After 후 재시도.",
                },
                headers={"Retry-After": "30"},
            )
        _active_messages += 1

    # audio 임시 저장 — 진입 try 안에서 실패하면 counter 즉시 복구.
    try:
        suffix = Path(audio.filename or "audio.wav").suffix or ".wav"
        fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="msg_")
        os.close(fd)
        tmp_path = Path(tmp_name)
        content = await audio.read()
        tmp_path.write_bytes(content)
        await audio.close()
    except Exception:
        async with _message_counter_lock:
            _active_messages -= 1
        raise

    async def event_generator():
        global _active_messages
        try:
            # 같은 세션의 동시 메시지 차단 (한 세션은 한 번에 한 메시지만).
            async with session.lock:
                async for ev in process_message(session, tmp_path, registry, settings.persona_dir):
                    yield ev
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            async with _message_counter_lock:
                _active_messages -= 1

    return EventSourceResponse(event_generator())
