"""대화 세션 라우터 (이슈 #10, 명세서 흐름 B).

엔드포인트:
- POST `/internal/sessions/start` → sessionId 발급
- POST `/internal/sessions/{sessionId}/message` (multipart audio) → SSE
- POST `/internal/sessions/{sessionId}/end` → 세션 정리

큐 길이 초과 시 503 + Retry-After.
"""

import time
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi import status as http_status

from ..auth import require_internal_token
from ..db import repository
from ..models.registry import ModelRegistry
from ..prompts.system_prompt import build_system_prompt
from ..schemas.common import Envelope, ok
from ..schemas.session import SessionStartData, SessionStartRequest
from ..sessions.schemas import SessionState
from ..sessions.store import SessionStore

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
