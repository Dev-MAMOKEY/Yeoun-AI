"""`GET /internal/health` — liveness / readiness signal.

Fields that depend on subsystems not yet built (model registry, DB
repository, session store) report safe placeholders. They will switch
to live values as issues #4, #6, #7, #8 land.
"""

import time
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from .. import __version__
from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..schemas.common import Envelope, ok

router = APIRouter(prefix="/internal", tags=["health"])


class ModelStatus(BaseModel):
    llm: str = Field(
        "not_loaded",
        description="Gemma 4 LLM 로더 상태 (placeholder, 이슈 #6 이후 실값).",
        examples=["not_loaded", "loaded", "loading", "error"],
    )
    tts: str = Field(
        "not_loaded",
        description="OmniVoice TTS 로더 상태 (placeholder, 이슈 #7 이후 실값).",
        examples=["not_loaded", "loaded", "loading", "error"],
    )
    ditto: str = Field(
        "not_loaded",
        description="Ditto-TalkingHead 로더 상태 (placeholder, 이슈 #8 이후 실값; 온디맨드 로드라 평시 not_loaded).",
        examples=["not_loaded", "loaded", "loading", "error"],
    )


class HealthData(BaseModel):
    status: Literal["ok", "degraded", "starting"] = Field(
        ...,
        description="`starting` — 부팅 직후, `ok` — 모든 서브시스템 정상, `degraded` — 일부 장애.",
    )
    version: str = Field(..., description="Yeoun Persona Engine 버전.", examples=["0.1.0"])
    uptime_seconds: float = Field(..., description="부팅 후 경과 시간(초).")
    models: ModelStatus
    sessions: int = Field(..., description="현재 활성 대화 세션 수 (이슈 #4/#10 이후 실값).")
    gpu_enabled: bool = Field(..., description="`GPU_ENABLED` 설정. false이면 모델 로더가 더미 동작.")
    db: str = Field(
        ...,
        description="DB 접속 상태. `mock`/`not_initialized`/`ok`/`error`.",
        examples=["mock", "not_initialized", "ok", "error"],
    )


@router.get(
    "/health",
    response_model=Envelope[HealthData],
    summary="서비스 상태 확인",
    description=(
        "모델 로드 상태·활성 세션 수·GPU 활성 여부·DB 연결 상태를 반환합니다.\n\n"
        "이 엔드포인트는 컨테이너 HEALTHCHECK 및 운영 대시보드에서 사용됩니다.\n"
        "응답은 명세서 공통 규칙대로 `{ success, data, error }` 봉투에 감싸여 반환됩니다."
    ),
    dependencies=[Depends(require_internal_token)],
    responses={
        401: {"description": "토큰이 없거나 유효하지 않음 (`UNAUTHORIZED`)."},
    },
)
async def health(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> Envelope[HealthData]:
    started_at: float | None = getattr(request.app.state, "started_at", None)
    if started_at is None:
        # Lifespan never set the start time — surface as `degraded` instead
        # of silently masking the failure with a fresh timestamp.
        status_value: Literal["ok", "degraded", "starting"] = "degraded"
        uptime = 0.0
    else:
        status_value = "starting"
        uptime = max(0.0, time.time() - started_at)
    return ok(
        HealthData(
            status=status_value,
            version=__version__,
            uptime_seconds=uptime,
            models=ModelStatus(),
            sessions=0,
            gpu_enabled=settings.gpu_enabled,
            db="mock" if settings.use_db_mock else "not_initialized",
        )
    )
