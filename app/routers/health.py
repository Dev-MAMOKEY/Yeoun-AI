"""`GET /internal/health` — 라이브니스 / 레디니스 신호.

아직 구현되지 않은 서브시스템(모델 레지스트리, DB 리포지토리, 세션 스토어)
관련 필드는 안전한 placeholder를 반환한다. 이슈 #4, #6, #7, #8이 머지되면
실제 값으로 전환된다.
"""

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from .. import __version__
from ..auth import require_internal_token
from ..config import Settings, get_settings
from ..db.engine import ping as db_ping
from ..models.registry import ModelRegistry
from ..schemas.common import Envelope, ok
from ..sessions.store import SessionStore

# health 가 매 호출마다 SELECT 1 으로 DB 상태를 확인할 때 사용하는 타임아웃.
# DB 가 응답이 늦더라도 헬스 응답이 무한정 늘어지지 않도록 짧게 둔다.
_DB_PING_TIMEOUT_SECONDS = 2.0


async def _measure_db_status() -> str:
    """타임아웃 안에서 `SELECT 1` 시도해 'ok' 또는 'error' 반환."""
    try:
        await asyncio.wait_for(db_ping(), timeout=_DB_PING_TIMEOUT_SECONDS)
        return "ok"
    except Exception:  # noqa: BLE001 — 모든 실패를 degraded 로 환원
        return "error"

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
        description=(
            "DB 접속 상태. `USE_DB_MOCK=true` 모드에서는 즉시 `ok`, 실 모드에서는 "
            "매 호출마다 `SELECT 1` 실측 결과. 타임아웃·예외 시 `error`."
        ),
        examples=["ok", "error"],
    )


@router.get(
    "/health",
    response_model=Envelope[HealthData],
    summary="서비스 상태 확인",
    description=(
        "모델 로드 상태·활성 세션 수·GPU 활성 여부·DB 연결 상태를 반환합니다.\n\n"
        "이 엔드포인트는 컨테이너 HEALTHCHECK 와 운영 대시보드에서 사용됩니다.\n"
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
    # 매 호출마다 DB 상태를 실측. lifespan 부팅 시 ping 결과는 캐시하지 않는다.
    db_status = await _measure_db_status()
    session_store: SessionStore | None = getattr(request.app.state, "session_store", None)
    sessions = session_store.count() if session_store is not None else 0
    registry: ModelRegistry | None = getattr(request.app.state, "registry", None)
    llm_status = registry.llm.status if registry is not None and registry.llm is not None else "not_loaded"
    tts_status = registry.tts.status if registry is not None and registry.tts is not None else "not_loaded"
    # Ditto 는 매 render subprocess 모델 — `loaded` 는 "경로 검증 통과, 호출 가능" 의 의미.
    # 평시 GPU 메모리는 점유하지 않는다.
    ditto_status = registry.ditto.status if registry is not None and registry.ditto is not None else "not_loaded"
    models = ModelStatus(llm=llm_status, tts=tts_status, ditto=ditto_status)

    # status 도출:
    # - lifespan 미완료 → degraded
    # - DB ping 실패 → degraded
    # - 상주 모델(LLM/TTS) 둘 다 loaded → ok (Ditto 는 온디맨드라 평시 not_loaded 정상)
    # - 그 외 → starting
    if started_at is None:
        status_value: Literal["ok", "degraded", "starting"] = "degraded"
        uptime = 0.0
    else:
        uptime = max(0.0, time.time() - started_at)
        if db_status == "error":
            status_value = "degraded"
        elif models.llm == "loaded" and models.tts == "loaded":
            status_value = "ok"
        else:
            status_value = "starting"

    return ok(
        HealthData(
            status=status_value,
            version=__version__,
            uptime_seconds=uptime,
            models=models,
            sessions=sessions,
            gpu_enabled=settings.gpu_enabled,
            db=db_status,
        )
    )
