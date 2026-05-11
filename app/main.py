"""FastAPI 애플리케이션 진입점.

Yeoun Persona Engine 서비스를 조립한다:
- lifespan: 시작 시각 기록, stdlib 로깅 설정
- routers: `/internal/*` 아래에 마운트된 라우터들을 등록
- exception handlers: 모든 에러를 표준 봉투로 감싸 반환
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .config import get_settings
from .db.engine import ping as db_ping
from .routers import health
from .schemas.common import fail
from .sessions.store import SessionStore

logger = logging.getLogger("yeoun")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """프로세스 부팅 시 공유 상태를 초기화.

    후속 이슈에서 모델 로더, DB 엔진, 세션 스토어가 같은 자리에 연결된다.
    지금은 시작 시각 기록과 stdlib 로깅 설정만 수행.
    """
    settings = get_settings()
    # `basicConfig`는 핸들러가 이미 있으면 no-op이라 (예: pytest `caplog`,
    # 운영자 사전 설정) `force=True`를 두지 않아 매번 lifespan 시작 시 핸들러가
    # 리셋되지 않게 한다.
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app.state.started_at = time.time()

    # 부팅 시 DB 연결성 1회 진단 — 부팅 직후 빠르게 알려주기 위한 로그용.
    # 런타임 db_status 는 캐시하지 않고 health 엔드포인트가 매 호출마다 ping 한다.
    try:
        await db_ping()
        logger.info("DB ping 성공")
    except Exception as exc:  # noqa: BLE001 — 부팅 진단 로그만, 기동은 계속
        logger.warning("DB ping 실패 (런타임 health 가 재시도): %s", exc)

    # 세션 스토어 + TTL sweeper 백그라운드 태스크 기동.
    session_store = SessionStore()
    await session_store.start()
    app.state.session_store = session_store

    logger.info(
        "Yeoun Persona Engine 시작 (version=%s, gpu_enabled=%s, db_mock=%s)",
        __version__,
        settings.gpu_enabled,
        settings.use_db_mock,
    )
    try:
        yield
    finally:
        await session_store.stop()
        logger.info("Yeoun Persona Engine 종료 중")


_DESCRIPTION = """
**Yeoun Persona Engine** — 여운(Yeoun) 디지털 추모 서비스의 로컬 AI 서버.

## 인증
모든 `/internal/*` 호출은 다음 헤더를 포함해야 합니다:

```
Authorization: Bearer <INTERNAL_TOKEN>
```

토큰은 Spring Boot와 환경변수로 공유합니다. 외부 사용자는 직접 호출하지 않으며,
WireGuard 내부망에서 Spring Boot가 프록시합니다.

## 응답 포맷
명세서 공통 규칙에 따라 모든 응답은 봉투(Envelope) 구조입니다:

```json
{ "success": true,  "data": { ... }, "error": null }
{ "success": false, "data": null,    "error": { "code": "...", "message": "..." } }
```

## 명세서
- [기능 명세서 (Notion)](https://www.notion.so/35b0e70c795680059393e2999c46e320)
"""


app = FastAPI(
    title="Yeoun Persona Engine",
    description=_DESCRIPTION,
    version=__version__,
    lifespan=lifespan,
    docs_url="/internal/docs",
    redoc_url="/internal/redoc",
    openapi_url="/internal/openapi.json",
)


# HTTP 상태 코드 → Envelope 에러 코드 매핑 (detail 이 dict 형식이 아닐 때 사용).
_STATUS_CODE_TO_ERROR_CODE: dict[int, str] = {
    400: "BAD_REQUEST",
    401: "UNAUTHORIZED",
    403: "FORBIDDEN",
    404: "NOT_FOUND",
    405: "METHOD_NOT_ALLOWED",
    409: "CONFLICT",
    422: "VALIDATION_ERROR",
    429: "TOO_MANY_REQUESTS",
    500: "INTERNAL_ERROR",
    503: "SERVICE_UNAVAILABLE",
}


@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        envelope = fail(str(exc.detail["code"]), str(exc.detail["message"]))
    else:
        code = _STATUS_CODE_TO_ERROR_CODE.get(exc.status_code, "ERROR")
        envelope = fail(code, str(exc.detail) if exc.detail else code)
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(),
        headers=exc.headers,
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=fail("VALIDATION_ERROR", "요청 본문 검증에 실패했습니다.").model_dump(),
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "처리되지 않은 예외 발생: %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=fail("INTERNAL_ERROR", "서버 내부 오류가 발생했습니다.").model_dump(),
    )


app.include_router(health.router)
