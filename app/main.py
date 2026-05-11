"""FastAPI application entry point.

Composes the Yeoun Persona Engine service:
- lifespan: records start time, configures stdlib logging
- routers: includes everything mounted under `/internal/*`
- exception handlers: wraps every error in the standard envelope
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
from .routers import health
from .schemas.common import fail

logger = logging.getLogger("yeoun")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire up cross-cutting state at process boot.

    Model loaders, DB engines, and the session store hook in here as
    later issues land. For now we only record the start time and
    configure stdlib logging.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )
    app.state.started_at = time.time()
    logger.info(
        "Yeoun Persona Engine starting (version=%s, gpu_enabled=%s, db_mock=%s)",
        __version__,
        settings.gpu_enabled,
        settings.use_db_mock,
    )
    try:
        yield
    finally:
        logger.info("Yeoun Persona Engine shutting down")


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
        headers=getattr(exc, "headers", None),
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
        "Unhandled exception while processing %s %s",
        request.method,
        request.url.path,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=fail("INTERNAL_ERROR", "서버 내부 오류가 발생했습니다.").model_dump(),
    )


app.include_router(health.router)
