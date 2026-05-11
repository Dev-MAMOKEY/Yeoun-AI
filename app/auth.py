"""Internal Bearer-token authentication for `/internal/*` routes.

The token is shared with Spring Boot via the `INTERNAL_TOKEN` env var.
Spring Boot's per-user authentication (JWT) lives upstream; once a
request reaches this service over WireGuard it carries the internal
token instead.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings

bearer_scheme = HTTPBearer(
    auto_error=False,
    bearerFormat="opaque",
    description="`INTERNAL_TOKEN` 환경변수로 Spring Boot와 공유하는 내부 서비스 토큰.",
)


async def require_internal_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject requests without a valid internal Bearer token."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_TOKEN",
                "message": "Authorization 헤더가 없거나 형식이 잘못됐습니다.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not secrets.compare_digest(credentials.credentials, settings.internal_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN",
                "message": "인증 토큰이 유효하지 않습니다.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
