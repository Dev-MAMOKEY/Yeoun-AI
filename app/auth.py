"""`/internal/*` 라우트용 내부 Bearer 토큰 인증.

토큰은 `INTERNAL_TOKEN` 환경 변수로 Spring Boot와 공유한다. 사용자별
인증(JWT)은 Spring Boot가 상위에서 처리하고, 이 서비스로 들어오는 요청은
WireGuard 내부망을 거쳐 내부 토큰만 들고 들어온다.
"""

import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings

bearer_scheme = HTTPBearer(
    auto_error=False,
    description="`INTERNAL_TOKEN` 환경 변수로 Spring Boot와 공유하는 내부 서비스 토큰.",
)


async def require_internal_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """유효한 내부 Bearer 토큰이 없으면 요청을 거부."""
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "MISSING_TOKEN",
                "message": "Authorization 헤더가 없거나 형식이 잘못됐습니다.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
    # 비ASCII 토큰에서 `compare_digest`가 TypeError로 깨지지 않도록 bytes로 비교.
    if not secrets.compare_digest(
        credentials.credentials.encode("utf-8"),
        settings.internal_token.encode("utf-8"),
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "INVALID_TOKEN",
                "message": "인증 토큰이 유효하지 않습니다.",
            },
            headers={"WWW-Authenticate": "Bearer"},
        )
