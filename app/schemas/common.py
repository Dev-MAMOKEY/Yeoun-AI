"""모든 `/internal/*` 엔드포인트가 공유하는 공통 응답 봉투(Envelope).

명세서 「공통 규칙」 — 모든 응답은
`{ "success": bool, "data": ... | null, "error": { code, message } | null }`
구조로 감싸 반환한다.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str = Field(
        ...,
        description="기계 판독용 에러 코드 (예: UNAUTHORIZED, VALIDATION_ERROR).",
        examples=["UNAUTHORIZED"],
    )
    message: str = Field(
        ...,
        description="운영자가 읽을 수 있는 한국어 에러 메시지.",
        examples=["인증 토큰이 유효하지 않습니다."],
    )


class Envelope(BaseModel, Generic[T]):
    """모든 `/internal/*` 응답이 따르는 표준 봉투."""

    success: bool = Field(
        ...,
        description="성공이면 true, 실패이면 false (이때 `error`가 채워짐).",
    )
    data: T | None = Field(
        None,
        description="성공 시 페이로드. 실패 시 null.",
    )
    error: ErrorBody | None = Field(
        None,
        description="실패 시 에러 본문. 성공 시 null.",
    )


def ok(data: T) -> Envelope[T]:
    """성공 페이로드를 표준 봉투로 감싸 반환."""
    # 본문 안에서 `Envelope[T]` 서브스크립트를 쓰면 런타임에 T가 TypeVar
    # 자체라 Pydantic이 미해결 제너릭으로 받아 `data`를 Any로 폴백한다.
    # 인자 없는 생성자로 두면 런타임 스키마가 정상적으로 잡히고, 시그니처의
    # 반환 타입 표기는 호출자에게 그대로 정보를 전달한다.
    return Envelope(success=True, data=data, error=None)


def fail(code: str, message: str) -> Envelope[None]:
    """에러를 표준 봉투로 감싸 반환."""
    return Envelope(success=False, data=None, error=ErrorBody(code=code, message=message))
