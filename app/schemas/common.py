"""Common response envelope shared by every `/internal/*` endpoint.

Spec (`공통 규칙`): every response is wrapped in
`{ "success": bool, "data": ... | null, "error": { code, message } | null }`.
"""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class ErrorBody(BaseModel):
    code: str = Field(
        ...,
        description="Machine-readable error code (e.g. UNAUTHORIZED, VALIDATION_ERROR).",
        examples=["UNAUTHORIZED"],
    )
    message: str = Field(
        ...,
        description="Human-readable Korean error message for operators.",
        examples=["인증 토큰이 유효하지 않습니다."],
    )


class Envelope(BaseModel, Generic[T]):
    """Standard envelope for every `/internal/*` response."""

    success: bool = Field(
        ...,
        description="`true` on success, `false` when `error` is populated.",
    )
    data: T | None = Field(
        None,
        description="Payload on success; `null` on failure.",
    )
    error: ErrorBody | None = Field(
        None,
        description="Error details on failure; `null` on success.",
    )


def ok(data: T) -> Envelope[T]:
    """Wrap a success payload in the standard envelope."""
    return Envelope[T](success=True, data=data, error=None)


def fail(code: str, message: str) -> Envelope[None]:
    """Wrap an error in the standard envelope."""
    return Envelope[None](success=False, data=None, error=ErrorBody(code=code, message=message))
