"""대화 세션 시작·종료 응답 스키마 (이슈 #10, 명세서 흐름 B).

POST /internal/sessions/{sessionId}/message 의 응답은 SSE 스트림이라 Pydantic
응답 모델이 아니라 `EventSourceResponse` 가 직접 토큰을 yield 한다. 본 모듈은
JSON 봉투 응답(시작·종료)만 다룬다.
"""

from uuid import UUID

from pydantic import BaseModel, Field


class SessionStartRequest(BaseModel):
    """POST /internal/sessions/start 요청 본문."""

    user_id: UUID = Field(..., description="대화를 시작하는 사용자 PK.")
    persona_id: UUID = Field(
        ...,
        description="대상 페르소나 PK — `status='ready'` 인 경우만 시작 가능.",
    )


class SessionStartData(BaseModel):
    """POST /internal/sessions/start 응답 페이로드."""

    session_id: UUID = Field(
        ...,
        description="이후 `/message` / `/end` 호출에 사용할 세션 식별자.",
    )


class SessionEndData(BaseModel):
    """POST /internal/sessions/{sessionId}/end 응답 페이로드."""

    session_id: UUID = Field(..., description="종료된 세션의 식별자.")
    cleaned_messages: int = Field(
        ...,
        ge=0,
        description="세션 기간 동안 누적된 메시지 수 — 메모리에서 폐기된 건수.",
    )
