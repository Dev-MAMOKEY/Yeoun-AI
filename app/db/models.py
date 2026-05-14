"""ERD(2026-05-10) 의 컬럼명을 그대로 따르는 데이터 모델.

명세서 결정에 따라 ERD 의 컬럼명을 Python 속성명에 1:1 로 반영한다.
"""

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class PersonaRecord(BaseModel):
    """`personas` 테이블 1행."""

    personas_id: UUID = Field(..., description="페르소나 PK.")
    owner_user_id: UUID = Field(..., description="페르소나를 소유한 사용자 PK.")
    name: str = Field(..., max_length=100, description="고인의 호칭/이름.")
    nickname: str = Field(..., max_length=50, description="대화에서 사용하는 호칭.")
    status: str = Field(
        ...,
        max_length=20,
        description="페르소나 생성 진척 상태. DRAFT/PROCESSING/READY/FAILED (실 DB `personas_status_check`).",
        examples=["DRAFT", "PROCESSING", "READY", "FAILED"],
    )
    created_at: datetime = Field(..., description="레코드 생성 시각(UTC).")


class InterviewAnswer(BaseModel):
    """`persona_interviews` 테이블 1행. 한 페르소나당 정확히 10개를 시스템 프롬프트에 그대로 박는다."""

    interview_id: UUID
    persona_id: UUID
    question_number: int = Field(..., description="질문 번호 (1-based).")
    answer_text: str | None = Field(None, description="답변 본문. 비어 있을 수 있음.")
    created_at: datetime


class SafetyEvent(BaseModel):
    """`safety_logs` 테이블 1행. (logs_id, user_id) 복합 PK.

    원본 사용자 메시지·키워드 주변 문맥은 절대 기록하지 않는다(명세서 「안전 정책」).
    """

    logs_id: UUID
    user_id: UUID
    event_type: str | None = Field(
        None,
        max_length=50,
        description="감지된 이벤트 분류. crisis_keyword / forbidden_topic / ... 등.",
    )
    # ERD 컬럼명 그대로 (오타: action_taken → action_katen).
    action_katen: str | None = Field(
        None,
        max_length=50,
        description="안전 가드가 취한 조치. block / mask / cooldown 등.",
    )
    # ERD 컬럼명 그대로 (오타: detected_at → deceted_at).
    deceted_at: datetime | None = Field(None, description="이벤트 감지 시각(UTC).")
    cooldown_ended_at: datetime | None = Field(
        None,
        description="쿨다운 종료 시각. NULL 이면 쿨다운 미적용 또는 진행 중.",
    )
