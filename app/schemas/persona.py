"""페르소나 생성·조회 응답 스키마.

이슈 #9 의 라우트(`/internal/personas/{id}/process|status|idle-clips`) 가 반환하는
페이로드 모델. 모든 응답은 `Envelope[T]` 에 감싸 반환된다 ([[Envelope]]).
"""

from enum import Enum
from uuid import UUID

from pydantic import BaseModel, Field


class ProcessingStep(str, Enum):
    """페르소나 생성 진행률 단계.

    명세서 항목 `processing.step` 의 도메인. status=`processing` 일 때 어디까지
    왔는지를 클라이언트가 폴링으로 확인할 수 있도록 노출한다.
    """

    PENDING = "pending"  # 작업 대기 중 (process 호출 직후)
    SELECTING_VOICE = "selecting_voice"  # voice 파일 선택 단계
    TRANSCRIBING = "transcribing"  # Gemma 전사 단계
    EXTRACTING_REF = "extracting_ref"  # OmniVoice ref 추출 단계
    RENDERING_IDLE = "rendering_idle"  # Ditto idle 렌더 단계
    READY = "ready"  # 완료
    FAILED = "failed"  # 실패


class PersonaStatusData(BaseModel):
    """`GET /internal/personas/{id}/status` 응답 페이로드."""

    persona_id: UUID = Field(..., description="페르소나 PK.")
    status: str = Field(
        ...,
        description="DB 상의 페르소나 status — created/processing/ready/failed.",
        examples=["processing", "ready", "failed"],
    )
    step: ProcessingStep | None = Field(
        None,
        description=(
            "현재 진행률 세부 단계. status=`processing` 일 때만 의미. "
            "워커 메모리 기반(In-memory PersonaProcessingStore)이라 워커 재시작 시 사라진다."
        ),
    )
    error_reason: str | None = Field(
        None,
        description="status=`failed` 일 때 운영자/Spring 진단용 한국어 에러 요약.",
        examples=["Gemma 전사 실패: cuda OOM"],
    )


class IdleClipMeta(BaseModel):
    """단일 idle 클립의 메타."""

    index: int = Field(..., ge=0, description="클립 번호 — 0/1 둘 중 하나.")
    path: str = Field(
        ...,
        description=(
            "PERSONA_DIR 기준 상대 경로(`{persona_id}/idle/{n}.mp4`). "
            "내부 절대경로 노출을 피하기 위해 컨테이너 마운트와 무관한 상대 표현 사용. "
            "실제 mp4 스트리밍은 이슈 #11 의 미디어 라우터가 별도 URL 로 제공."
        ),
        examples=["00000000-0000-0000-0000-000000000001/idle/0.mp4"],
    )
    size_bytes: int = Field(..., ge=0, description="파일 크기(바이트).")


class IdleClipsData(BaseModel):
    """`GET /internal/personas/{id}/idle-clips` 응답 페이로드."""

    persona_id: UUID = Field(..., description="페르소나 PK.")
    clips: list[IdleClipMeta] = Field(
        ...,
        description="idle 클립 메타 목록. status=`ready` 일 때 보통 2개.",
    )
