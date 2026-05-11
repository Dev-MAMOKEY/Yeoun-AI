"""페르소나·인터뷰·안전로그 영속화 함수.

스키마는 ERD(2026-05-10) 기준. 실제 SQL 구현은 Spring/DB 팀과 마이그레이션
합의 후 추가한다. 현재는 mock 분기만 채워두고 실 DB 호출은
`NotImplementedError` 로 막아 silent bug 를 방지한다.

`USE_DB_MOCK=true` 일 때:
- 모든 함수가 프로세스 인메모리 dict 를 대상으로 동작
- 테스트·로컬 개발에서 실 DB 없이도 흐름을 끝까지 돌릴 수 있음
- 인메모리 데이터는 워커 재시작 시 사라짐
"""

import asyncio
from datetime import datetime
from uuid import UUID

from ..config import get_settings
from .models import InterviewAnswer, PersonaRecord, SafetyEvent

# --- 프로세스 인메모리 mock 저장소 -------------------------------------------
# 워커 1개 가정. asyncio.Lock 으로 동시 갱신 직렬화.
_mock_personas: dict[UUID, PersonaRecord] = {}
_mock_interviews: dict[UUID, list[InterviewAnswer]] = {}
_mock_safety_logs: list[SafetyEvent] = []
_mock_lock = asyncio.Lock()


def _real_db_not_supported(reason: str) -> NotImplementedError:
    return NotImplementedError(
        f"{reason}. 실 PostgreSQL 스키마 합의 후 SQL 구현 예정. "
        "현재는 USE_DB_MOCK=true 로 운영/테스트하십시오."
    )


# --- 페르소나 ----------------------------------------------------------------
async def get_persona(persona_id: UUID) -> PersonaRecord | None:
    """`personas` 1행 조회 (PK personas_id)."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _mock_lock:
            return _mock_personas.get(persona_id)
    raise _real_db_not_supported("get_persona")


async def update_persona_status(persona_id: UUID, status: str) -> None:
    """`personas.status` 갱신."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _mock_lock:
            record = _mock_personas.get(persona_id)
            if record is None:
                return
            _mock_personas[persona_id] = record.model_copy(update={"status": status})
        return
    raise _real_db_not_supported("update_persona_status")


async def delete_persona_tx(persona_id: UUID) -> None:
    """페르소나 + CASCADE 관계 행을 한 트랜잭션으로 삭제.

    파일시스템 자원 정리는 이슈 #12 의 `storage/filesystem.py` 와 함께
    한 트랜잭션으로 묶일 예정 (이슈 #4 범위는 DB 측 시그니처만).
    """
    settings = get_settings()
    if settings.use_db_mock:
        async with _mock_lock:
            _mock_personas.pop(persona_id, None)
            _mock_interviews.pop(persona_id, None)
        return
    raise _real_db_not_supported("delete_persona_tx")


# --- 인터뷰 답변 -------------------------------------------------------------
async def get_persona_interviews(persona_id: UUID) -> list[InterviewAnswer]:
    """`persona_interviews` 의 전체 답변(보통 10개)을 question_number 오름차순으로 반환."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _mock_lock:
            answers = list(_mock_interviews.get(persona_id, []))
        return sorted(answers, key=lambda a: a.question_number)
    raise _real_db_not_supported("get_persona_interviews")


# --- 안전 로그 ---------------------------------------------------------------
async def insert_safety_log(
    *,
    logs_id: UUID,
    user_id: UUID,
    event_type: str | None,
    action_katen: str | None,
    deceted_at: datetime | None,
    cooldown_ended_at: datetime | None = None,
) -> None:
    """`safety_logs` 신규 행 삽입.

    원본 메시지·키워드 주변 문맥은 인자로 받지 않는다 — 명세서 「안전 정책」 에
    따라 비식별 패턴 모니터링 전용.
    """
    settings = get_settings()
    if settings.use_db_mock:
        event = SafetyEvent(
            logs_id=logs_id,
            user_id=user_id,
            event_type=event_type,
            action_katen=action_katen,
            deceted_at=deceted_at,
            cooldown_ended_at=cooldown_ended_at,
        )
        async with _mock_lock:
            _mock_safety_logs.append(event)
        return
    raise _real_db_not_supported("insert_safety_log")


# --- mock 헬퍼 (테스트·시드 용도) -------------------------------------------
async def mock_seed_persona(record: PersonaRecord) -> None:
    """테스트 편의: mock 저장소에 페르소나 1건 시드."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_seed_persona 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _mock_lock:
        _mock_personas[record.personas_id] = record


async def mock_seed_interviews(persona_id: UUID, answers: list[InterviewAnswer]) -> None:
    """테스트 편의: mock 저장소에 인터뷰 답변 시드."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_seed_interviews 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _mock_lock:
        _mock_interviews[persona_id] = list(answers)


async def mock_get_safety_logs() -> list[SafetyEvent]:
    """테스트 검증용: mock 안전 로그 스냅샷."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_get_safety_logs 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _mock_lock:
        return list(_mock_safety_logs)


async def _reset_mock() -> None:
    """테스트 격리: 모든 mock 저장소를 비운다."""
    async with _mock_lock:
        _mock_personas.clear()
        _mock_interviews.clear()
        _mock_safety_logs.clear()
