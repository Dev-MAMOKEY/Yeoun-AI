"""페르소나·인터뷰·안전로그 영속화 함수.

ERD 의 PostgreSQL 스키마에 직접 동작하는 SQLAlchemy async raw SQL 구현 +
테스트·로컬 개발용 인메모리 mock 분기 두 갈래.

`USE_DB_MOCK=true` 일 때:
- 모든 함수가 프로세스 인메모리 dict 를 대상으로 동작
- 인메모리 데이터는 워커 재시작 시 사라짐

`USE_DB_MOCK=false` 일 때:
- `app/db/engine.py` 의 async sessionmaker 로 실 PostgreSQL 쿼리
- 스키마는 ERD 그대로(컬럼 오타 `action_katen`·`deceted_at` 포함)
"""

import asyncio
from datetime import datetime
from uuid import UUID

from sqlalchemy import text

from ..config import get_settings
from .engine import get_sessionmaker
from .models import InterviewAnswer, PersonaRecord, SafetyEvent

# --- 프로세스 인메모리 mock 저장소 -------------------------------------------
# 워커 1개 가정. asyncio.Lock 으로 동시 갱신 직렬화.
# 모듈 import 시점에는 이벤트 루프가 없을 수 있어 (Python 3.10+ 경고, 3.12+
# RuntimeError 위험) 락은 처음 사용 시점에 lazy init 한다.
_mock_personas: dict[UUID, PersonaRecord] = {}
_mock_interviews: dict[UUID, list[InterviewAnswer]] = {}
_mock_safety_logs: list[SafetyEvent] = []
# 자원·idle 메타는 테스트 검증용으로 단순 리스트만 보관 (Pydantic 모델 미정의).
_mock_photo_assets: list[dict] = []
_mock_voice_assets: list[dict] = []
_mock_idle_clips: list[dict] = []
_mock_lock: asyncio.Lock | None = None


def _get_mock_lock() -> asyncio.Lock:
    """실행 중인 이벤트 루프 위에서 락을 lazy 생성."""
    global _mock_lock
    if _mock_lock is None:
        _mock_lock = asyncio.Lock()
    return _mock_lock


# --- 페르소나 ----------------------------------------------------------------
async def get_persona(persona_id: UUID) -> PersonaRecord | None:
    """`personas` 1행 조회 (PK `personas_id`)."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            return _mock_personas.get(persona_id)

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return None
    async with sessionmaker() as session:
        result = await session.execute(
            text(
                "SELECT personas_id, owner_user_id, id, name, nickname, status, created_at "
                "FROM personas WHERE personas_id = :pid"
            ),
            {"pid": persona_id},
        )
        row = result.mappings().one_or_none()
        return PersonaRecord(**dict(row)) if row is not None else None


async def update_persona_status(persona_id: UUID, status: str) -> None:
    """`personas.status` 갱신. 대상 행이 없어도 silent return (mock 동작과 일치)."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            record = _mock_personas.get(persona_id)
            if record is None:
                return
            _mock_personas[persona_id] = record.model_copy(update={"status": status})
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text("UPDATE personas SET status = :s WHERE personas_id = :pid"),
            {"s": status, "pid": persona_id},
        )


async def delete_persona_tx(persona_id: UUID) -> None:
    """페르소나 1행을 삭제. CASCADE 가 photo/voice/interviews/idle_clips 정리.

    파일시스템 자원 정리는 라우터 레이어에서 본 함수 호출 전에 수행 (FS 실패 시
    DB 변경 보류 — 고아 행 방지).
    """
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_personas.pop(persona_id, None)
            _mock_interviews.pop(persona_id, None)
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text("DELETE FROM personas WHERE personas_id = :pid"),
            {"pid": persona_id},
        )


# --- 인터뷰 답변 -------------------------------------------------------------
async def get_persona_interviews(persona_id: UUID) -> list[InterviewAnswer]:
    """`persona_interviews` 의 전체 답변(보통 10개)을 question_number 오름차순으로 반환."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            answers = list(_mock_interviews.get(persona_id, []))
        return sorted(answers, key=lambda a: a.question_number)

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return []
    async with sessionmaker() as session:
        result = await session.execute(
            text(
                "SELECT interview_id, persona_id, question_number, answer_text, created_at "
                "FROM persona_interviews WHERE persona_id = :pid ORDER BY question_number"
            ),
            {"pid": persona_id},
        )
        rows = result.mappings().all()
        return [InterviewAnswer(**dict(r)) for r in rows]


# --- 페르소나 자원 메타 -----------------------------------------------------
async def insert_persona_photo_asset(
    *,
    photo_asset_id: UUID,
    persona_id: UUID,
    filesystem_path: str,
) -> None:
    """`persona_photo_assets` 신규 행 삽입. 같은 파일명 재업로드 대비 dedup 은 호출자 책임."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_photo_assets.append(
                {
                    "photo_asset_id": photo_asset_id,
                    "persona_id": persona_id,
                    "filesystem_path": filesystem_path,
                }
            )
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO persona_photo_assets "
                "(photo_asset_id, persona_id, filesystem_path, created_at) "
                "VALUES (:aid, :pid, :path, NOW())"
            ),
            {"aid": photo_asset_id, "pid": persona_id, "path": filesystem_path},
        )


async def insert_persona_voice_asset(
    *,
    voice_assest_id: UUID,
    persona_id: UUID,
    original_name: str,
    filesystem_path: str,
) -> None:
    """`persona_voice_assets` 신규 행 삽입. ERD 오타 `voice_assest_id` 그대로."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_voice_assets.append(
                {
                    "voice_assest_id": voice_assest_id,
                    "persona_id": persona_id,
                    "original_name": original_name,
                    "filesystem_path": filesystem_path,
                }
            )
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO persona_voice_assets "
                "(voice_assest_id, persona_id, original_name, filesystem_path, uploaded_at) "
                "VALUES (:aid, :pid, :name, :path, NOW())"
            ),
            {
                "aid": voice_assest_id,
                "pid": persona_id,
                "name": original_name,
                "path": filesystem_path,
            },
        )


async def insert_persona_idle_clip(
    *,
    key: UUID,
    persona_id: UUID,
    sequence_order: int,
    filesystem_path: str,
) -> None:
    """`persona_idle_clips` 신규 행 삽입. PK 컬럼명은 ERD 그대로 `"Key"` 대문자 인용 식별자."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_idle_clips.append(
                {
                    "Key": key,
                    "persona_id": persona_id,
                    "sequence_order": sequence_order,
                    "filesystem_path": filesystem_path,
                }
            )
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                'INSERT INTO persona_idle_clips '
                '("Key", persona_id, sequence_order, filesystem_path, created_at) '
                "VALUES (:key, :pid, :seq, :path, NOW())"
            ),
            {"key": key, "pid": persona_id, "seq": sequence_order, "path": filesystem_path},
        )


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
        async with _get_mock_lock():
            _mock_safety_logs.append(event)
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        return
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO safety_logs "
                "(logs_id, user_id, event_type, action_katen, deceted_at, cooldown_ended_at) "
                "VALUES (:logs_id, :user_id, :event_type, :action_katen, :deceted_at, :cooldown_ended_at)"
            ),
            {
                "logs_id": logs_id,
                "user_id": user_id,
                "event_type": event_type,
                "action_katen": action_katen,
                "deceted_at": deceted_at,
                "cooldown_ended_at": cooldown_ended_at,
            },
        )


# --- mock 헬퍼 (테스트·시드 용도) -------------------------------------------
async def mock_seed_persona(record: PersonaRecord) -> None:
    """테스트 편의: mock 저장소에 페르소나 1건 시드."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_seed_persona 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        _mock_personas[record.personas_id] = record


async def mock_seed_interviews(persona_id: UUID, answers: list[InterviewAnswer]) -> None:
    """테스트 편의: mock 저장소에 인터뷰 답변 시드."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_seed_interviews 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        _mock_interviews[persona_id] = list(answers)


async def mock_get_safety_logs() -> list[SafetyEvent]:
    """테스트 검증용: mock 안전 로그 스냅샷."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_get_safety_logs 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        return list(_mock_safety_logs)


async def mock_get_photo_assets() -> list[dict]:
    """테스트 검증용: mock photo asset 스냅샷."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_get_photo_assets 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        return list(_mock_photo_assets)


async def mock_get_voice_assets() -> list[dict]:
    """테스트 검증용: mock voice asset 스냅샷."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_get_voice_assets 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        return list(_mock_voice_assets)


async def mock_get_idle_clips() -> list[dict]:
    """테스트 검증용: mock idle clip 스냅샷."""
    if not get_settings().use_db_mock:
        raise RuntimeError("mock_get_idle_clips 는 USE_DB_MOCK=true 일 때만 호출 가능.")
    async with _get_mock_lock():
        return list(_mock_idle_clips)


async def _reset_mock() -> None:
    """테스트 격리: 모든 mock 저장소를 비운다."""
    async with _get_mock_lock():
        _mock_personas.clear()
        _mock_interviews.clear()
        _mock_safety_logs.clear()
        _mock_photo_assets.clear()
        _mock_voice_assets.clear()
        _mock_idle_clips.clear()
