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
import logging
from datetime import datetime
from typing import NoReturn
from uuid import UUID

from sqlalchemy import text

from ..config import get_settings
from .engine import get_sessionmaker
from .models import InterviewAnswer, PersonaRecord, SafetyEvent

logger = logging.getLogger("yeoun")


# 로그 redact 대상 — 명세서 「안전 정책: 원본 메시지·비식별 패턴 모니터링 전용」 에
# 따라 식별자(UUID PK) 는 운영 로그에 평문 노출 금지. 부팅 결함 진단에 필요한 메타
# (event_type·status·sequence_order 등) 는 보존.
_PII_LOG_KEYS = frozenset(
    {
        "user_id",
        "persona_id",
        "logs_id",
        "photo_asset_id",
        "voice_asset_id",
        "clip_id",
    }
)


def _raise_db_uninitialized(func_name: str, **context: object) -> NoReturn:
    """쓰기 함수의 `sessionmaker is None` 분기 공통 처리.

    silent return 이 부팅 결함을 가리면 호출자(process_persona 등) 가 `status='FAILED'`
    같은 핵심 갱신을 못 한 채 정상 종료한 것처럼 보여 DB 가 `PROCESSING` 으로 굳어지는
    회귀 발생. 본 헬퍼가 ERROR 로그(운영자 즉시 인지) + RuntimeError(호출자 흐름에서
    잡혀 사용자에게 의미 있는 응답으로 환원) 두 가지를 수행. 읽기 함수(get_*) 는
    빈 결과 반환이 graceful degradation 이라 본 헬퍼를 호출하지 않는다.

    context 에 포함된 식별자(UUID PK) 키는 명세서 안전 정책에 따라 평문 로그 노출
    금지 — `<redacted>` 로 치환 후 운영자 진단용 메타(event_type 등) 만 보존.
    """
    safe_context = {
        key: ("<redacted>" if key in _PII_LOG_KEYS else value)
        for key, value in context.items()
    }
    logger.error(
        "%s sessionmaker 미초기화 (USE_DB_MOCK=false 부팅 결함): context=%s",
        func_name, safe_context,
    )
    raise RuntimeError(
        f"{func_name}: DB sessionmaker 가 초기화되지 않았습니다 — "
        "USE_DB_MOCK=true 가 아닌데 부팅 단계에서 DB 연결이 실패한 상태."
    )

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
                "SELECT personas_id, owner_user_id, name, nickname, status, created_at "
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
        _raise_db_uninitialized(
            "update_persona_status", persona_id=persona_id, status=status,
        )
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text("UPDATE personas SET status = :s WHERE personas_id = :pid"),
            {"s": status, "pid": persona_id},
        )


async def delete_persona_tx(persona_id: UUID) -> None:
    """페르소나 1행 + 자식 4개 테이블의 참조 행을 한 트랜잭션 안에서 삭제.

    실 DB DDL 이 자식 테이블의 FK 에 `ON DELETE CASCADE` 를 두지 않아 부모만 바로
    삭제하면 `ForeignKeyViolationError`. 따라서 본 함수가 자식부터 명시적 CASCADE.
    `safety_logs` 는 `user_id` 복합 PK 라 persona_id 직접 참조 안 함 → 제외.

    파일시스템 자원 정리는 라우터 레이어에서 본 함수 호출 전에 수행 (FS 실패 시
    DB 변경 보류 — 고아 행 방지).
    """
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            # 실 DB 분기와 같은 자식→부모 순서로 정리해 두 분기의 의도 비교 용이.
            _mock_interviews.pop(persona_id, None)
            _mock_photo_assets[:] = [a for a in _mock_photo_assets if a.get("persona_id") != persona_id]
            _mock_voice_assets[:] = [a for a in _mock_voice_assets if a.get("persona_id") != persona_id]
            _mock_idle_clips[:] = [c for c in _mock_idle_clips if c.get("persona_id") != persona_id]
            _mock_personas.pop(persona_id, None)
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        _raise_db_uninitialized("delete_persona_tx", persona_id=persona_id)
    async with sessionmaker() as session, session.begin():
        # 자식 테이블 4개 먼저 DELETE — FK 위반 회피. safety_logs 는 user_id 복합 PK
        # 라 persona_id 직접 참조 안 함 → 본 함수에서 정리 대상 아님.
        await session.execute(
            text("DELETE FROM persona_interviews WHERE persona_id = :pid"),
            {"pid": persona_id},
        )
        await session.execute(
            text("DELETE FROM persona_photo_assets WHERE persona_id = :pid"),
            {"pid": persona_id},
        )
        await session.execute(
            text("DELETE FROM persona_voice_assets WHERE persona_id = :pid"),
            {"pid": persona_id},
        )
        await session.execute(
            text("DELETE FROM persona_idle_clips WHERE persona_id = :pid"),
            {"pid": persona_id},
        )
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
        _raise_db_uninitialized(
            "insert_persona_photo_asset",
            persona_id=persona_id, photo_asset_id=photo_asset_id,
        )
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
    voice_asset_id: UUID,
    persona_id: UUID,
    original_name: str,
    filesystem_path: str,
) -> None:
    """`persona_voice_assets` 신규 행 삽입. 실 DB 는 ERD 오타를 정정한 `voice_asset_id` 사용."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_voice_assets.append(
                {
                    "voice_asset_id": voice_asset_id,
                    "persona_id": persona_id,
                    "original_name": original_name,
                    "filesystem_path": filesystem_path,
                }
            )
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        _raise_db_uninitialized(
            "insert_persona_voice_asset",
            persona_id=persona_id, voice_asset_id=voice_asset_id,
        )
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO persona_voice_assets "
                "(voice_asset_id, persona_id, original_name, filesystem_path, uploaded_at) "
                "VALUES (:aid, :pid, :name, :path, NOW())"
            ),
            {
                "aid": voice_asset_id,
                "pid": persona_id,
                "name": original_name,
                "path": filesystem_path,
            },
        )


async def insert_persona_idle_clip(
    *,
    clip_id: UUID,
    persona_id: UUID,
    sequence_order: int,
    filesystem_path: str,
) -> None:
    """`persona_idle_clips` 신규 행 삽입. 실 DB 는 ERD 의 `"Key"` 가 아니라 `clip_id` 사용."""
    settings = get_settings()
    if settings.use_db_mock:
        async with _get_mock_lock():
            _mock_idle_clips.append(
                {
                    "clip_id": clip_id,
                    "persona_id": persona_id,
                    "sequence_order": sequence_order,
                    "filesystem_path": filesystem_path,
                }
            )
        return

    sessionmaker = get_sessionmaker()
    if sessionmaker is None:
        _raise_db_uninitialized(
            "insert_persona_idle_clip",
            persona_id=persona_id, clip_id=clip_id, sequence_order=sequence_order,
        )
    async with sessionmaker() as session, session.begin():
        await session.execute(
            text(
                "INSERT INTO persona_idle_clips "
                "(clip_id, persona_id, sequence_order, filesystem_path, created_at) "
                "VALUES (:cid, :pid, :seq, :path, NOW())"
            ),
            {"cid": clip_id, "pid": persona_id, "seq": sequence_order, "path": filesystem_path},
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
        _raise_db_uninitialized(
            "insert_safety_log",
            logs_id=logs_id, user_id=user_id, event_type=event_type,
        )
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
