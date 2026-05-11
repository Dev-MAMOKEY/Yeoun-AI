"""SQLAlchemy async 엔진과 세션 메이커.

`USE_DB_MOCK=true` 일 때는 실제 엔진을 만들지 않고 None 을 반환한다.
repository 계층이 mock 분기로 우회하므로 실 PostgreSQL 연결 없이도 개발/
테스트가 가능하다.

엔진은 `lru_cache` 로 싱글톤 유지. 테스트에서 환경 변수 변경 후 캐시를
비우려면 `get_engine.cache_clear()` 와 `get_sessionmaker.cache_clear()` 호출.
"""

from functools import lru_cache

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..config import get_settings


@lru_cache
def get_engine() -> AsyncEngine | None:
    """프로세스 전역 async 엔진을 반환. mock 모드면 None."""
    settings = get_settings()
    if settings.use_db_mock:
        return None
    return create_async_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        pool_recycle=300,
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession] | None:
    """async session 메이커 반환. mock 모드면 None."""
    engine = get_engine()
    if engine is None:
        return None
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def ping() -> bool:
    """실 DB 연결성 점검. `SELECT 1` 한 번. mock 모드면 항상 True.

    헬스 체크 부팅 단계(`/internal/health` 의 `db` 필드 도출) 에서 사용.
    예외 발생 시 False 반환하여 호출자가 `db_status="error"` 로 표시.
    """
    settings = get_settings()
    if settings.use_db_mock:
        return True
    session_maker = get_sessionmaker()
    if session_maker is None:
        return False
    async with session_maker() as session:
        await session.execute(text("SELECT 1"))
    return True
