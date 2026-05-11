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

    `/internal/health` 가 매 호출마다 짧은 타임아웃 안에서 부르므로 ORM session
    대신 raw connection (`engine.connect()`) 을 빌려 `SELECT 1` 만 실행한다.
    트랜잭션·rollback 처리 없이 connection 만 풀에 빠르게 반납하므로 누수·
    오염 위험이 없다. 예외 발생 시 호출자가 잡아 'error' 로 환원.
    """
    settings = get_settings()
    if settings.use_db_mock:
        return True
    engine = get_engine()
    if engine is None:
        return False
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))
    return True
