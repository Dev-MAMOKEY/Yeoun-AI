"""SessionStore — 인메모리 dict + 30분 TTL sweeper.

명세서 결정에 따라 Redis 미사용. uvicorn 워커 1 개 전제로 프로세스 단일 저장소를
운영한다. 워커가 재시작되면 모든 세션이 사라진다 (의도된 폐기, 명세서 「공통 규칙」).

라이프사이클은 FastAPI `lifespan` 에서 다룬다:
- `start()` 가 진입 시 호출되어 sweeper 백그라운드 태스크를 깨운다.
- `stop()` 이 종료 시 sweeper 를 취소·대기한다.
"""

from __future__ import annotations

import asyncio
import logging
import time
from uuid import UUID

from .schemas import MessageRecord, SessionState

logger = logging.getLogger("yeoun")

_DEFAULT_TTL_SECONDS = 30 * 60  # 명세서 — 30분 비활성 시 세션 만료
_DEFAULT_SWEEP_INTERVAL = 60  # 60초마다 sweep


class SessionStore:
    """프로세스 단일 인메모리 세션 저장소."""

    def __init__(
        self,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
        sweep_interval: float = _DEFAULT_SWEEP_INTERVAL,
    ) -> None:
        self._sessions: dict[UUID, SessionState] = {}
        self._ttl = ttl_seconds
        self._sweep_interval = sweep_interval
        self._sweeper_task: asyncio.Task[None] | None = None

    # --- 라이프사이클 -------------------------------------------------------
    async def start(self) -> None:
        """sweeper 백그라운드 태스크 기동. 이미 실행 중이면 무시."""
        if self._sweeper_task is not None and not self._sweeper_task.done():
            return
        self._sweeper_task = asyncio.create_task(self._sweep_loop(), name="session-sweeper")

    async def stop(self) -> None:
        """sweeper 를 취소하고 종료를 기다린다."""
        task = self._sweeper_task
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self._sweeper_task = None

    # --- CRUD ---------------------------------------------------------------
    def create(self, session: SessionState) -> None:
        """새 세션을 저장소에 등록."""
        self._sessions[session.session_id] = session

    def get(self, session_id: UUID) -> SessionState | None:
        """세션 ID 로 조회. 없으면 None."""
        return self._sessions.get(session_id)

    def touch(self, session_id: UUID) -> bool:
        """마지막 활동 시각 갱신. 세션이 없으면 False."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.last_activity_at = time.time()
        return True

    def add_message(self, session_id: UUID, message: MessageRecord) -> bool:
        """세션 history 에 메시지 추가 + last_activity_at 갱신."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        session.history.append(message)
        session.last_activity_at = message.created_at
        return True

    def end(self, session_id: UUID) -> SessionState | None:
        """세션 종료 — 메모리에서 제거. 호출자가 파일시스템 정리를 별도로 수행한다."""
        return self._sessions.pop(session_id, None)

    def count(self) -> int:
        """현재 활성 세션 수."""
        return len(self._sessions)

    # --- TTL sweeper --------------------------------------------------------
    async def _sweep_loop(self) -> None:
        """`_sweep_interval` 초마다 만료 세션을 폐기. 예외 발생해도 루프는 유지."""
        while True:
            try:
                await asyncio.sleep(self._sweep_interval)
                expired = self._sweep_once()
                if expired:
                    logger.info("세션 sweeper: %d 개 만료 세션 폐기", expired)
            except asyncio.CancelledError:
                logger.debug("세션 sweeper 종료")
                break
            except Exception:
                logger.exception("세션 sweeper 루프 예외 — 다음 주기 계속")

    def _sweep_once(self) -> int:
        """현재 시각 기준 만료된 세션을 dict 에서 제거. 폐기 개수 반환."""
        now = time.time()
        cutoff = now - self._ttl
        expired_ids = [
            sid for sid, session in self._sessions.items() if session.last_activity_at < cutoff
        ]
        for sid in expired_ids:
            del self._sessions[sid]
        return len(expired_ids)
