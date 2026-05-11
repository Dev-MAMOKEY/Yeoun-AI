"""대화 세션 인메모리 데이터 모델.

명세서 「공통 규칙」 — 대화 메시지·세션 미디어는 PostgreSQL 에 영속화하지 않고
세션 메모리에만 둔다. 세션 종료 시 즉시 폐기.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID


@dataclass
class MessageRecord:
    """세션 메모리에 머무르는 한 메시지 한 줄. 절대 영속화하지 않는다."""

    role: Literal["user", "assistant"]
    text: str
    created_at: float  # epoch seconds (UTC)


@dataclass
class SessionState:
    """한 대화 세션의 상태.

    `lock` 으로 한 세션의 동시 메시지 처리를 직렬화한다 (한 세션이 두 메시지를
    동시에 보내지 못하게). `last_activity_at` 을 갱신하면 TTL sweeper 가 만료
    여부를 그 시각 기준으로 판정한다.
    """

    session_id: UUID
    user_id: UUID
    persona_id: UUID
    system_prompt: str
    started_at: float
    last_activity_at: float
    history: list[MessageRecord] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
