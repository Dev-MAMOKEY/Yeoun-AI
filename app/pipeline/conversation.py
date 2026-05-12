"""대화 메시지 한 건의 SSE 흐름 (이슈 #10, 명세서 흐름 B).

`process_message` 가 audio → text 전사 → 위기 검증 → Gemma 응답 스트림 → 출력
필터 → TTS+Ditto 합성 → media_ready 순서로 SSE 이벤트 dict 를 yield 한다.
라우터(`app/routers/sessions.py`)는 이 비동기 iterator 를 `EventSourceResponse`
로 wrap 해 클라이언트에 송출.

SSE 이벤트 종류:
- `token` — Gemma 응답 토큰(부분 텍스트)
- `text_done` — 응답 전체 텍스트 + message_id
- `media_ready` — TTS+Ditto 합성 완료, mp4 경로
- `crisis` — 위기 키워드 감지, 안전 가이드 메시지로 종료
- `error` — 처리 실패
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import AsyncIterator
from uuid import uuid4

from ..safety.input_guard import check_crisis
from ..safety.output_guard import filter_response
from ..sessions.schemas import MessageRecord, SessionState

logger = logging.getLogger("yeoun")


async def process_message(
    session: SessionState,
    audio_path: Path,
    registry,  # ModelRegistry — 순환 import 회피
    persona_dir: str,
) -> AsyncIterator[dict]:
    """메시지 한 건의 SSE 이벤트 dict 를 yield.

    호출자(`app/routers/sessions.py`)가 세션 락을 잡고 본 generator 를
    `EventSourceResponse` 로 감싸 송출한다. 본 함수는 락을 건드리지 않는다.
    """
    message_id = uuid4()

    # 1. 사전 전사 — 입력 안전 가드용. Gemma audio-in 으로 deterministic 전사.
    try:
        user_text = await registry.llm.transcribe(audio_path)
    except Exception as exc:  # noqa: BLE001 — 전사 실패는 SSE 에러로 환원
        logger.exception("transcribe 실패: session=%s", session.session_id)
        yield {"event": "error", "data": json.dumps({"reason": f"transcribe_failed: {type(exc).__name__}"})}
        return

    # 2. 입력 안전 가드 — 위기 키워드 매칭 시 event:crisis 송출 후 종료.
    crisis = check_crisis(user_text)
    if crisis.matched:
        logger.info("위기 키워드 감지: session=%s, keyword=%s", session.session_id, crisis.keyword)
        yield {
            "event": "crisis",
            "data": json.dumps(
                {
                    "message_id": str(message_id),
                    "keyword": crisis.keyword,
                    "guidance": "지금 많이 힘드신가요? 24시간 상담 전화 109 로 연락해 주세요.",
                }
            ),
        }
        return

    # 3. Gemma 응답 스트림 — audio-in 으로 직접 처리, 토큰을 누적하며 SSE 로 전달.
    history_messages = [{"role": m.role, "text": m.text} for m in session.history]
    full_text_parts: list[str] = []
    try:
        async for token in registry.llm.stream_response(
            system_prompt=session.system_prompt,
            history=history_messages,
            user_audio_path=audio_path,
        ):
            full_text_parts.append(token)
            yield {"event": "token", "data": token}
    except Exception as exc:  # noqa: BLE001 — 스트림 실패는 SSE 에러로 환원
        logger.exception("stream_response 실패: session=%s", session.session_id)
        yield {"event": "error", "data": json.dumps({"reason": f"stream_failed: {type(exc).__name__}"})}
        return

    full_text = "".join(full_text_parts).strip()

    # 4. 출력 안전 필터 — 금지 주제 키워드 마스킹 (이슈 #5).
    filtered = filter_response(full_text)
    final_text = filtered.text

    yield {
        "event": "text_done",
        "data": json.dumps({"message_id": str(message_id), "text": final_text}),
    }
