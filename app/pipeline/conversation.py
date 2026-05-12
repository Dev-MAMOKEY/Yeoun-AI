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
        logger.info(
            "위기 키워드 감지: session=%s, keyword=%s", session.session_id, crisis.matched_keyword
        )
        yield {
            "event": "crisis",
            "data": json.dumps(
                {
                    "message_id": str(message_id),
                    "keyword": crisis.matched_keyword,
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

    # 4. 출력 안전 필터 — 금지 주제 키워드 매칭 시 안전 fallback 으로 대체 (이슈 #5).
    filtered = filter_response(full_text)
    if filtered.matched:
        logger.info(
            "금지 주제 감지: session=%s, keyword=%s",
            session.session_id,
            filtered.matched_keyword,
        )
        final_text = "(이 주제는 답변드리기 어려워요. 다른 이야기를 나눌까요?)"
    else:
        final_text = filtered.text

    yield {
        "event": "text_done",
        "data": json.dumps({"message_id": str(message_id), "text": final_text}),
    }

    # 5. 세션 history 갱신 — 텍스트만 보관 (명세 「공통 규칙」 영속화 금지).
    now = time.time()
    session.history.append(MessageRecord(role="user", text=user_text, created_at=now))
    session.history.append(MessageRecord(role="assistant", text=final_text, created_at=now))
    session.last_activity_at = now

    # 6. TTS + Ditto 합성 — 본 SSE 안에서 await, 완료 후 media_ready yield.
    try:
        persona_root = Path(persona_dir) / str(session.persona_id)
        speak_dir = persona_root / "speak" / str(session.session_id)
        speak_dir.mkdir(parents=True, exist_ok=True)
        wav_path = speak_dir / f"{message_id}.wav"
        mp4_path = speak_dir / f"{message_id}.mp4"

        voice_ref_dir = persona_root / "voice_ref"
        ref_text_path = voice_ref_dir / "ref_text.txt"
        ref_audio_candidates = sorted(voice_ref_dir.glob("ref_audio.*")) if voice_ref_dir.exists() else []
        if not ref_audio_candidates or not ref_text_path.exists():
            raise FileNotFoundError(f"voice_ref 누락: {voice_ref_dir}")
        ref_audio_path = ref_audio_candidates[0]
        ref_text = ref_text_path.read_text(encoding="utf-8")

        async with registry.gpu_semaphore:
            await registry.tts.synthesize(
                text=final_text,
                ref_audio_path=str(ref_audio_path),
                ref_text=ref_text,
                output_path=str(wav_path),
            )

        photo_dir = persona_root / "photo"
        photo_candidates = (
            [p for p in photo_dir.iterdir() if p.is_file()] if photo_dir.exists() else []
        )
        if not photo_candidates:
            raise FileNotFoundError(f"photo 누락: {photo_dir}")
        photo_path = max(photo_candidates, key=lambda p: p.stat().st_mtime)

        async with registry.ditto_semaphore:
            await registry.ditto.render_speak(
                image_path=str(photo_path),
                audio_path=str(wav_path),
                output_path=str(mp4_path),
            )
    except Exception as exc:  # noqa: BLE001 — 합성 실패는 SSE 에러로 환원
        logger.exception("미디어 합성 실패: session=%s", session.session_id)
        yield {
            "event": "error",
            "data": json.dumps({"reason": f"media_failed: {type(exc).__name__}: {exc}"[:200]}),
        }
        return

    # PERSONA_DIR 기준 상대 경로 — 내부 절대경로 노출 차단 (#9 의 IdleClipMeta.path 와 동일 패턴).
    relative_mp4 = f"{session.persona_id}/speak/{session.session_id}/{message_id}.mp4"
    yield {
        "event": "media_ready",
        "data": json.dumps({"message_id": str(message_id), "path": relative_mp4}),
    }
