"""세션 시작/메시지/종료 흐름 더미 모드 스모크 테스트.

GPU_ENABLED=false + USE_DB_MOCK=true 에서 시스템 프롬프트 조립, conversation.process_message
정상/위기 분기, POST /start·/end 라우터를 검증한다. SSE 응답 자체(`/message`)는 라이브
검증·운영자 검증에 의존하며, 본 스위트는 conversation.process_message 를 직접 호출.
"""

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from app.config import get_settings
from app.db import repository
from app.db.models import InterviewAnswer, PersonaRecord
from app.models.registry import ModelRegistry
from app.pipeline.conversation import process_message
from app.prompts.system_prompt import build_system_prompt
from app.sessions.schemas import SessionState
from tests.conftest import TEST_TOKEN


def _run(coro):
    """동기 테스트에서 async helper 한 줄 실행 — Python 3.14 에서 get_event_loop deprecation 회피."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_persona(persona_id: UUID, status: str = "ready") -> PersonaRecord:
    return PersonaRecord(
        personas_id=persona_id,
        owner_user_id=uuid4(),
        id=None,
        name="고인",
        nickname="할아버지",
        status=status,
        created_at=datetime.now(timezone.utc),
    )


def _make_interview(persona_id: UUID, n: int) -> InterviewAnswer:
    return InterviewAnswer(
        interview_id=uuid4(),
        persona_id=persona_id,
        question_number=n,
        answer_text=f"답변 {n}",
        created_at=datetime.now(timezone.utc),
    )


def _seed_files(persona_dir: Path, persona_id: UUID) -> None:
    """voice_ref + photo 디렉토리 시드 — 합성 단계 통과용."""
    root = persona_dir / str(persona_id)
    (root / "voice_ref").mkdir(parents=True)
    (root / "voice_ref" / "ref_text.txt").write_text("기준 음성 전사", encoding="utf-8")
    (root / "voice_ref" / "ref_audio.wav").write_bytes(b"RIFF....WAVE")
    (root / "photo").mkdir(parents=True)
    (root / "photo" / "face.jpg").write_bytes(b"\xff\xd8\xff\xe0")


# --- 시스템 프롬프트 빌더 -----------------------------------------------------


def test_build_system_prompt_includes_persona_and_interviews():
    persona = _make_persona(uuid4())
    interviews = [_make_interview(persona.personas_id, i) for i in (3, 1, 2)]
    text = build_system_prompt(persona, interviews, today=date(2026, 5, 12))

    assert "할아버지" in text
    assert "2026-05-12" in text
    # 인터뷰는 question_number 오름차순 출력.
    assert text.index("1. 답변 1") < text.index("2. 답변 2") < text.index("3. 답변 3")
    assert "위기" in text or "상담" in text  # 가이드라인 포함


def test_build_system_prompt_with_no_interviews():
    persona = _make_persona(uuid4())
    text = build_system_prompt(persona, [], today=date(2026, 5, 12))
    # 인터뷰 섹션은 빈 입력에서 생략.
    assert "생전 인터뷰" not in text
    assert "할아버지" in text


# --- conversation.process_message --------------------------------------------


async def _collect_events(session: SessionState, audio_path: Path, registry, persona_dir: str) -> list[dict]:
    events: list[dict] = []
    async for ev in process_message(session, audio_path, registry, persona_dir):
        events.append(ev)
    return events


async def test_process_message_dummy_full_flow(tmp_path: Path):
    settings = get_settings()
    assert settings.gpu_enabled is False
    await repository._reset_mock()

    persona_id = uuid4()
    _seed_files(tmp_path, persona_id)
    audio = tmp_path / "msg.wav"
    audio.write_bytes(b"RIFF....WAVE")

    registry = ModelRegistry(settings)
    await registry.start()
    try:
        session = SessionState(
            session_id=uuid4(),
            user_id=uuid4(),
            persona_id=persona_id,
            system_prompt="dummy prompt",
            started_at=0.0,
            last_activity_at=0.0,
        )
        events = await _collect_events(session, audio, registry, str(tmp_path))
    finally:
        await registry.stop()

    kinds = [ev["event"] for ev in events]
    assert "token" in kinds
    assert "text_done" in kinds
    assert "media_ready" in kinds
    # 위기 이벤트는 없어야 (transcribe 더미가 위기 키워드 아님).
    assert "crisis" not in kinds
    # text_done 직후 history 가 2건 추가 (user + assistant).
    assert len(session.history) == 2


async def test_process_message_crisis_branch(tmp_path: Path, monkeypatch):
    """transcribe 결과를 위기 키워드로 강제하면 즉시 event:crisis 만 송출."""
    settings = get_settings()
    await repository._reset_mock()

    audio = tmp_path / "msg.wav"
    audio.write_bytes(b"RIFF....WAVE")

    registry = ModelRegistry(settings)
    await registry.start()

    async def fake_transcribe(audio_path):  # noqa: ARG001 — instance bind 후라 self 자동
        return "정말 죽고 싶다"

    # GemmaLLM.transcribe 만 위기 텍스트로 우회 — 인스턴스 단위 패치.
    monkeypatch.setattr(registry.llm, "transcribe", fake_transcribe)
    try:
        session = SessionState(
            session_id=uuid4(),
            user_id=uuid4(),
            persona_id=uuid4(),
            system_prompt="dummy",
            started_at=0.0,
            last_activity_at=0.0,
        )
        events = await _collect_events(session, audio, registry, str(tmp_path))
    finally:
        await registry.stop()

    assert len(events) == 1
    assert events[0]["event"] == "crisis"
    # 위기 분기에선 history 갱신·합성 안 함.
    assert len(session.history) == 0
    # safety_logs 행 1건 INSERT — 비식별 이벤트만.
    logs = await repository.mock_get_safety_logs()
    assert len(logs) == 1
    assert logs[0].event_type == "crisis_keyword"
    assert logs[0].action_katen == "block"


# --- 라우터 (TestClient) ------------------------------------------------------


def test_start_session_404_when_persona_missing(client):
    payload = {"user_id": str(uuid4()), "persona_id": str(uuid4())}
    res = client.post(
        "/internal/sessions/start",
        json=payload,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 404
    body = res.json()
    assert body["success"] is False
    assert body["error"]["code"] == "NOT_FOUND"


def test_start_session_409_when_not_ready(client):
    pid = uuid4()
    _run(repository.mock_seed_persona(_make_persona(pid, status="processing")))
    payload = {"user_id": str(uuid4()), "persona_id": str(pid)}
    res = client.post(
        "/internal/sessions/start",
        json=payload,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 409
    assert res.json()["error"]["code"] == "CONFLICT"


def test_start_then_end_session(client):
    pid = uuid4()
    _run(repository.mock_seed_persona(_make_persona(pid, status="ready")))
    payload = {"user_id": str(uuid4()), "persona_id": str(pid)}
    res = client.post(
        "/internal/sessions/start",
        json=payload,
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res.status_code == 200
    session_id = res.json()["data"]["session_id"]

    res2 = client.post(
        f"/internal/sessions/{session_id}/end",
        headers={"Authorization": f"Bearer {TEST_TOKEN}"},
    )
    assert res2.status_code == 200
    data = res2.json()["data"]
    assert data["session_id"] == session_id
    assert data["cleaned_messages"] == 0
