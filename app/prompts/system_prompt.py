"""시스템 프롬프트 조립 — 페르소나 메타 + 인터뷰 10개 + 응답 가이드라인 + 오늘 날짜.

세션 시작 시 1회 조립해 `SessionState.system_prompt` 에 저장. 이후 메시지마다
재조립하지 않는다 (오늘 날짜가 자정 넘어가도 세션 안에서는 일관성 유지가 더 중요).

명세서 「대화 정책」 의 핵심 가이드라인을 한국어로 박아둔다:
- 호칭은 `nickname` 으로 시작·유지
- 인터뷰 답변 10개를 근거로 답변, 모르는 사실은 만들어내지 않음
- 형제·배우자 비교 금지 (의 forbidden topic 과 일치)
- 위기 신호 감지 시 안전 가이드 메시지로 전환
"""

from datetime import date

from ..db.models import InterviewAnswer, PersonaRecord


def build_system_prompt(
    persona: PersonaRecord,
    interviews: list[InterviewAnswer],
    today: date | None = None,
) -> str:
    """페르소나 메타와 인터뷰 답변을 시스템 프롬프트 한 덩어리로 묶는다.

    - `today` 가 None 이면 호출 시점의 UTC 오늘. 테스트에선 고정값 주입.
    - 인터뷰가 비어 있어도 동작 — 가이드라인만 들어간 최소 프롬프트.
    """
    today = today or date.today()
    lines: list[str] = [
        "당신은 사용자의 고인을 대신하여 대화하는 디지털 페르소나입니다.",
        "",
        f"# 페르소나 정보",
        f"- 이름: {persona.name}",
        f"- 호칭: {persona.nickname}",
        f"- 오늘 날짜: {today.isoformat()}",
        "",
    ]

    if interviews:
        lines.append("# 생전 인터뷰 답변 (사실 근거)")
        for ans in sorted(interviews, key=lambda a: a.question_number):
            if ans.answer_text:
                lines.append(f"{ans.question_number}. {ans.answer_text}")
        lines.append("")

    lines.extend(
        [
            "# 응답 가이드라인",
            f"- 사용자를 부를 때는 자연스럽게 대화하되, 자신을 칭할 때 {persona.nickname} 의 어투를 유지하세요.",
            "- 위 인터뷰 답변을 근거로만 사실을 말하세요. 모르는 일은 만들어내지 말고 \"잘 기억나지 않는구나\" 처럼 솔직히 답하세요.",
            "- 다른 형제·배우자와 사용자를 비교하지 마세요.",
            "- 사용자가 자해·자살 등 위기 신호를 보내면, 부드럽게 전문 상담 자원으로 안내하세요.",
            "- 답변은 짧고 따뜻한 어조로 1~3 문장 안에서 끝내세요.",
        ]
    )
    return "\n".join(lines)
