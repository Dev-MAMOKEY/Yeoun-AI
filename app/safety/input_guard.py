"""사용자 입력 위기 키워드 매칭.

LLM 추론 전에 호출되어 위기 키워드를 감지한다. 매칭 시 호출자(#10 conversation
pipeline) 가 LLM 추론을 즉시 중단하고 위기 안내 카드 오버레이로 전환한다
(명세서 「안전 정책」).

키워드 매칭은 단순 정규화 substring 비교 — 외부 API·ML 호출 없음.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .keywords import CRISIS_KEYWORDS, normalize


@dataclass(frozen=True)
class CrisisResult:
    """위기 매칭 결과.

    `matched=True` 면 호출자는 LLM 추론을 호출하지 않고 위기 안내 카드 응답을
    보낸 뒤 `repository.insert_safety_log` 로 비식별 이벤트 1 행만 기록한다
    (원본 메시지·문맥은 절대 기록하지 않는다 — 명세서 「안전 정책」).
    """

    matched: bool
    matched_keyword: str | None = None
    action: Literal["block", "pass"] = "pass"


def check_crisis(text: str) -> CrisisResult:
    """트랜스크립트에서 위기 키워드를 검색해 결과 반환.

    공백 제거 + 소문자 정규화 후 `CRISIS_KEYWORDS` 어간형 substring 매칭.
    매칭 시 첫 번째 키워드를 반환하고 단축 평가로 즉시 종료한다.
    """
    if not text:
        return CrisisResult(matched=False)
    haystack = normalize(text)
    for keyword in CRISIS_KEYWORDS:
        needle = normalize(keyword)
        if needle and needle in haystack:
            return CrisisResult(matched=True, matched_keyword=keyword, action="block")
    return CrisisResult(matched=False)
