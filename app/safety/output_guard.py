"""LLM 응답 금지 주제 매칭.

LLM 추론 직후 호출되어 응답에 금지 주제(형제·배우자 비교 등) 가 포함됐는지
검사한다. 매칭 시 호출자(#10 conversation pipeline) 가 원본 응답을 그대로
내보내지 않고 안전 fallback 으로 대체하거나 LLM 에 재생성을 요청한다.

키워드 매칭은 단순 정규화 substring 비교 — 외부 API·ML 호출 없음.
사후 사건 적극 의견 등 의미 수준 가드는 시스템 프롬프트(#10) 가이드라인에서
담당한다 (본 모듈 범위 밖).
"""

from __future__ import annotations

from dataclasses import dataclass

from .keywords import FORBIDDEN_TOPIC_KEYWORDS, normalize


@dataclass(frozen=True)
class FilteredResponse:
    """출력 필터링 결과.

    `matched=True` 면 호출자는 원본 텍스트 대신 안전 fallback 응답을 사용하거나
    LLM 재생성을 요청한다. 필요 시 `repository.insert_safety_log` 로 비식별
    이벤트 1 행만 기록한다 (원본 응답 본문은 절대 기록하지 않는다).
    """

    text: str  # 원본 텍스트 그대로 — 마스킹/대체 정책은 호출자가 결정
    matched: bool
    matched_keyword: str | None = None
    category: str | None = None  # 현재는 "forbidden_topic" 만; 후속 확장 가능


def filter_response(text: str) -> FilteredResponse:
    """LLM 응답에서 금지 주제 키워드를 검색해 결과 반환."""
    if not text:
        return FilteredResponse(text=text, matched=False)
    haystack = normalize(text)
    for keyword in FORBIDDEN_TOPIC_KEYWORDS:
        needle = normalize(keyword)
        if needle and needle in haystack:
            return FilteredResponse(
                text=text,
                matched=True,
                matched_keyword=keyword,
                category="forbidden_topic",
            )
    return FilteredResponse(text=text, matched=False)
