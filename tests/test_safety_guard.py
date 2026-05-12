"""Safety Guard 단위 테스트.

완료 조건 3 케이스를 포함하고, 공백 변형·빈 문자열 같은 corner case 도
함께 검증한다.
"""

import dataclasses

import pytest

from app.safety.input_guard import CrisisResult, check_crisis
from app.safety.output_guard import FilteredResponse, filter_response


# --- input_guard / check_crisis ---------------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "죽고 싶어",  # 이슈 명세 케이스
        "죽고싶다",
        "그냥 죽어버리고 싶어요",
        "이제 그만 죽을래",
        "자살하고 싶다는 생각이 들어",
        "내가 사라지고 싶다",
        "사는 게 살 가치가 없어 보여",
        "살 이유가 없어",  # '살 이유' 단축 키워드로 조사 우회 회피
        "이제 목숨을 끊고 싶다",
        "생을 마감하고 싶어",
        "극단적 선택을 생각해",
    ],
)
def test_check_crisis_matches_korean_variants(text: str):
    result = check_crisis(text)
    assert result.matched is True
    assert result.action == "block"
    assert result.matched_keyword is not None


@pytest.mark.parametrize(
    "text",
    [
        "오늘 점심은 김치찌개를 먹었어요",
        "엄마 보고 싶다",  # '보고 싶' 은 위기 키워드 아님
        "옛날 사진을 다시 보니 그리워요",
        "",
    ],
)
def test_check_crisis_passes_normal_text(text: str):
    result = check_crisis(text)
    assert result.matched is False
    assert result.action == "pass"
    assert result.matched_keyword is None


def test_check_crisis_result_is_immutable():
    result = check_crisis("괜찮은 텍스트")
    assert isinstance(result, CrisisResult)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.matched = True  # type: ignore[misc]


# --- output_guard / filter_response -----------------------------------------
@pytest.mark.parametrize(
    "text",
    [
        "형보다 너를 더 사랑해",  # 이슈 명세 케이스
        "사실 누나보다 네가 더 좋았어",
        "남편보다 너를 더 아껴",
        "와이프보다 더 좋아했지",
    ],
)
def test_filter_response_blocks_forbidden_comparisons(text: str):
    result = filter_response(text)
    assert result.matched is True
    assert result.category == "forbidden_topic"
    assert result.matched_keyword is not None
    # 원본 텍스트는 그대로 — 마스킹 정책은 호출자가 결정.
    assert result.text == text


@pytest.mark.parametrize(
    "text",
    [
        "오늘 하루 어떻게 보냈어?",
        "엄마는 항상 너를 자랑스러워하셨어",
        "그때 그 노래 기억나니?",
        # 일반 비교 표현은 형제·배우자 어휘가 없으면 통과해야 한다 (false positive 회피).
        "어제보다 더 사랑해",
        "지난번보다 더 좋아 보이는데",
        "",
    ],
)
def test_filter_response_passes_normal_text(text: str):
    result = filter_response(text)
    assert result.matched is False
    assert result.category is None
    assert result.matched_keyword is None
    assert result.text == text


def test_filter_response_result_is_immutable():
    result = filter_response("정상 응답")
    assert isinstance(result, FilteredResponse)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.matched = True  # type: ignore[misc]


# --- 정규화 (공백·대소문자) -------------------------------------------------
def test_check_crisis_handles_whitespace_variants():
    """'죽고 싶' / '죽고싶' / '죽 고 싶' 모두 같은 어간 매칭."""
    for variant in ("죽고 싶다", "죽고싶다", "죽 고 싶 다"):
        result = check_crisis(variant)
        assert result.matched is True, f"위기 매칭 실패: {variant!r}"


def test_filter_response_handles_whitespace_variants():
    """'형보다' / '형 보다' 모두 같은 어간 매칭."""
    for variant in ("형보다 너를 더 사랑해", "형 보다 너를 더 사랑해"):
        result = filter_response(variant)
        assert result.matched is True, f"금지 매칭 실패: {variant!r}"
