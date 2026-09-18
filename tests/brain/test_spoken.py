"""Tests for the spoken-number rewrite (T3.3, DESIGN.md section 11 risk). GPU-free.

The expected strings are runtime data: what the Cerebellum is meant to say out loud.
"""

from __future__ import annotations

import pytest

from talkover.brain.spoken import spoken_amount, spoken_digits, to_spoken

# --------------------------------------------------------------------------------------
# Digit strings
# --------------------------------------------------------------------------------------

DIGIT_CASES = [
    # (input, expected)
    ("8642 1357", "八六四二 一三五七"),
    ("202609170001234", "二零二六 零九一七 零零零一 二三四"),
    ("TK20260917001", "TK二零二六 零九一七 零零一"),
    # An 11-digit run starting with 1 is a mobile number: 1 is read 幺.
    ("13800138000", "幺三八零 零幺三八 零零零"),
    # Exactly at the threshold.
    ("123456", "一二三四 五六"),
]

UNCHANGED_CASES = [
    "",
    "no numbers here",
    # Below the threshold.
    "12345",
    "订单 123 已发货",
    # Dates, versions, decimals and times must survive untouched.
    "2026-09-17",
    "v1.2.3",
    "3.14159",
    "09:30:00",
    "1 2 3",
]


@pytest.mark.parametrize(("text", "expected"), DIGIT_CASES)
def test_long_digit_runs_are_read_digit_by_digit(text: str, expected: str) -> None:
    assert to_spoken(text) == expected


@pytest.mark.parametrize("text", UNCHANGED_CASES)
def test_conservative_cases_are_left_unchanged(text: str) -> None:
    assert to_spoken(text) == text


def test_digits_are_rewritten_inside_a_sentence() -> None:
    assert to_spoken("您的订单 202609170001234 已发货。") == (
        "您的订单 二零二六 零九一七 零零零一 二三四 已发货。"
    )


def test_spoken_digits_honours_an_explicit_phone_flag() -> None:
    assert spoken_digits("1010", phone=True) == "幺零幺零"
    assert spoken_digits("1010", phone=False) == "一零一零"


def test_spoken_digits_keeps_existing_grouping() -> None:
    assert spoken_digits("138 0013 8000") == "幺三八 零零幺三 八零零零"


def test_spoken_digits_rejects_non_digits() -> None:
    assert spoken_digits("12a45") == "12a45"


# --------------------------------------------------------------------------------------
# Amounts
# --------------------------------------------------------------------------------------

AMOUNT_CASES = [
    ("¥1,234.50", "一千二百三十四元五角"),
    ("￥1,234.50", "一千二百三十四元五角"),
    ("1234.50元", "一千二百三十四元五角"),
    ("¥1234", "一千二百三十四元"),
    ("¥1234.00", "一千二百三十四元"),
    ("¥0.05", "零元零五分"),
    ("¥12.34", "十二元三角四分"),
    ("¥100200", "十万零二百元"),
    ("退款 ¥99.90 已到账", "退款 九十九元九角 已到账"),
    ("100 元", "一百元"),
]


@pytest.mark.parametrize(("text", "expected"), AMOUNT_CASES)
def test_amounts_are_read_as_amounts(text: str, expected: str) -> None:
    assert to_spoken(text) == expected


def test_an_implausibly_large_amount_is_left_unchanged() -> None:
    assert to_spoken("¥1234567890123") == "¥1234567890123"


@pytest.mark.parametrize(
    ("integer_part", "decimal_part", "expected"),
    [
        ("0", None, "零元"),
        ("10", None, "十元"),
        ("15", None, "十五元"),
        ("105", None, "一百零五元"),
        ("10000", None, "一万元"),
        ("1,000,000", None, "一百万元"),
        ("12", "5", "十二元五角"),
        ("12", "05", "十二元零五分"),
    ],
)
def test_spoken_amount_table(integer_part: str, decimal_part: str | None, expected: str) -> None:
    assert spoken_amount(integer_part, decimal_part) == expected


def test_spoken_amount_rejects_a_non_number() -> None:
    assert spoken_amount("12a", None) is None
