"""Rewrite numbers into segmented spoken form for the Cerebellum's TTS.

DESIGN.md section 11 lists "the Cerebellum misreads digit strings (order numbers,
amounts)" as a risk, with "have the Brain rewrite numbers into segmented spoken form
before calling `share`" as the mitigation. This module is that mitigation. It is off by
default and enabled per session through `BrainSession(spoken_numbers=True)`.

Two rewrites are performed, in this order:

* **Amounts** -- ``¥1,234.50`` or ``1234.50元`` become ``一千二百三十四元五角``.
* **Long digit strings** -- six or more digits, optionally already grouped with spaces,
  are read digit by digit in groups of four: ``8642 1357`` becomes ``八六四二 一三五七``.
  An 11-digit run starting with ``1`` is treated as a mainland mobile number, so ``1`` is
  read ``幺`` to keep it distinct from ``七`` on a telephony channel.

Everything else is left alone. The rule is deliberately conservative: short numbers,
decimals, dates (``2026-09-17``), version strings and percentages are never touched,
because a wrong rewrite is worse than no rewrite. Digit runs separated by anything other
than a single space are not joined, which is what keeps dates out.

The Chinese characters in this module are runtime data (what the customer hears), as
allowed by CLAUDE.md. All comments and identifiers stay in English.
"""

from __future__ import annotations

import re

__all__ = [
    "MIN_SPOKEN_DIGITS",
    "spoken_amount",
    "spoken_digits",
    "to_spoken",
]

#: Digit runs shorter than this are left unchanged; four-digit years must survive.
MIN_SPOKEN_DIGITS = 6

#: Digits are spoken in groups of this size, separated by a space the TTS pauses on.
_GROUP_SIZE = 4

#: Amounts above this are left unchanged rather than risking a wrong reading.
_MAX_SPOKEN_AMOUNT = 10**12

_DIGITS = "零一二三四五六七八九"
#: On a telephony channel "一" and "七" are easily confused, so phone numbers use "幺".
_PHONE_ONE = "幺"

_UNITS = ("", "十", "百", "千")
_SECTIONS = ("", "万", "亿")


def _digit_char(digit: str, *, phone: bool) -> str:
    if phone and digit == "1":
        return _PHONE_ONE
    return _DIGITS[int(digit)]


def spoken_digits(digits: str, *, phone: bool | None = None) -> str:
    """Read `digits` one by one, in groups of four separated by spaces.

    `digits` may already contain single spaces; existing groups are then preserved
    instead of being regrouped. `phone` defaults to autodetection (an 11-digit run
    starting with ``1``).
    """
    groups = digits.split(" ") if " " in digits else None
    bare = digits.replace(" ", "")
    if not bare.isdigit():
        return digits
    if phone is None:
        phone = len(bare) == 11 and bare.startswith("1")
    if groups is None:
        groups = [bare[i : i + _GROUP_SIZE] for i in range(0, len(bare), _GROUP_SIZE)]
    return " ".join(
        "".join(_digit_char(d, phone=phone) for d in group) for group in groups if group
    )


def _section_to_chinese(value: int) -> str:
    """Read a value below 10000 with its 十 / 百 / 千 units."""
    out: list[str] = []
    text = str(value)
    length = len(text)
    pending_zero = False
    for index, char in enumerate(text):
        digit = int(char)
        power = length - index - 1
        if digit == 0:
            pending_zero = bool(out)
            continue
        if pending_zero:
            out.append(_DIGITS[0])
            pending_zero = False
        out.append(_DIGITS[digit])
        out.append(_UNITS[power])
    return "".join(out)


def _int_to_chinese(value: int) -> str:
    """Read a non-negative integer below 10^12 as a Chinese numeral."""
    if value == 0:
        return _DIGITS[0]
    sections: list[int] = []
    remainder = value
    while remainder:
        sections.append(remainder % 10000)
        remainder //= 10000
    parts: list[str] = []
    for index in range(len(sections) - 1, -1, -1):
        section = sections[index]
        if section == 0:
            # A zero section becomes a single 零 separator, never two in a row.
            if parts and not parts[-1].endswith(_DIGITS[0]):
                parts.append(_DIGITS[0])
            continue
        text = _section_to_chinese(section)
        # Inside a lower section a leading zero must be voiced: 100200 -> 十万零二百.
        if parts and section < 1000 and not parts[-1].endswith(_DIGITS[0]):
            parts.append(_DIGITS[0])
        parts.append(text + _SECTIONS[index])
    # A zero section at the end contributed a separator with nothing after it.
    out = "".join(parts).rstrip(_DIGITS[0])
    # Mandarin drops the leading 一 of a bare 十几.
    if out.startswith("一十"):
        out = out[1:]
    return out


def spoken_amount(integer_part: str, decimal_part: str | None) -> str | None:
    """Read a currency amount, or return None when it should be left unchanged."""
    digits = integer_part.replace(",", "")
    if not digits.isdigit():
        return None
    value = int(digits)
    if value >= _MAX_SPOKEN_AMOUNT:
        return None
    out = _int_to_chinese(value) + "元"
    if not decimal_part:
        return out
    cents = (decimal_part + "0")[:2]
    jiao, fen = int(cents[0]), int(cents[1])
    if jiao == 0 and fen == 0:
        return out
    if fen == 0:
        return out + _DIGITS[jiao] + "角"
    if jiao == 0:
        return out + _DIGITS[0] + _DIGITS[fen] + "分"
    return out + _DIGITS[jiao] + "角" + _DIGITS[fen] + "分"


# A thousands-grouped or plain integer, with at most two decimals.
_NUMBER = r"(?P<int>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<dec>\d{1,2}))?"
# Digit runs joined only by single spaces; anything else (-, /, :, .) ends the run.
_RUN = r"\d+(?: \d+)*"

_PATTERN = re.compile(
    # Currency first, so that its digits are never taken by the digit-run branch.
    rf"(?P<prefix>[¥￥]\s?){_NUMBER}"
    rf"|{_NUMBER.replace('int', 'int2').replace('dec', 'dec2')}(?P<suffix>\s?元)"
    rf"|(?P<run>{_RUN})"
)


def _replace(match: re.Match[str]) -> str:
    if match.group("prefix") is not None:
        spoken = spoken_amount(match.group("int"), match.group("dec"))
        return spoken if spoken is not None else match.group(0)
    if match.group("suffix") is not None:
        spoken = spoken_amount(match.group("int2"), match.group("dec2"))
        return spoken if spoken is not None else match.group(0)
    run = match.group("run")
    if len(run.replace(" ", "")) < MIN_SPOKEN_DIGITS:
        return run
    return spoken_digits(run)


def to_spoken(text: str) -> str:
    """Rewrite amounts and long digit strings in `text` into spoken form."""
    if not text:
        return text
    return _PATTERN.sub(_replace, text)
