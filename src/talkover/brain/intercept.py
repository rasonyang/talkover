"""Pre-interception for transfer to human, using pure keyword-list rule matching.

Running the whole Brain loop costs two to three seconds (DESIGN.md 6.4). When the trusted
text of a `task_start` already contains one of `brain.transfer_keywords`, the outcome is
known without a model: hand the call to a human agent. This module is that rule, and
nothing else -- it performs no I/O, holds no session state and never imports
`gander_runtime`, so it is testable on its own. `talkover.brain.provider` applies it on the
task-start path, where a hit opens the run without entering the loop
(`open_run(autostart=False)`) and raises the call through `BusinessRun.emit_client_tool`.

Matching is substring matching on a normalised form, so the punctuation and spacing an ASR
transcript happens to carry never decide the outcome:

* NFKC folds full-width forms onto ASCII, so "转人工！" and "转人工!" normalise alike;
* case is folded, so an ASCII keyword matches in any case;
* whitespace, control characters, punctuation and symbols are dropped.

Normalisation is applied to the keywords too, and a keyword that normalises to nothing is
dropped rather than allowed to match every utterance.

The keywords are runtime data and stay in the caller's language (Chinese in both example
configs). Everything else here is English.

The emitted call carries only `department`: `reason` would become the interaction's
`prompt`, which a client without tool support speaks back to the customer, and an English
sentence in the middle of a Chinese call is worse than the provider's own fallback
(`TRANSFER_PROMPT`). The matched keyword is reported on :class:`TransferMatch` for logging
instead of being smuggled into the tool arguments, whose schema is closed
(`additionalProperties: False`).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME
from talkover.config import BrainConfig

__all__ = [
    "INTERCEPT_DEPARTMENT",
    "TRANSFER_TO_HUMAN_NAME",
    "TransferInterceptor",
    "TransferMatch",
    "normalize",
]

#: `transfer_to_human.department` for a rule hit. The rule knows the customer wants a
#: person; it does not know which desk, so it uses the schema's neutral value and lets the
#: client's routing decide.
INTERCEPT_DEPARTMENT = "general"

#: Unicode general-category initials dropped before matching: separators, control/format
#: characters, punctuation and symbols.
_NOISE_CATEGORIES = frozenset({"Z", "C", "P", "S"})


def _is_noise(char: str) -> bool:
    return char.isspace() or unicodedata.category(char)[0] in _NOISE_CATEGORIES


def normalize(text: str) -> str:
    """Fold `text` to the form both sides of the substring test are compared in.

    NFKC first (full-width -> ASCII, compatibility forms unified), then case folding, then
    every whitespace, control, punctuation and symbol character removed.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(char for char in folded if not _is_noise(char))


@dataclass(frozen=True, slots=True)
class TransferMatch:
    """One keyword hit, kept for logging and for the tests to assert against."""

    #: The keyword as it is written in the configuration.
    keyword: str
    #: The normalised keyword that was found in the normalised text.
    normalized_keyword: str
    #: The utterance as it arrived.
    text: str
    #: The normalised utterance the keyword was found in.
    normalized_text: str

    @property
    def department(self) -> str:
        return INTERCEPT_DEPARTMENT


class TransferInterceptor:
    """Decides whether a trusted user turn already asks for a human agent.

    Pure and reusable: build one per provider from `brain.transfer_keywords` and call
    :meth:`match` per `task_start`. An interceptor with no usable keyword is falsy and
    matches nothing, which is the default configuration.
    """

    __slots__ = ("_department", "_keywords", "_normalized")

    def __init__(
        self, keywords: Iterable[str] = (), *, department: str = INTERCEPT_DEPARTMENT
    ) -> None:
        kept: list[str] = []
        normalized: list[str] = []
        seen: set[str] = set()
        for keyword in keywords:
            folded = normalize(keyword)
            # A keyword that is only punctuation would match every utterance.
            if not folded or folded in seen:
                continue
            seen.add(folded)
            kept.append(keyword)
            normalized.append(folded)
        self._keywords: tuple[str, ...] = tuple(kept)
        self._normalized: tuple[str, ...] = tuple(normalized)
        self._department = department

    @classmethod
    def from_config(cls, config: BrainConfig) -> TransferInterceptor:
        return cls(config.transfer_keywords)

    @property
    def keywords(self) -> tuple[str, ...]:
        """The configured keywords that survived normalisation, in order."""
        return self._keywords

    @property
    def department(self) -> str:
        return self._department

    def __bool__(self) -> bool:
        return bool(self._normalized)

    def __len__(self) -> int:
        return len(self._normalized)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keywords)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"TransferInterceptor(keywords={self._keywords!r})"

    def match(self, text: str) -> TransferMatch | None:
        """Return the first keyword contained in `text`, or None.

        The configuration order is the priority order, so a deployment can list its most
        specific phrase first.
        """
        if not self._normalized or not text:
            return None
        normalized_text = normalize(text)
        if not normalized_text:
            return None
        for keyword, folded in zip(self._keywords, self._normalized, strict=True):
            if folded in normalized_text:
                return TransferMatch(
                    keyword=keyword,
                    normalized_keyword=folded,
                    text=text,
                    normalized_text=normalized_text,
                )
        return None

    def matches(self, text: str) -> bool:
        """`match(text) is not None`, for callers that only need the decision."""
        return self.match(text) is not None

    def transfer_arguments(self, match: TransferMatch | None = None) -> dict[str, Any]:
        """The `transfer_to_human` arguments for a hit (see the module docstring)."""
        department = match.department if match is not None else self._department
        return {"department": department}
