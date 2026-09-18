"""A scripted `BrainLLM` for tests that exercise the Brain loop without a network call.

`FakeLLM` replays a fixed list of `LLMResponse` objects and records every call it was
given, so a test can assert both what the loop asked for (messages, tools, system
prompt) and what it did with each answer. `error` fails every call and `errors` scripts
one exception per call, which is how "fails once, then answers" is expressed. T3.3 uses it
for the session loop.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from talkover.brain.llm import (
    LLMResponse,
    Message,
    TextPart,
    ToolCallPart,
    ToolSpecLike,
    Usage,
)

__all__ = ["FakeLLM", "RecordedCall", "ScriptExhausted", "text_response", "tool_call_response"]


class ScriptExhausted(AssertionError):
    """Raised when the loop asks for one more completion than the script provides."""


@dataclass(frozen=True, slots=True)
class RecordedCall:
    """One `complete()` invocation, captured verbatim."""

    messages: tuple[Message, ...]
    tools: tuple[Any, ...]
    system: str | None

    @property
    def tool_names(self) -> tuple[str, ...]:
        names = []
        for tool in self.tools:
            names.append(tool["name"] if isinstance(tool, Mapping) else tool.name)
        return tuple(names)

    @property
    def last_message(self) -> Message:
        return self.messages[-1]


@dataclass
class FakeLLM:
    """Replays `script` in order; records each call in `calls`.

    Set `error` to raise the same exception on every call. Set `errors` to script them per
    call instead: call number *i* raises `errors[i]` when it is not `None`, and calls past
    the end of the list answer normally. Only the calls that answer consume `script`, so
    "fails once, then answers" is one entry in each list.
    """

    script: list[LLMResponse] = field(default_factory=list)
    calls: list[RecordedCall] = field(default_factory=list)
    error: Exception | None = None
    errors: list[Exception | None] = field(default_factory=list)
    closed: bool = False
    served: int = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpecLike | Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> LLMResponse:
        self.calls.append(RecordedCall(messages=tuple(messages), tools=tuple(tools), system=system))
        if self.error is not None:
            raise self.error
        index = len(self.calls) - 1
        if index < len(self.errors) and self.errors[index] is not None:
            raise self.errors[index]
        if self.served >= len(self.script):
            raise ScriptExhausted(
                f"FakeLLM script has {len(self.script)} response(s) but complete() was called "
                f"{len(self.calls)} time(s); the last request had "
                f"{len(messages)} message(s) and tools {self.calls[-1].tool_names}"
            )
        response = self.script[self.served]
        self.served += 1
        return response

    async def aclose(self) -> None:
        self.closed = True

    @property
    def call_count(self) -> int:
        return len(self.calls)


def text_response(text: str, *, stop_reason: str = "end_turn") -> LLMResponse:
    """A plain assistant reply."""
    return LLMResponse(parts=(TextPart(text),), stop_reason=stop_reason, usage=Usage(10, 5))


def tool_call_response(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    call_id: str = "call_1",
    text: str = "",
) -> LLMResponse:
    """An assistant reply that requests one tool call, optionally with leading text."""
    parts: list[Any] = []
    if text:
        parts.append(TextPart(text))
    parts.append(ToolCallPart(id=call_id, name=name, arguments=dict(arguments or {})))
    return LLMResponse(parts=tuple(parts), stop_reason="tool_use", usage=Usage(12, 8))
