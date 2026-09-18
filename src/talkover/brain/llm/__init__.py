"""`BrainLLM` protocol, provider-neutral message types and implementation selection.

DESIGN.md 6.5: the business LLM is abstracted behind a small protocol so that the
Brain session loop never sees a provider wire format. Two implementations ship in the
first version: `anthropic.py` (DeepSeek's Anthropic-compatible endpoint by default, and
Anthropic's own Messages API unchanged) and `openai_compat.py` (vLLM and local servers).

Conventions shared by every implementation:

* The system prompt is a separate `system` argument of `complete()`, never a message
  with `role="system"`. `Message.role` is only `user` or `assistant`.
* Tool results are `ToolResultPart`s inside a `user` message; each client translates
  them into whatever its wire format needs.
* `LLMResponse.stop_reason` uses the Anthropic vocabulary (`end_turn`, `tool_use`,
  `max_tokens`, `stop_sequence`); the OpenAI client maps its finish reasons onto it.
* Every failure is an `LLMError` carrying a `retryable` flag.

Streaming is not implemented: the Brain loop needs whole tool calls before it can act,
and `share(text)` is emitted per round rather than per token.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_TIMEOUT_SEC",
    "BrainLLM",
    "ContentPart",
    "LLMError",
    "LLMResponse",
    "Message",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "ToolSpecLike",
    "Usage",
    "create_llm",
    "decode_arguments",
    "decode_json",
    "join_url",
    "raise_for_status",
    "tool_fields",
]

DEFAULT_TIMEOUT_SEC = 30.0
DEFAULT_MAX_TOKENS = 1024

Role = Literal["user", "assistant"]
StopReason = Literal["end_turn", "tool_use", "max_tokens", "stop_sequence"]


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class LLMError(RuntimeError):
    """Any failure of an LLM call: transport, HTTP status, or unusable response body.

    `retryable` is True when repeating the same request has a reasonable chance of
    succeeding: timeouts, connection errors, 408, 429 and 5xx, and also a malformed
    tool-call argument string, which is a sampling accident rather than a bug in the
    request. Authentication and request-shape errors are not retryable.
    """

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        base = super().__str__()
        if self.status_code is None:
            return base
        return f"{base} (HTTP {self.status_code})"


# --------------------------------------------------------------------------------------
# Content parts and messages
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextPart:
    """Plain text written by the user or the model."""

    text: str


@dataclass(frozen=True, slots=True)
class ToolCallPart:
    """A tool invocation requested by the model.

    `arguments` is always a decoded JSON object; the OpenAI client decodes the
    JSON-string form on the way in.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultPart:
    """The outcome of a tool call, sent back to the model in a `user` message."""

    tool_call_id: str
    content: str
    is_error: bool = False


ContentPart = TextPart | ToolCallPart | ToolResultPart


@dataclass(frozen=True, slots=True)
class Message:
    """One conversation turn. The system prompt is not a message; see the module docstring."""

    role: Role
    content: tuple[ContentPart, ...]

    @staticmethod
    def user(*parts: ContentPart | str) -> Message:
        return Message("user", tuple(_coerce_part(p) for p in parts))

    @staticmethod
    def assistant(*parts: ContentPart | str) -> Message:
        return Message("assistant", tuple(_coerce_part(p) for p in parts))

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.content if isinstance(p, TextPart))


def _coerce_part(part: ContentPart | str) -> ContentPart:
    return TextPart(part) if isinstance(part, str) else part


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts reported by the provider; zero when it reports nothing."""

    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """One assistant turn: its content parts, why it stopped, and its token usage."""

    parts: tuple[ContentPart, ...] = ()
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> tuple[ToolCallPart, ...]:
        return tuple(p for p in self.parts if isinstance(p, ToolCallPart))

    def as_message(self) -> Message:
        """The assistant message to append to the conversation before the next round."""
        return Message("assistant", self.parts)


# --------------------------------------------------------------------------------------
# Tool specs
# --------------------------------------------------------------------------------------


@runtime_checkable
class ToolSpecLike(Protocol):
    """Anything with a name, a description and a JSON-schema parameter object.

    `talkover.brain.tools` defines the concrete `ToolSpec`; this module deliberately does
    not import it, so a plain mapping with the same three keys works just as well.
    """

    name: str
    description: str
    parameters: Mapping[str, Any]


_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def tool_fields(tool: ToolSpecLike | Mapping[str, Any]) -> tuple[str, str, dict[str, Any]]:
    """Read `name` / `description` / `parameters` from a mapping or an object."""
    if isinstance(tool, Mapping):
        try:
            name = tool["name"]
        except KeyError as exc:
            raise LLMError(f"tool spec is missing 'name': {tool!r}") from exc
        description = tool.get("description", "")
        parameters = tool.get("parameters")
    else:
        try:
            name = tool.name
        except AttributeError as exc:
            raise LLMError(f"tool spec is missing 'name': {tool!r}") from exc
        description = getattr(tool, "description", "")
        parameters = getattr(tool, "parameters", None)
    if parameters is None:
        parameters = _EMPTY_SCHEMA
    if not isinstance(parameters, Mapping):
        raise LLMError(f"tool {name!r}: 'parameters' must be a JSON schema object")
    return str(name), str(description or ""), dict(parameters)


# --------------------------------------------------------------------------------------
# The protocol
# --------------------------------------------------------------------------------------


class BrainLLM(Protocol):
    """The business LLM as the Brain session sees it."""

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpecLike | Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> LLMResponse:
        """Run one non-streaming completion.

        Raises `LLMError` on transport failure, an error status, or an unusable body.
        """
        ...

    async def aclose(self) -> None:
        """Release the underlying HTTP connections."""
        ...


# --------------------------------------------------------------------------------------
# Shared helpers for the HTTP clients
# --------------------------------------------------------------------------------------


def join_url(base_url: str, version: str, endpoint: str) -> str:
    """Build a request URL tolerant of how the configured `base_url` is written.

    `endpoint` is the final path segment (`messages`, `chat/completions`) and `version`
    the API version prefix (`v1`). A base URL that already names the endpoint is used
    as-is, one that already ends in the version prefix only gets the endpoint appended,
    and anything else gets both. Trailing slashes never matter.

        https://api.deepseek.com/anthropic  -> https://api.deepseek.com/anthropic/v1/messages
        http://localhost:8000/v1/           -> http://localhost:8000/v1/chat/completions
    """
    base = base_url.strip().rstrip("/")
    if not base:
        raise LLMError("brain.llm.base_url is empty")
    if base.endswith("/" + endpoint):
        return base
    if base.rsplit("/", 1)[-1] == version:
        return f"{base}/{endpoint}"
    return f"{base}/{version}/{endpoint}"


def _retryable_status(status_code: int) -> bool:
    return status_code in (408, 409, 425, 429) or status_code >= 500


def raise_for_status(status_code: int, body: str, provider: str) -> None:
    """Map an error status onto a typed `LLMError`, quoting a trimmed response body."""
    if status_code < 400:
        return
    snippet = body.strip().replace("\n", " ")
    if len(snippet) > 500:
        snippet = snippet[:500] + "..."
    if status_code in (401, 403):
        detail = "authentication failed; check brain.llm.api_key"
    elif status_code == 404:
        detail = "endpoint not found; check brain.llm.base_url"
    elif status_code == 429:
        detail = "rate limited"
    elif status_code >= 500:
        detail = "upstream server error"
    else:
        detail = "request rejected"
    raise LLMError(
        f"{provider}: {detail}: {snippet}",
        retryable=_retryable_status(status_code),
        status_code=status_code,
        body=body,
    )


def decode_json(body: str, provider: str) -> Any:
    """Parse a response body, turning a non-JSON payload into an `LLMError`."""
    try:
        return json.loads(body)
    except ValueError as exc:
        snippet = body[:200]
        raise LLMError(f"{provider}: response is not valid JSON: {snippet!r}", body=body) from exc


def decode_arguments(raw: Any, tool_name: str, provider: str) -> dict[str, Any]:
    """Normalise a tool call's arguments into a JSON object.

    Providers send either an object (Anthropic `tool_use.input`) or a JSON string
    (OpenAI `tool_calls[].function.arguments`). A truncated or otherwise malformed
    string is retryable: another sample usually produces valid JSON.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise LLMError(
                f"{provider}: tool {tool_name!r} returned malformed JSON arguments: {raw!r}",
                retryable=True,
            ) from exc
        if not isinstance(parsed, Mapping):
            raise LLMError(
                f"{provider}: tool {tool_name!r} arguments must be a JSON object, got {raw!r}",
                retryable=True,
            )
        return dict(parsed)
    raise LLMError(
        f"{provider}: tool {tool_name!r} arguments must be a JSON object, got {type(raw).__name__}",
        retryable=True,
    )


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def create_llm(config: Any, **kwargs: Any) -> BrainLLM:
    """Build the `BrainLLM` named by `config.kind` (a `talkover.config.LLMConfig`).

    Extra keyword arguments (`timeout_sec`, `max_tokens`, `client`) are passed through
    to the client; `LLMConfig` deliberately does not carry them.
    """
    # Imported lazily: both modules import the types defined above.
    from talkover.brain.llm.anthropic import AnthropicLLM
    from talkover.brain.llm.openai_compat import OpenAICompatLLM

    kind = getattr(config, "kind", None)
    if kind == "anthropic":
        return AnthropicLLM(
            base_url=config.base_url, api_key=config.api_key, model=config.model, **kwargs
        )
    if kind == "openai_compat":
        return OpenAICompatLLM(
            base_url=config.base_url, api_key=config.api_key, model=config.model, **kwargs
        )
    raise LLMError(f"brain.llm.kind: expected 'anthropic' or 'openai_compat', got {kind!r}")
