"""`BrainLLM` over the Anthropic Messages API.

The default target is DeepSeek's Anthropic-compatible endpoint (DESIGN.md 6.5:
`base_url: https://api.deepseek.com/anthropic`, `model: deepseek-flash`, key from
`DEEPSEEK_API_KEY`). The same code runs unchanged against `https://api.anthropic.com`:
both speak `POST {base}/v1/messages` with an `x-api-key` header, an `anthropic-version`
header, a required `max_tokens`, and `tool_use` / `tool_result` content blocks.

The `anthropic` SDK is not a dependency of this project: the request shape is a handful
of JSON fields, and going straight to `httpx.AsyncClient` keeps the client testable with
`httpx.MockTransport` and free of a second HTTP stack.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import httpx

from talkover.brain.llm import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_SEC,
    ContentPart,
    LLMError,
    LLMResponse,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    ToolSpecLike,
    Usage,
    decode_arguments,
    decode_json,
    join_url,
    raise_for_status,
    tool_fields,
)

__all__ = ["ANTHROPIC_VERSION", "AnthropicLLM"]

ANTHROPIC_VERSION = "2023-06-01"
_PROVIDER = "anthropic"

# Anthropic stop reasons already are the neutral vocabulary; `tool_use` is the one that
# drives the Brain loop.
_KNOWN_STOP_REASONS = ("end_turn", "tool_use", "max_tokens", "stop_sequence")


class AnthropicLLM:
    """Non-streaming Messages API client."""

    def __init__(
        self,
        *,
        base_url: str = "https://api.deepseek.com/anthropic",
        api_key: str = "",
        model: str = "deepseek-flash",
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.timeout_sec = timeout_sec
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._api_key = api_key
        self._url = join_url(base_url, "v1", "messages")
        self._client = client
        self._owns_client = client is None

    # -- lifecycle ---------------------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout_sec)
        return self._client

    async def aclose(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- request -----------------------------------------------------------------------

    def headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": ANTHROPIC_VERSION,
        }
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def build_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpecLike | Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Serialise a request exactly as it goes on the wire (also used by the tests)."""
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [_encode_message(m) for m in messages],
        }
        if system:
            body["system"] = system
        if tools:
            body["tools"] = [_encode_tool(t) for t in tools]
        if self.temperature is not None:
            body["temperature"] = self.temperature
        return body

    async def complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpecLike | Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> LLMResponse:
        body = self.build_body(messages, tools, system=system)
        client = self._ensure_client()
        try:
            response = await client.post(
                self._url, json=body, headers=self.headers(), timeout=self.timeout_sec
            )
        except httpx.TimeoutException as exc:
            raise LLMError(
                f"{_PROVIDER}: request to {self._url} timed out after {self.timeout_sec}s",
                retryable=True,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"{_PROVIDER}: request to {self._url} failed: {exc}", retryable=True
            ) from exc

        raise_for_status(response.status_code, response.text, _PROVIDER)
        return _parse_response(decode_json(response.text, _PROVIDER))


# --------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------


def _encode_tool(tool: ToolSpecLike | Mapping[str, Any]) -> dict[str, Any]:
    name, description, parameters = tool_fields(tool)
    return {"name": name, "description": description, "input_schema": parameters}


def _encode_part(part: ContentPart) -> dict[str, Any]:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    if isinstance(part, ToolCallPart):
        return {"type": "tool_use", "id": part.id, "name": part.name, "input": part.arguments}
    if isinstance(part, ToolResultPart):
        block: dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": part.tool_call_id,
            "content": part.content,
        }
        if part.is_error:
            block["is_error"] = True
        return block
    raise LLMError(f"{_PROVIDER}: unsupported content part {type(part).__name__}")


def _encode_message(message: Message) -> dict[str, Any]:
    return {"role": message.role, "content": [_encode_part(p) for p in message.content]}


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _parse_response(payload: Any) -> LLMResponse:
    if not isinstance(payload, Mapping):
        raise LLMError(f"{_PROVIDER}: response body must be a JSON object, got {payload!r}")
    if payload.get("type") == "error":
        error = payload.get("error")
        detail = error.get("message") if isinstance(error, Mapping) else error
        raise LLMError(f"{_PROVIDER}: {detail}")

    content = payload.get("content")
    if not isinstance(content, list):
        raise LLMError(f"{_PROVIDER}: response is missing a 'content' array: {payload!r}")

    parts: list[ContentPart] = []
    for block in content:
        if not isinstance(block, Mapping):
            raise LLMError(f"{_PROVIDER}: content block must be an object, got {block!r}")
        kind = block.get("type")
        if kind == "text":
            parts.append(TextPart(str(block.get("text", ""))))
        elif kind == "tool_use":
            name = str(block.get("name", ""))
            parts.append(
                ToolCallPart(
                    id=str(block.get("id", "")),
                    name=name,
                    arguments=decode_arguments(block.get("input"), name, _PROVIDER),
                )
            )
        elif kind == "thinking":
            # Extended thinking blocks carry no instruction for the Brain loop.
            continue
        else:
            raise LLMError(f"{_PROVIDER}: unsupported content block type {kind!r}")

    stop_reason = payload.get("stop_reason") or "end_turn"
    if stop_reason not in _KNOWN_STOP_REASONS:
        stop_reason = str(stop_reason)

    raw_usage = payload.get("usage")
    usage = Usage()
    if isinstance(raw_usage, Mapping):
        usage = Usage(
            input_tokens=int(raw_usage.get("input_tokens") or 0),
            output_tokens=int(raw_usage.get("output_tokens") or 0),
        )
    return LLMResponse(parts=tuple(parts), stop_reason=str(stop_reason), usage=usage)
