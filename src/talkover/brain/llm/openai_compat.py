"""`BrainLLM` over an OpenAI-compatible Chat Completions API.

The target is a self-hosted server (vLLM, llama.cpp, LM Studio, Ollama) as described in
DESIGN.md 6.5, but any service that speaks `POST {base}/v1/chat/completions` works.
Differences from the Anthropic client that this module has to bridge:

* the system prompt is a `role="system"` message, prepended here;
* tool specs are wrapped in `{"type": "function", "function": {...}}`;
* tool calls come back with `arguments` as a JSON **string**, not an object;
* tool results are their own `role="tool"` messages, not blocks inside a user message;
* `finish_reason` uses a different vocabulary, mapped back onto the neutral one;
* authentication is an optional `Authorization: Bearer` header — local servers usually
  run without a key, so an empty `api_key` sends no header at all.

`is_error` on a tool result has no wire representation here; the content is prefixed
with `Error: ` so the model still sees that the lookup failed.

Implemented directly on `httpx.AsyncClient`: the `openai` SDK is not a declared
dependency of this project and the request shape does not justify adding one.
"""

from __future__ import annotations

import json
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

__all__ = ["OpenAICompatLLM"]

_PROVIDER = "openai_compat"

# OpenAI finish reasons mapped onto the neutral (Anthropic) vocabulary.
_FINISH_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "function_call": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


class OpenAICompatLLM:
    """Non-streaming Chat Completions client."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str = "",
        model: str = "",
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
        self._url = join_url(base_url, "v1", "chat/completions")
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
        headers = {"content-type": "application/json"}
        if self._api_key:
            headers["authorization"] = f"Bearer {self._api_key}"
        return headers

    def build_body(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpecLike | Mapping[str, Any]] = (),
        *,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Serialise a request exactly as it goes on the wire (also used by the tests)."""
        wire: list[dict[str, Any]] = []
        if system:
            wire.append({"role": "system", "content": system})
        for message in messages:
            wire.extend(_encode_message(message))
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": wire,
        }
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
    return {
        "type": "function",
        "function": {"name": name, "description": description, "parameters": parameters},
    }


def _encode_message(message: Message) -> list[dict[str, Any]]:
    """Expand one neutral message into one or more Chat Completions messages.

    Tool results have to leave the user message and become `role="tool"` messages of
    their own, which must precede any further user text.
    """
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    out: list[dict[str, Any]] = []

    for part in message.content:
        if isinstance(part, TextPart):
            text_parts.append(part.text)
        elif isinstance(part, ToolCallPart):
            tool_calls.append(
                {
                    "id": part.id,
                    "type": "function",
                    "function": {
                        "name": part.name,
                        "arguments": json.dumps(part.arguments, ensure_ascii=False),
                    },
                }
            )
        elif isinstance(part, ToolResultPart):
            content = f"Error: {part.content}" if part.is_error else part.content
            out.append({"role": "tool", "tool_call_id": part.tool_call_id, "content": content})
        else:
            raise LLMError(f"{_PROVIDER}: unsupported content part {type(part).__name__}")

    text = "".join(text_parts)
    if message.role == "assistant":
        if text or tool_calls:
            entry: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            out.append(entry)
    elif text or not out:
        # A user turn with neither text nor tool results still has to appear on the wire.
        out.append({"role": "user", "content": text})
    return out


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _parse_response(payload: Any) -> LLMResponse:
    if not isinstance(payload, Mapping):
        raise LLMError(f"{_PROVIDER}: response body must be a JSON object, got {payload!r}")
    if "error" in payload and "choices" not in payload:
        error = payload["error"]
        detail = error.get("message") if isinstance(error, Mapping) else error
        raise LLMError(f"{_PROVIDER}: {detail}")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMError(f"{_PROVIDER}: response has no choices: {payload!r}")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise LLMError(f"{_PROVIDER}: choice must be an object, got {choice!r}")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise LLMError(f"{_PROVIDER}: choice is missing a 'message' object: {choice!r}")

    parts: list[ContentPart] = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append(TextPart(content))
    elif isinstance(content, list):
        # Some servers mirror the multimodal content-array form.
        for block in content:
            if isinstance(block, Mapping) and block.get("type") == "text":
                parts.append(TextPart(str(block.get("text", ""))))

    for call in message.get("tool_calls") or []:
        if not isinstance(call, Mapping):
            raise LLMError(f"{_PROVIDER}: tool call must be an object, got {call!r}")
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise LLMError(f"{_PROVIDER}: tool call is missing a 'function' object: {call!r}")
        name = str(function.get("name", ""))
        parts.append(
            ToolCallPart(
                id=str(call.get("id", "")),
                name=name,
                arguments=decode_arguments(function.get("arguments"), name, _PROVIDER),
            )
        )

    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason = _FINISH_REASONS.get(str(finish_reason), str(finish_reason))

    raw_usage = payload.get("usage")
    usage = Usage()
    if isinstance(raw_usage, Mapping):
        usage = Usage(
            input_tokens=int(raw_usage.get("prompt_tokens") or 0),
            output_tokens=int(raw_usage.get("completion_tokens") or 0),
        )
    return LLMResponse(parts=tuple(parts), stop_reason=stop_reason, usage=usage)
