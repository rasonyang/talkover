"""Unit tests for the `BrainLLM` clients (T3.2).

Every case runs against `httpx.MockTransport` with a hand-written response body in the
shape the real service returns, so the tests pin both the request Talkover sends and the
parsing of what comes back. The only test that touches the network is the opt-in DeepSeek
smoke at the bottom, which is skipped unless `DEEPSEEK_API_KEY` is set.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
import pytest

from talkover.brain.llm import (
    LLMError,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    create_llm,
    join_url,
)
from talkover.brain.llm.anthropic import ANTHROPIC_VERSION, AnthropicLLM
from talkover.brain.llm.openai_compat import OpenAICompatLLM
from talkover.config import LLMConfig

from .fake_llm import FakeLLM, ScriptExhausted, text_response, tool_call_response

# --------------------------------------------------------------------------------------
# Recorded response bodies
# --------------------------------------------------------------------------------------

ANTHROPIC_TEXT = {
    "id": "msg_01XFDUDYJgAACzvnptvVoYEL",
    "type": "message",
    "role": "assistant",
    "model": "deepseek-flash",
    "content": [{"type": "text", "text": "您的订单预计明天送达。"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 412, "output_tokens": 17},
}

ANTHROPIC_TOOL_USE = {
    "id": "msg_01Aq9w938a2furi1MmjBeit9",
    "type": "message",
    "role": "assistant",
    "model": "deepseek-flash",
    "content": [
        {"type": "text", "text": "好的，我帮您查一下。"},
        {
            "type": "tool_use",
            "id": "toolu_01A09q90qw90lq917835lq9",
            "name": "query_order",
            "input": {"order_id": "SO20260917001"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {"input_tokens": 486, "output_tokens": 64},
}

OPENAI_TEXT = {
    "id": "chatcmpl-9x1t0Wc7hXk2",
    "object": "chat.completion",
    "created": 1789600000,
    "model": "qwen2.5-7b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "您的订单预计明天送达。",
                "tool_calls": None,
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 388, "completion_tokens": 15, "total_tokens": 403},
}

OPENAI_TOOL_CALL = {
    "id": "chatcmpl-9x1t3Kd2mQr8",
    "object": "chat.completion",
    "created": 1789600042,
    "model": "qwen2.5-7b-instruct",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_9f2b1c",
                        "type": "function",
                        "function": {
                            "name": "query_order",
                            "arguments": '{"order_id": "SO20260917001"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 401, "completion_tokens": 28, "total_tokens": 429},
}

QUERY_ORDER_TOOL = {
    "name": "query_order",
    "description": "Look up an order by its id or the customer's phone number.",
    "parameters": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string", "description": "The order number."},
            "phone": {"type": "string", "description": "The customer's phone number."},
        },
        "required": [],
    },
}


class _ToolSpecObject:
    """Stand-in for `talkover.brain.tools.ToolSpec`, which this module must not import."""

    def __init__(self, name: str, description: str, parameters: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _recording_client(
    payload: Any = None,
    *,
    status_code: int = 200,
    raises: Exception | None = None,
    text: str | None = None,
) -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    """An `AsyncClient` that records requests and replays one canned response."""
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if raises is not None:
            raise raises
        if text is not None:
            return httpx.Response(status_code, text=text, request=request)
        return httpx.Response(status_code, json=payload, request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content.decode())


CONVERSATION = [
    Message.user("我想查一下订单 SO20260917001"),
    Message.assistant(
        TextPart("好的，我帮您查一下。"),
        ToolCallPart(id="toolu_1", name="query_order", arguments={"order_id": "SO20260917001"}),
    ),
    Message.user(
        ToolResultPart(tool_call_id="toolu_1", content='{"status": "shipped"}'),
    ),
]


# --------------------------------------------------------------------------------------
# URL building
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("https://api.deepseek.com/anthropic", "https://api.deepseek.com/anthropic/v1/messages"),
        ("https://api.deepseek.com/anthropic/", "https://api.deepseek.com/anthropic/v1/messages"),
        ("https://api.anthropic.com", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/", "https://api.anthropic.com/v1/messages"),
        ("https://api.anthropic.com/v1/messages", "https://api.anthropic.com/v1/messages"),
    ],
)
def test_anthropic_url_variants(base_url: str, expected: str) -> None:
    assert AnthropicLLM(base_url=base_url)._url == expected


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://localhost:8000", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1", "http://localhost:8000/v1/chat/completions"),
        ("http://localhost:8000/v1/", "http://localhost:8000/v1/chat/completions"),
        (
            "http://localhost:8000/v1/chat/completions",
            "http://localhost:8000/v1/chat/completions",
        ),
    ],
)
def test_openai_url_variants(base_url: str, expected: str) -> None:
    assert OpenAICompatLLM(base_url=base_url)._url == expected


def test_join_url_rejects_empty_base() -> None:
    with pytest.raises(LLMError):
        join_url("   ", "v1", "messages")


# --------------------------------------------------------------------------------------
# Anthropic client
# --------------------------------------------------------------------------------------


async def test_anthropic_plain_text_reply() -> None:
    client, seen = _recording_client(ANTHROPIC_TEXT)
    llm = AnthropicLLM(api_key="sk-test", client=client, max_tokens=512)

    response = await llm.complete([Message.user("订单什么时候到？")], system="你是客服助手。")

    assert response.text == "您的订单预计明天送达。"
    assert response.tool_calls == ()
    assert response.stop_reason == "end_turn"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (412, 17)

    request = seen[0]
    assert str(request.url) == "https://api.deepseek.com/anthropic/v1/messages"
    assert request.headers["x-api-key"] == "sk-test"
    assert request.headers["anthropic-version"] == ANTHROPIC_VERSION
    assert request.headers["content-type"] == "application/json"
    assert _body(request) == {
        "model": "deepseek-flash",
        "max_tokens": 512,
        "system": "你是客服助手。",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "订单什么时候到？"}]}],
    }


async def test_anthropic_tool_call_and_tool_serialization() -> None:
    client, seen = _recording_client(ANTHROPIC_TOOL_USE)
    llm = AnthropicLLM(api_key="sk-test", client=client)

    response = await llm.complete([Message.user("查订单")], tools=[QUERY_ORDER_TOOL])

    assert response.stop_reason == "tool_use"
    assert response.text == "好的，我帮您查一下。"
    (call,) = response.tool_calls
    assert (call.id, call.name) == ("toolu_01A09q90qw90lq917835lq9", "query_order")
    assert call.arguments == {"order_id": "SO20260917001"}

    assert _body(seen[0])["tools"] == [
        {
            "name": "query_order",
            "description": QUERY_ORDER_TOOL["description"],
            "input_schema": QUERY_ORDER_TOOL["parameters"],
        }
    ]


async def test_anthropic_accepts_tool_spec_objects() -> None:
    client, seen = _recording_client(ANTHROPIC_TEXT)
    llm = AnthropicLLM(client=client)
    spec = _ToolSpecObject(
        QUERY_ORDER_TOOL["name"], QUERY_ORDER_TOOL["description"], QUERY_ORDER_TOOL["parameters"]
    )

    await llm.complete([Message.user("hi")], tools=[spec])

    assert _body(seen[0])["tools"][0]["name"] == "query_order"


async def test_anthropic_multi_turn_request_shape() -> None:
    client, seen = _recording_client(ANTHROPIC_TEXT)
    llm = AnthropicLLM(api_key="sk-test", client=client)

    await llm.complete(CONVERSATION, tools=[QUERY_ORDER_TOOL], system="你是客服助手。")

    assert _body(seen[0])["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "我想查一下订单 SO20260917001"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "好的，我帮您查一下。"},
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "query_order",
                    "input": {"order_id": "SO20260917001"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": '{"status": "shipped"}',
                }
            ],
        },
    ]


async def test_anthropic_tool_result_error_flag() -> None:
    client, seen = _recording_client(ANTHROPIC_TEXT)
    llm = AnthropicLLM(client=client)

    await llm.complete([Message.user(ToolResultPart("toolu_1", "upstream timeout", is_error=True))])

    block = _body(seen[0])["messages"][0]["content"][0]
    assert block["is_error"] is True


async def test_anthropic_without_api_key_sends_no_auth_header() -> None:
    client, seen = _recording_client(ANTHROPIC_TEXT)
    await AnthropicLLM(api_key="", client=client).complete([Message.user("hi")])
    assert "x-api-key" not in seen[0].headers


async def test_anthropic_skips_thinking_blocks() -> None:
    payload = dict(ANTHROPIC_TEXT)
    payload["content"] = [
        {"type": "thinking", "thinking": "internal", "signature": "abc"},
        {"type": "text", "text": "好的。"},
    ]
    client, _ = _recording_client(payload)
    response = await AnthropicLLM(client=client).complete([Message.user("hi")])
    assert response.text == "好的。"


# --------------------------------------------------------------------------------------
# OpenAI-compatible client
# --------------------------------------------------------------------------------------


async def test_openai_plain_text_reply() -> None:
    client, seen = _recording_client(OPENAI_TEXT)
    llm = OpenAICompatLLM(
        base_url="http://localhost:8000/v1",
        api_key="local-key",
        model="qwen2.5-7b-instruct",
        client=client,
        max_tokens=256,
    )

    response = await llm.complete([Message.user("订单什么时候到？")], system="你是客服助手。")

    assert response.text == "您的订单预计明天送达。"
    assert response.stop_reason == "end_turn"
    assert (response.usage.input_tokens, response.usage.output_tokens) == (388, 15)

    request = seen[0]
    assert str(request.url) == "http://localhost:8000/v1/chat/completions"
    assert request.headers["authorization"] == "Bearer local-key"
    assert _body(request) == {
        "model": "qwen2.5-7b-instruct",
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": "你是客服助手。"},
            {"role": "user", "content": "订单什么时候到？"},
        ],
    }


async def test_openai_tool_call_arguments_are_decoded() -> None:
    client, seen = _recording_client(OPENAI_TOOL_CALL)
    llm = OpenAICompatLLM(model="qwen2.5-7b-instruct", client=client)

    response = await llm.complete([Message.user("查订单")], tools=[QUERY_ORDER_TOOL])

    assert response.stop_reason == "tool_use"
    (call,) = response.tool_calls
    assert (call.id, call.name) == ("call_9f2b1c", "query_order")
    assert call.arguments == {"order_id": "SO20260917001"}

    assert _body(seen[0])["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "query_order",
                "description": QUERY_ORDER_TOOL["description"],
                "parameters": QUERY_ORDER_TOOL["parameters"],
            },
        }
    ]


async def test_openai_multi_turn_request_shape() -> None:
    client, seen = _recording_client(OPENAI_TEXT)
    llm = OpenAICompatLLM(model="qwen2.5-7b-instruct", client=client)

    await llm.complete(CONVERSATION, system="你是客服助手。")

    assert _body(seen[0])["messages"] == [
        {"role": "system", "content": "你是客服助手。"},
        {"role": "user", "content": "我想查一下订单 SO20260917001"},
        {
            "role": "assistant",
            "content": "好的，我帮您查一下。",
            "tool_calls": [
                {
                    "id": "toolu_1",
                    "type": "function",
                    "function": {
                        "name": "query_order",
                        "arguments": '{"order_id": "SO20260917001"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_1", "content": '{"status": "shipped"}'},
    ]


async def test_openai_tool_result_error_is_marked_in_content() -> None:
    client, seen = _recording_client(OPENAI_TEXT)
    llm = OpenAICompatLLM(model="m", client=client)

    await llm.complete([Message.user(ToolResultPart("call_1", "upstream timeout", is_error=True))])

    assert _body(seen[0])["messages"] == [
        {"role": "tool", "tool_call_id": "call_1", "content": "Error: upstream timeout"}
    ]


async def test_openai_without_api_key_sends_no_auth_header() -> None:
    client, seen = _recording_client(OPENAI_TEXT)
    await OpenAICompatLLM(api_key="", model="m", client=client).complete([Message.user("hi")])
    assert "authorization" not in seen[0].headers


async def test_openai_maps_length_finish_reason() -> None:
    payload = json.loads(json.dumps(OPENAI_TEXT))
    payload["choices"][0]["finish_reason"] = "length"
    client, _ = _recording_client(payload)
    response = await OpenAICompatLLM(model="m", client=client).complete([Message.user("hi")])
    assert response.stop_reason == "max_tokens"


# --------------------------------------------------------------------------------------
# Error mapping
# --------------------------------------------------------------------------------------


ANTHROPIC_401 = {
    "type": "error",
    "error": {"type": "authentication_error", "message": "invalid x-api-key"},
}
OPENAI_429 = {
    "error": {
        "message": "Rate limit reached for requests",
        "type": "rate_limit_error",
        "code": "rate_limit_exceeded",
    }
}


@pytest.mark.parametrize("factory", [AnthropicLLM, OpenAICompatLLM])
@pytest.mark.parametrize(
    ("status_code", "retryable"),
    [(401, False), (403, False), (404, False), (400, False), (429, True), (500, True), (503, True)],
)
async def test_http_error_status_mapping(factory: Any, status_code: int, retryable: bool) -> None:
    payload = ANTHROPIC_401 if factory is AnthropicLLM else OPENAI_429
    client, _ = _recording_client(payload, status_code=status_code)
    llm = factory(model="m", client=client)

    with pytest.raises(LLMError) as excinfo:
        await llm.complete([Message.user("hi")])

    assert excinfo.value.status_code == status_code
    assert excinfo.value.retryable is retryable
    assert excinfo.value.body


@pytest.mark.parametrize("factory", [AnthropicLLM, OpenAICompatLLM])
async def test_timeout_is_retryable_llm_error(factory: Any) -> None:
    client, _ = _recording_client(raises=httpx.ReadTimeout("timed out"))
    llm = factory(model="m", client=client, timeout_sec=1.5)

    with pytest.raises(LLMError) as excinfo:
        await llm.complete([Message.user("hi")])

    assert excinfo.value.retryable is True
    assert excinfo.value.status_code is None
    assert "timed out" in str(excinfo.value)


@pytest.mark.parametrize("factory", [AnthropicLLM, OpenAICompatLLM])
async def test_connection_error_is_retryable_llm_error(factory: Any) -> None:
    client, _ = _recording_client(raises=httpx.ConnectError("connection refused"))
    llm = factory(model="m", client=client)

    with pytest.raises(LLMError) as excinfo:
        await llm.complete([Message.user("hi")])

    assert excinfo.value.retryable is True


@pytest.mark.parametrize("factory", [AnthropicLLM, OpenAICompatLLM])
async def test_non_json_body_is_llm_error(factory: Any) -> None:
    client, _ = _recording_client(text="<html>502 Bad Gateway</html>")
    llm = factory(model="m", client=client)

    with pytest.raises(LLMError) as excinfo:
        await llm.complete([Message.user("hi")])

    assert excinfo.value.retryable is False
    assert "not valid JSON" in str(excinfo.value)


async def test_openai_malformed_tool_arguments_are_retryable() -> None:
    payload = json.loads(json.dumps(OPENAI_TOOL_CALL))
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = (
        '{"order_id": "SO2026'
    )
    client, _ = _recording_client(payload)

    with pytest.raises(LLMError) as excinfo:
        await OpenAICompatLLM(model="m", client=client).complete([Message.user("hi")])

    assert excinfo.value.retryable is True
    assert "malformed JSON arguments" in str(excinfo.value)


async def test_openai_non_object_tool_arguments_are_rejected() -> None:
    payload = json.loads(json.dumps(OPENAI_TOOL_CALL))
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '"SO2026"'
    client, _ = _recording_client(payload)

    with pytest.raises(LLMError, match="must be a JSON object"):
        await OpenAICompatLLM(model="m", client=client).complete([Message.user("hi")])


async def test_openai_empty_tool_arguments_become_empty_dict() -> None:
    payload = json.loads(json.dumps(OPENAI_TOOL_CALL))
    payload["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = ""
    client, _ = _recording_client(payload)

    response = await OpenAICompatLLM(model="m", client=client).complete([Message.user("hi")])

    assert response.tool_calls[0].arguments == {}


async def test_anthropic_unsupported_content_block_is_llm_error() -> None:
    payload = dict(ANTHROPIC_TEXT)
    payload["content"] = [{"type": "server_tool_use", "id": "x", "name": "web_search"}]
    client, _ = _recording_client(payload)

    with pytest.raises(LLMError, match="unsupported content block"):
        await AnthropicLLM(client=client).complete([Message.user("hi")])


async def test_openai_response_without_choices_is_llm_error() -> None:
    client, _ = _recording_client({"object": "chat.completion", "choices": []})

    with pytest.raises(LLMError, match="no choices"):
        await OpenAICompatLLM(model="m", client=client).complete([Message.user("hi")])


def test_tool_spec_without_name_is_llm_error() -> None:
    llm = AnthropicLLM()
    with pytest.raises(LLMError, match="missing 'name'"):
        llm.build_body([Message.user("hi")], tools=[{"description": "x"}])


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def test_create_llm_dispatches_on_kind() -> None:
    anthropic_llm = create_llm(LLMConfig())
    assert isinstance(anthropic_llm, AnthropicLLM)
    assert anthropic_llm.model == "deepseek-flash"
    assert anthropic_llm._url == "https://api.deepseek.com/anthropic/v1/messages"

    openai_llm = create_llm(
        LLMConfig(kind="openai_compat", base_url="http://localhost:8000/v1", model="qwen")
    )
    assert isinstance(openai_llm, OpenAICompatLLM)
    assert openai_llm.model == "qwen"


def test_create_llm_rejects_unknown_kind() -> None:
    with pytest.raises(LLMError, match="expected 'anthropic' or 'openai_compat'"):
        create_llm(LLMConfig(kind="ollama"))


def test_create_llm_passes_through_overrides() -> None:
    llm = create_llm(LLMConfig(), timeout_sec=2.0, max_tokens=64)
    assert (llm.timeout_sec, llm.max_tokens) == (2.0, 64)


# --------------------------------------------------------------------------------------
# FakeLLM
# --------------------------------------------------------------------------------------


async def test_fake_llm_replays_script_and_records_calls() -> None:
    fake = FakeLLM(
        script=[
            tool_call_response("query_order", {"order_id": "SO1"}, text="好的。"),
            text_response("您的订单已发货。"),
        ]
    )

    first = await fake.complete([Message.user("查订单")], tools=[QUERY_ORDER_TOOL], system="sys")
    assert first.stop_reason == "tool_use"
    assert first.tool_calls[0].arguments == {"order_id": "SO1"}

    second = await fake.complete(
        [Message.user("查订单"), first.as_message(), Message.user(ToolResultPart("call_1", "{}"))]
    )
    assert second.text == "您的订单已发货。"

    assert fake.call_count == 2
    assert fake.calls[0].system == "sys"
    assert fake.calls[0].tool_names == ("query_order",)
    assert len(fake.calls[1].messages) == 3


async def test_fake_llm_raises_when_script_is_exhausted() -> None:
    fake = FakeLLM(script=[text_response("only one")])
    await fake.complete([Message.user("hi")])

    with pytest.raises(ScriptExhausted, match="script has 1 response"):
        await fake.complete([Message.user("hi again")])


async def test_fake_llm_can_raise_a_scripted_error() -> None:
    fake = FakeLLM(error=LLMError("upstream down", retryable=True))
    with pytest.raises(LLMError):
        await fake.complete([Message.user("hi")])
    assert fake.call_count == 1


# --------------------------------------------------------------------------------------
# Opt-in live smoke against DeepSeek
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY is not set; the live DeepSeek smoke is opt-in",
)
async def test_deepseek_live_tool_call() -> None:
    """One real tool-calling round trip against the configured DeepSeek defaults."""
    config = LLMConfig(api_key=os.environ["DEEPSEEK_API_KEY"])
    llm = create_llm(config, timeout_sec=30.0, max_tokens=256)
    try:
        response = await llm.complete(
            [Message.user("帮我查一下订单 SO20260917001 的状态。")],
            tools=[QUERY_ORDER_TOOL],
            system="你是客服助手。需要查询订单时必须调用 query_order 工具，不要凭空回答。",
        )
    finally:
        await llm.aclose()

    assert response.tool_calls, f"expected a tool call, got {response!r}"
    call = response.tool_calls[0]
    assert call.name == "query_order"
    assert isinstance(call.arguments, dict)
