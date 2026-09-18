"""Tests for the Brain function-calling loop (T3.3). GPU-free, no marker.

Every case drives `BrainSession` with `FakeLLM` and either a fake tool runner or a real
`ToolExecutor` on an `httpx.MockTransport`, so nothing here touches a network or a model.
"""

from __future__ import annotations

import copy
from typing import Any

import httpx
import pytest

from talkover.brain.llm import (
    LLMError,
    LLMResponse,
    Message,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from talkover.brain.session import (
    BRAIN_SYSTEM_PROMPT,
    FORK_SYSTEM_SUFFIX,
    NO_ANSWER_RESULT_TEXT,
    BrainSession,
    ClientTool,
    Result,
    Share,
)
from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME, ToolExecutor, ToolResult
from talkover.config import BusinessApiConfig

from .fake_llm import FakeLLM, text_response, tool_call_response

BASE_URL = "http://business.test"


# --------------------------------------------------------------------------------------
# Doubles
# --------------------------------------------------------------------------------------


class FakeExecutor:
    """Replays a scripted `ToolResult` per tool name and records every call."""

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        self.calls.append((name, dict(arguments or {})))
        if name in self.results:
            return self.results[name]
        if name == TRANSFER_TO_HUMAN_NAME:
            return ToolResult(
                name=name,
                ok=True,
                data=dict(arguments or {}),
                client_tool=True,
            )
        return ToolResult(name=name, ok=True, data={"stub": name})


async def drain(generator) -> list[Any]:  # type: ignore[no-untyped-def]
    return [event async for event in generator]


def order_session(
    script: list[LLMResponse],
    *,
    executor: Any | None = None,
    **kwargs: Any,
) -> tuple[BrainSession, FakeLLM, Any]:
    llm = FakeLLM(script=script)
    runner = executor if executor is not None else FakeExecutor()
    return BrainSession(llm, runner, **kwargs), llm, runner


# --------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------


async def test_scripted_order_lookup_shares_then_results() -> None:
    executor = FakeExecutor(
        {
            "query_order": ToolResult(
                name="query_order",
                ok=True,
                data={"order_id": "202609170001234", "status": "shipped"},
            )
        }
    )
    session, llm, _ = order_session(
        [
            tool_call_response(
                "query_order",
                {"order_id": "202609170001234"},
                text="Let me check that order for you.",
            ),
            text_response("Your order has shipped and arrives tomorrow."),
        ],
        executor=executor,
    )

    events = await drain(session.start("Where is my order 202609170001234?"))

    assert events == [
        Share("Let me check that order for you."),
        Result("Your order has shipped and arrives tomorrow.", ok=True),
    ]
    assert executor.calls == [("query_order", {"order_id": "202609170001234"})]
    assert llm.call_count == 2
    # The system prompt carries the phrasing, and it is not a conversation message.
    assert llm.calls[0].system == BRAIN_SYSTEM_PROMPT
    assert all(message.role in ("user", "assistant") for message in llm.calls[1].messages)
    # The tool result went back as an error-free tool result part.
    tool_message = llm.calls[1].messages[-1]
    part = tool_message.content[0]
    assert isinstance(part, ToolResultPart)
    assert part.is_error is False
    assert "shipped" in part.content


async def test_all_three_tools_are_offered_on_the_main_lane() -> None:
    session, llm, _ = order_session([text_response("Hello.")])
    await drain(session.start("hi"))
    assert llm.calls[0].tool_names == ("query_ticket", "query_order", TRANSFER_TO_HUMAN_NAME)


async def test_a_silent_final_turn_falls_back_to_a_spoken_apology() -> None:
    session, _, _ = order_session([text_response("   ")])
    assert await drain(session.start("hi")) == [Result(NO_ANSWER_RESULT_TEXT, ok=True)]


async def test_a_lookup_runs_against_a_real_executor_over_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/orders"
        assert request.url.params["order_id"] == "202609170001234"
        return httpx.Response(200, json={"status": "shipped"})

    executor = ToolExecutor(
        BusinessApiConfig(base_url=BASE_URL, timeout_sec=3.0),
        transport=httpx.MockTransport(handler),
    )
    session, _, _ = order_session(
        [
            tool_call_response("query_order", {"order_id": "202609170001234"}, text="One moment."),
            text_response("It has shipped."),
        ],
        executor=executor,
    )
    events = await drain(session.start("where is my order?"))
    assert events == [Share("One moment."), Result("It has shipped.", ok=True)]
    await executor.aclose()


# --------------------------------------------------------------------------------------
# Fork (TaskQuery)
# --------------------------------------------------------------------------------------


async def test_a_fork_query_does_not_mutate_the_main_conversation() -> None:
    session, _, _ = order_session(
        [
            tool_call_response("query_order", {"order_id": "1"}, text="Checking."),
            text_response("It has shipped."),
            text_response("The order was placed yesterday."),
        ]
    )
    await drain(session.start("where is my order 1?"))
    before = copy.deepcopy(session.messages)

    events = await drain(session.query("when did I place it?"))

    assert events == [Result("The order was placed yesterday.", ok=True)]
    assert session.messages == before


async def test_a_fork_is_not_offered_the_transfer_tool() -> None:
    session, llm, _ = order_session([text_response("Yesterday.")])
    await drain(session.query("when did I place it?"))
    assert llm.calls[0].tool_names == ("query_ticket", "query_order")
    assert llm.calls[0].system.endswith(FORK_SYSTEM_SUFFIX)


async def test_a_fork_that_calls_transfer_anyway_gets_a_tool_error() -> None:
    executor = FakeExecutor()
    session, llm, _ = order_session(
        [
            tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"}),
            text_response("Yesterday."),
        ],
        executor=executor,
    )
    events = await drain(session.query("when did I place it?"))
    assert events == [Result("Yesterday.", ok=True)]
    # The transfer never reached the executor and never became a ClientTool event.
    assert executor.calls == []
    part = llm.calls[1].messages[-1].content[0]
    assert isinstance(part, ToolResultPart)
    assert part.is_error is True
    assert session.pending_client_tool is None


# --------------------------------------------------------------------------------------
# Round cap
# --------------------------------------------------------------------------------------


async def test_the_round_cap_is_enforced() -> None:
    script = [
        tool_call_response("query_order", {"order_id": "1"}, call_id=f"c{i}") for i in range(10)
    ]
    session, llm, _ = order_session(script, max_rounds=3)

    events = await drain(session.start("where is my order?"))

    assert llm.call_count == 3
    assert len(events) == 1
    assert isinstance(events[0], Result)
    assert events[0].ok is False


def test_max_rounds_must_be_positive() -> None:
    with pytest.raises(ValueError):
        BrainSession(FakeLLM(), FakeExecutor(), max_rounds=0)


# --------------------------------------------------------------------------------------
# Lookup failure
# --------------------------------------------------------------------------------------


async def test_a_failed_lookup_goes_back_to_the_model_as_an_error() -> None:
    executor = FakeExecutor(
        {
            "query_order": ToolResult(
                name="query_order",
                ok=False,
                error_kind="timeout",
                detail="business API did not respond within 3.0s",
            )
        }
    )
    session, llm, _ = order_session(
        [
            tool_call_response("query_order", {"order_id": "1"}, text="Checking."),
            text_response("I cannot reach the system. Shall I put you through to an agent?"),
        ],
        executor=executor,
    )

    events = await drain(session.start("where is my order 1?"))

    # A failed lookup produces no share; there is nothing true to say yet.
    assert events == [
        Result("I cannot reach the system. Shall I put you through to an agent?", ok=True)
    ]
    part = llm.calls[1].messages[-1].content[0]
    assert isinstance(part, ToolResultPart)
    assert part.is_error is True
    assert "timeout" in part.content


# --------------------------------------------------------------------------------------
# Transfer to human
# --------------------------------------------------------------------------------------


async def test_transfer_emits_a_client_tool_and_stops_calling_the_llm() -> None:
    session, llm, _ = order_session(
        [
            tool_call_response(
                TRANSFER_TO_HUMAN_NAME,
                {"department": "after_sales", "reason": "customer asked"},
                call_id="call_x",
            ),
            text_response("never reached"),
        ]
    )

    events = await drain(session.start("get me a person"))

    assert events == [
        ClientTool(
            "call_x",
            TRANSFER_TO_HUMAN_NAME,
            {"department": "after_sales", "reason": "customer asked"},
        )
    ]
    assert llm.call_count == 1
    assert session.pending_client_tool == "call_x"
    assert session.pending_client_tool_name == TRANSFER_TO_HUMAN_NAME


async def test_resolving_the_client_tool_continues_and_ends_the_task() -> None:
    session, llm, _ = order_session(
        [
            tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "billing"}, call_id="call_x"),
            text_response("You are being transferred now."),
        ]
    )
    await drain(session.start("get me a person"))

    events = await drain(session.resolve('{"ok": true}', call_id="call_x"))

    assert events == [Result("You are being transferred now.", ok=True)]
    assert session.pending_client_tool is None
    part = llm.calls[1].messages[-1].content[0]
    assert isinstance(part, ToolResultPart)
    assert part.tool_call_id == "call_x"
    assert part.content == '{"ok": true}'


async def test_resolving_without_a_pending_call_is_a_caller_bug() -> None:
    session, _, _ = order_session([])
    with pytest.raises(RuntimeError):
        await drain(session.resolve("{}"))


async def test_resolving_the_wrong_call_id_is_rejected() -> None:
    session, _, _ = order_session(
        [tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "billing"}, call_id="call_x")]
    )
    await drain(session.start("agent please"))
    with pytest.raises(RuntimeError):
        await drain(session.resolve("{}", call_id="other"))


# --------------------------------------------------------------------------------------
# LLM failures
# --------------------------------------------------------------------------------------


async def test_a_retryable_llm_error_is_retried_once() -> None:
    llm = FakeLLM(
        script=[text_response("It has shipped.")],
        errors=[LLMError("rate limited", retryable=True), None],
    )
    session = BrainSession(llm, FakeExecutor())

    events = await drain(session.start("where is my order?"))

    assert events == [Result("It has shipped.", ok=True)]
    assert llm.call_count == 2


async def test_a_second_retryable_failure_ends_the_task() -> None:
    llm = FakeLLM(
        script=[],
        errors=[
            LLMError("rate limited", retryable=True),
            LLMError("rate limited", retryable=True),
        ],
    )
    session = BrainSession(llm, FakeExecutor(), failure_text="sorry, transfer")

    events = await drain(session.start("where is my order?"))

    assert events == [Result("sorry, transfer", ok=False)]
    assert llm.call_count == 2


async def test_a_non_retryable_llm_error_is_not_retried() -> None:
    llm = FakeLLM(error=LLMError("bad api key", retryable=False))
    session = BrainSession(llm, FakeExecutor(), failure_text="sorry, transfer")

    events = await drain(session.start("where is my order?"))

    assert events == [Result("sorry, transfer", ok=False)]
    assert llm.call_count == 1


# --------------------------------------------------------------------------------------
# TaskUpdate
# --------------------------------------------------------------------------------------


async def test_an_update_appends_to_the_same_conversation() -> None:
    session, llm, _ = order_session(
        [
            text_response("Could you tell me your order number?"),
            text_response("That order has shipped."),
        ]
    )
    await drain(session.start("where is my order?"))

    events = await drain(session.update("it is 202609170001234"))

    assert events == [Result("That order has shipped.", ok=True)]
    second = llm.calls[1].messages
    assert second[0] == Message.user("where is my order?")
    assert second[1].role == "assistant"
    assert second[2] == Message.user("it is 202609170001234")
    assert len(second) == 3


async def test_an_update_clears_a_pending_client_tool() -> None:
    session, _, _ = order_session(
        [
            tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "billing"}, call_id="call_x"),
            text_response("Fine, staying with me."),
        ]
    )
    await drain(session.start("agent please"))
    assert session.pending_client_tool == "call_x"

    await drain(session.update("actually, never mind"))

    assert session.pending_client_tool is None


# --------------------------------------------------------------------------------------
# Spoken numbers
# --------------------------------------------------------------------------------------

SPOKEN_TEXT = "订单 202609170001234 已发货，退款 ¥1,234.50。"
SPOKEN_REWRITTEN = "订单 二零二六 零九一七 零零零一 二三四 已发货，退款 一千二百三十四元五角。"


def _spoken_script() -> list[LLMResponse]:
    return [
        tool_call_response("query_order", {"order_id": "202609170001234"}, text=SPOKEN_TEXT),
        text_response(SPOKEN_TEXT),
    ]


async def test_spoken_numbers_off_leaves_the_text_alone() -> None:
    session, _, _ = order_session(_spoken_script())
    events = await drain(session.start("where is my order?"))
    assert events == [Share(SPOKEN_TEXT), Result(SPOKEN_TEXT, ok=True)]


async def test_spoken_numbers_on_rewrites_share_and_result() -> None:
    session, _, _ = order_session(_spoken_script(), spoken_numbers=True)
    events = await drain(session.start("where is my order?"))
    assert events == [Share(SPOKEN_REWRITTEN), Result(SPOKEN_REWRITTEN, ok=True)]


# --------------------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------------------


async def test_aclose_closes_the_llm_but_not_the_executor() -> None:
    session, llm, executor = order_session([])
    await session.aclose()
    assert llm.closed is True
    assert executor.calls == []


async def test_parallel_tool_calls_are_answered_in_one_user_message() -> None:
    executor = FakeExecutor()
    llm = FakeLLM(
        script=[
            LLMResponse(
                parts=(
                    TextPart("Checking both."),
                    ToolCallPart(id="a", name="query_order", arguments={"order_id": "1"}),
                    ToolCallPart(id="b", name="query_ticket", arguments={"ticket_id": "T1"}),
                ),
                stop_reason="tool_use",
            ),
            text_response("Both are fine."),
        ]
    )
    session = BrainSession(llm, executor)

    events = await drain(session.start("check my order and ticket"))

    assert events == [Share("Checking both."), Result("Both are fine.", ok=True)]
    assert [name for name, _ in executor.calls] == ["query_order", "query_ticket"]
    last = llm.calls[1].messages[-1]
    assert last.role == "user"
    assert [part.tool_call_id for part in last.content] == ["a", "b"]
