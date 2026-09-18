"""Realtime <-> Brain wiring (T3.7): the `transfer_to_human` round trip.

GPU-free and network-free: `FakeEngine` plays the Cerebellum's `task_start` unit and
`FakeLLM` plays the business model, so the whole chain

    engine tool call -> BusinessProject.open_run -> BrainSession
        -> ProviderEvent(kind="interaction") -> response.function_call_arguments.delta/.done
        -> conversation.item.create(function_call_output) -> BusinessRun.respond
        -> BrainSession.resolve

runs in one event loop with no model and no HTTP.

The cases cover what DESIGN.md 5.3 / 6.3 and section 8.4 of `docs/protocol-profile.md`
fix: the emitted pair, the reply path, the pre-intercepted (keyword) path, the rejection
of an unknown `call_id`, that `share` / `result` never reach the client, and that a client
which never declared the tool still gets the call.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from brain.fake_llm import FakeLLM, text_response, tool_call_response
from fastapi.testclient import TestClient
from gander_runtime.coordination import ProjectRecord

from talkover.app import build_app
from talkover.brain.provider import BusinessProvider
from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME, ToolResult
from talkover.config import BrainConfig, RealtimeConfig, ServerConfig, TalkoverConfig
from talkover.engine.asr import Transcript
from talkover.engine.protocol import WORKER_DELIVERY_TOPICS, WORKER_DELIVERY_TYPE
from talkover.realtime import events as ev
from talkover.realtime.brain_bridge import DELIVERY_TOPICS, TASK_START_TOOL, BrainBridge
from talkover.realtime.session import RealtimeSession

from .fake_engine import FakeEngine, step_event

PROFILE = Path(__file__).resolve().parents[2] / "docs" / "protocol-profile.md"

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

#: What the keyword rule (T3.5) is configured with here; runtime data, hence Chinese.
TRANSFER_KEYWORD = "转人工"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class ListSink:
    """An `EventSink` collecting the wire dicts a session emits."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, event: Mapping[str, Any]) -> None:
        json.dumps(event)
        self.events.append(dict(event))


class FakeExecutor:
    """Answers every lookup with a canned success; `transfer_to_human` is a client tool."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        self.calls.append((name, dict(arguments or {})))
        if name == TRANSFER_TO_HUMAN_NAME:
            return ToolResult(name=name, ok=True, data=dict(arguments or {}), client_tool=True)
        return ToolResult(name=name, ok=True, data={"order_id": "A1", "status": "shipped"})


class Harness:
    """One Realtime session, one Brain project and the bridge between them."""

    def __init__(
        self,
        session: RealtimeSession,
        bridge: BrainBridge,
        provider: BusinessProvider,
        llm: FakeLLM,
        engine: FakeEngine,
    ) -> None:
        self.session = session
        self.bridge = bridge
        self.provider = provider
        self.llm = llm
        self.engine = engine

    @property
    def events(self) -> list[dict[str, Any]]:
        sink = self.session.sink
        assert isinstance(sink, ListSink)
        return sink.events

    def types(self) -> list[str]:
        return [event["type"] for event in self.events]

    def one(self, kind: str) -> dict[str, Any]:
        matches = [event for event in self.events if event["type"] == kind]
        assert len(matches) == 1, f"expected exactly one {kind}, got {self.types()}"
        return matches[0]

    def drop(self) -> None:
        self.events.clear()

    @property
    def deliveries(self) -> list[dict[str, Any]]:
        """What reached the Cerebellum through `EngineProtocol.feed_worker_delivery`."""
        return self.engine.deliveries

    @property
    def tool_responses(self) -> list[dict[str, Any]]:
        """What answered the model's native tool calls (`feed_tool_response`)."""
        return self.engine.tool_responses

    async def aclose(self) -> None:
        await self.bridge.aclose()
        await self.provider.close()


async def make_harness(
    script: list[Any],
    *,
    keywords: tuple[str, ...] = (),
    transcript: str = "",
) -> Harness:
    llm = FakeLLM(script=list(script))
    provider = BusinessProvider(
        BrainConfig(transfer_keywords=keywords), llm=llm, executor=FakeExecutor()
    )
    project = await provider.open_project(
        ProjectRecord(
            project_id="project_1",
            owner_id="owner_1",
            label="customer service",
            provider_name=provider.name,
        )
    )
    engine = FakeEngine()
    bridge = BrainBridge(project, engine=engine)
    session = RealtimeSession(
        TalkoverConfig(),
        engine,
        sink=ListSink(),
        on_engine_event=bridge.on_engine_event,
        on_function_call_output=bridge.on_function_call_output,
    )
    bridge.bind(session)
    if transcript:
        await bridge.on_asr_event(Transcript(text=transcript, start_ms=0, end_ms=1200))
    return Harness(session, bridge, provider, llm, engine)


def task_start_unit(index: int = 1, name: str = "order status") -> Any:
    """The unit in which the Cerebellum asks the runtime to open a task."""
    return step_event(
        index,
        is_listen=False,
        is_tool_call=True,
        tool_calls=({"name": TASK_START_TOOL, "arguments": {"name": name}},),
        tool_response_expected=True,
    )


def function_call_output(call_id: str, output: str = '{"status": "transferred"}') -> dict[str, Any]:
    return {
        "type": "conversation.item.create",
        "item": {"type": "function_call_output", "call_id": call_id, "output": output},
    }


def wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    """Poll from the test thread while the application's own loop runs elsewhere."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the application never reached the expected state")
        time.sleep(0.005)


async def settle(predicate: Any, *, timeout: float = 2.0) -> None:
    """Let the bridge's pump task run until `predicate` holds."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not predicate():
        if loop.time() > deadline:
            raise AssertionError("the Brain bridge never reached the expected state")
        await asyncio.sleep(0.002)


FUNCTION_CALL_CHAIN = (
    "response.created",
    "response.output_item.added",
    "conversation.item.created",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.output_item.done",
    "response.done",
)


# ---------------------------------------------------------------------------
# the round trip
# ---------------------------------------------------------------------------


async def test_transfer_round_trip_reaches_brain_session_resolve() -> None:
    """The acceptance case: Brain asks, client answers, `BrainSession.resolve` continues."""
    harness = await make_harness(
        [
            tool_call_response(
                TRANSFER_TO_HUMAN_NAME,
                {"department": "billing", "reason": "退款"},
                call_id="call_9",
            ),
            text_response("I am connecting you to an agent now."),
        ],
        transcript="I want a refund on order A1",
    )
    try:
        await harness.session.handle_engine_event(task_start_unit())
        await settle(lambda: harness.bridge.pending_call_ids)

        assert tuple(harness.types()) == FUNCTION_CALL_CHAIN
        delta = harness.one("response.function_call_arguments.delta")
        done = harness.one("response.function_call_arguments.done")
        assert delta["call_id"] == "call_9"
        assert done["name"] == TRANSFER_TO_HUMAN_NAME
        assert json.loads(delta["delta"]) == {"department": "billing", "reason": "退款"}
        assert delta["delta"] == done["arguments"]
        item = harness.one("conversation.item.created")["item"]
        assert item["type"] == "function_call"
        assert item["call_id"] == "call_9"

        # The Brain saw the customer's trusted text, not the task name.
        assert harness.llm.calls[0].last_message.text == "I want a refund on order A1"

        harness.drop()
        await harness.session.handle_raw(function_call_output("call_9"))
        await settle(lambda: harness.llm.call_count == 2)

        # `respond()` reached `BrainSession.resolve`: the output is in the conversation
        # as the tool result of the pending call, and the loop ran one more round.
        resolved = harness.llm.calls[1].messages[-1]
        assert resolved.role == "user"
        assert resolved.content[0].tool_call_id == "call_9"
        assert resolved.content[0].content == '{"status": "transferred"}'

        # The client only sees the acknowledgement of its own item; the Brain's result
        # goes to the Cerebellum (DESIGN.md 5.3).
        assert harness.types() == ["conversation.item.created"]
        await settle(lambda: harness.deliveries)
        # The envelope is upstream's `worker_delivery_response` shape: the model is told
        # the task name it chose itself, and a Brain result is the terminal `final`.
        assert harness.deliveries[-1] == {
            "type": WORKER_DELIVERY_TYPE,
            "task_name": "order status",
            "topic": "final",
            "content": "I am connecting you to an agent now.",
            "status": "completed",
        }
        assert harness.bridge.pending_call_ids == ()
    finally:
        await harness.aclose()


async def test_intercepted_transfer_skips_the_llm() -> None:
    """T3.5: a keyword hit emits the same pair without a single LLM round."""
    harness = await make_harness([], keywords=(TRANSFER_KEYWORD,), transcript="你好，我要转人工")
    try:
        await harness.session.handle_engine_event(task_start_unit(name="转接"))
        await settle(lambda: harness.bridge.pending_call_ids)

        assert tuple(harness.types()) == FUNCTION_CALL_CHAIN
        assert harness.llm.call_count == 0
        delta = harness.one("response.function_call_arguments.delta")
        assert json.loads(delta["delta"]) == {"department": "general"}

        call_id = harness.bridge.pending_call_ids[0]
        harness.drop()
        await harness.session.handle_raw(function_call_output(call_id))
        # The intercepted call never entered the loop, so answering it completes the task.
        await settle(lambda: not harness.bridge.task_ids)
        assert harness.llm.call_count == 0
        assert harness.types() == ["conversation.item.created"]
    finally:
        await harness.aclose()


async def test_unknown_call_id_is_rejected() -> None:
    """Section 8.4: an output for a call this session never made is `invalid_value`."""
    harness = await make_harness([])
    try:
        await harness.session.handle_raw(function_call_output("call_nope"))
        error = harness.one("error")["error"]
        assert error["code"] == "invalid_value"
        assert error["param"] == "item.call_id"
        assert "call_nope" in error["message"]
        # The item itself is acknowledged first; the rejection is about the routing.
        assert harness.types() == ["conversation.item.created", "error"]
    finally:
        await harness.aclose()


async def test_shares_are_not_surfaced_to_the_client() -> None:
    """DESIGN.md 5.3: Brain `share` goes to the Cerebellum, never to the WebSocket."""
    harness = await make_harness(
        [
            tool_call_response(
                "query_order", {"order_id": "A1"}, call_id="call_1", text="Let me check that."
            ),
            text_response("Your order shipped yesterday."),
        ],
        transcript="Where is order A1?",
    )
    try:
        await harness.session.handle_engine_event(task_start_unit())
        await settle(lambda: len(harness.deliveries) == 2)
        assert [delivery["topic"] for delivery in harness.deliveries] == [
            "milestone",
            "final",
        ]
        assert harness.deliveries[0] == {
            "type": WORKER_DELIVERY_TYPE,
            "task_name": "order status",
            "topic": "milestone",
            "content": "Let me check that.",
        }
        assert "status" not in harness.deliveries[0]
        assert harness.types() == []
        assert harness.bridge.pending_call_ids == ()
    finally:
        await harness.aclose()


async def test_the_task_start_call_is_answered_with_the_task_id() -> None:
    """The model blocks on its native tool call, so the bridge answers it."""
    harness = await make_harness([text_response("Sure.")], transcript="hello")
    try:
        await harness.session.handle_engine_event(task_start_unit())
        assert len(harness.tool_responses) == 1
        answer = harness.tool_responses[0]
        assert answer["status"] == "ok"
        assert len(answer["task_ids"]) == 1
        assert answer["task_ids"][0].startswith("task")
    finally:
        await harness.aclose()


async def test_the_cerebellum_channel_is_the_engine() -> None:
    """T1.4: the whole round trip goes back into the model through `EngineProtocol`.

    The `task_start` unit is answered with `feed_tool_response` while the model waits, and
    the Brain result that the client's `function_call_output` unblocks is handed over with
    `feed_worker_delivery` — never to the WebSocket.
    """
    harness = await make_harness(
        [
            tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "billing"}, call_id="call_5"),
            text_response("An agent will be with you."),
        ],
        transcript="I need a person",
    )
    try:
        await harness.session.handle_engine_event(task_start_unit(name="transfer"))
        # Answered on the unit itself: the model produces nothing until it is.
        assert harness.engine.call_names == ["feed_tool_response"]
        await settle(lambda: harness.bridge.pending_call_ids)

        call_id = harness.bridge.pending_call_ids[0]
        await harness.session.handle_raw(function_call_output(call_id))
        await settle(lambda: harness.deliveries)

        assert harness.engine.call_names == ["feed_tool_response", "feed_worker_delivery"]
        assert harness.deliveries[-1] == {
            "type": WORKER_DELIVERY_TYPE,
            "task_name": "transfer",
            "topic": "final",
            "content": "An agent will be with you.",
            "status": "completed",
        }
        assert "response.function_call_arguments.done" in harness.types()
    finally:
        await harness.aclose()


async def test_the_delivery_topics_are_the_upstream_literal() -> None:
    """The envelope must stay inside `DeliveryRecord.topic`, or the model sees a new word."""
    from typing import get_args, get_type_hints

    from gander_runtime.coordination import DeliveryRecord

    hints = get_type_hints(DeliveryRecord)
    assert WORKER_DELIVERY_TOPICS == set(get_args(hints["topic"]))
    assert set(DELIVERY_TOPICS.values()) <= WORKER_DELIVERY_TOPICS


async def test_a_refused_delivery_does_not_end_the_session(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An engine that refuses the envelope costs one spoken update, not the call."""
    harness = await make_harness([text_response("Your order shipped.")], transcript="where is it")
    harness.engine.fail_on.add("feed_worker_delivery")
    try:
        with caplog.at_level("WARNING"):
            await harness.session.handle_engine_event(task_start_unit())
            await settle(lambda: not harness.bridge.task_ids)
        assert harness.deliveries == []
        assert any("never reached the model" in record.message for record in caplog.records)
        assert harness.types() == []
    finally:
        await harness.aclose()


async def test_a_bridge_without_an_engine_drops_the_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The engine stays optional so a unit test can build a bridge without one."""
    harness = await make_harness([text_response("Sure.")], transcript="hello")
    harness.bridge.engine = None
    try:
        with caplog.at_level("WARNING"):
            await harness.session.handle_engine_event(task_start_unit())
            await settle(lambda: not harness.bridge.task_ids)
        assert harness.engine.call_names == []
        assert any("stays unanswered" in record.message for record in caplog.records)
    finally:
        await harness.aclose()


async def test_the_call_is_emitted_even_when_the_client_declared_no_tools(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Section 9: the transfer decision is the Brain's, so an empty `tools` never drops it."""
    harness = await make_harness(
        [tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"}, call_id="call_2")],
        transcript="Put me through to someone",
    )
    try:
        assert harness.session.session["tools"] == []
        with caplog.at_level("WARNING"):
            await harness.session.handle_engine_event(task_start_unit())
            await settle(lambda: harness.bridge.pending_call_ids)
        assert tuple(harness.types()) == FUNCTION_CALL_CHAIN
        assert any(TRANSFER_TO_HUMAN_NAME in record.message for record in caplog.records)
    finally:
        await harness.aclose()


async def test_a_tool_call_unit_opens_no_speech_response() -> None:
    """A `task_start` unit is not speech: the mapper must not open a message response."""
    harness = await make_harness([text_response("Sure.")])
    try:
        await harness.session.handle_engine_event(task_start_unit())
        assert harness.session.active_response_id is None
        assert "response.created" not in harness.types()
    finally:
        await harness.aclose()


# ---------------------------------------------------------------------------
# profile section 8.4
# ---------------------------------------------------------------------------


BRIDGE_CASES: tuple[str, ...] = ("function_call_output_unknown_call_id",)


def _profile_bridge_cases() -> dict[str, tuple[str, str | None]]:
    """case id -> (code, param) as documented in section 8.4."""
    text = PROFILE.read_text(encoding="utf-8")
    start = text.index("### 8.4 Brain bridge rejections")
    rows: list[list[str]] = []
    seen_table = False
    for line in text[start:].splitlines()[1:]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if seen_table:
                break
            continue
        seen_table = True
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if all(set(cell) <= {"-", ":"} and cell for cell in cells):
            continue
        rows.append(cells)
    return {
        row[0].strip().strip("`"): (row[2].strip().strip("`"), row[3].strip().strip("`") or None)
        for row in rows[1:]
    }


def test_profile_bridge_cases_are_covered() -> None:
    assert set(_profile_bridge_cases()) == set(BRIDGE_CASES)


async def test_bridge_rejection_matches_the_profile() -> None:
    documented = _profile_bridge_cases()
    code, param = documented["function_call_output_unknown_call_id"]
    harness = await make_harness([])
    try:
        await harness.session.handle_raw(function_call_output("call_nope"))
        error = harness.one("error")["error"]
        assert (error["code"], error["param"]) == (code, param)
        assert error["type"] == ev.ERROR_CODE_TYPES[code]
    finally:
        await harness.aclose()


# ---------------------------------------------------------------------------
# the composed application (`talkover.app`)
# ---------------------------------------------------------------------------


def test_round_trip_over_the_websocket_transport() -> None:
    """The same round trip through `build_app`, a real WebSocket and the session slot."""
    llm = FakeLLM(
        script=[
            tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"}, call_id="call_7"),
            text_response("An agent is picking up."),
        ]
    )
    provider = BusinessProvider(BrainConfig(), llm=llm, executor=FakeExecutor())
    # `response.create` is the client-driven `flush_pending` this fake answers with the
    # Cerebellum's `task_start` unit; a zero window releases the slot on disconnect.
    engine = FakeEngine(on_call={"flush_pending": [[task_start_unit()]]})
    config = TalkoverConfig(
        server=ServerConfig(listen="127.0.0.1:8000", api_key=API_KEY),
        realtime=RealtimeConfig(trailing_silence_sec=0.0),
    )
    composed = asyncio.run(build_app(config, engine, provider=provider))
    client = TestClient(composed.app)
    try:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as websocket:
            websocket.receive_json()  # session.created
            websocket.receive_json()  # conversation.created
            websocket.send_json({"type": "response.create"})
            chain = [websocket.receive_json() for _ in range(len(FUNCTION_CALL_CHAIN))]
            assert tuple(event["type"] for event in chain) == FUNCTION_CALL_CHAIN
            call_id = chain[3]["call_id"]
            assert call_id == "call_7"

            websocket.send_json(function_call_output(call_id))
            created = websocket.receive_json()
            assert created["type"] == "conversation.item.created"
            assert created["item"]["type"] == "function_call_output"
            # The client's output reached `BrainSession.resolve`, which ran one more round.
            wait_until(lambda: llm.call_count == 2)
    finally:
        client.close()
        asyncio.run(composed.aclose())
    assert composed.bridge is None
