"""Session state machine tests (T2.4): client event in, engine call and ack out.

Every client event of the mapping table in DESIGN.md 5.3 has a case here asserting both
halves of the contract: the `EngineProtocol` call the session makes on the fake engine,
and the acknowledgement event `docs/protocol-profile.md` section 9 documents. Section 8.3
of the profile (the state-dependent rejections this layer owns) is machine-read the same
way `test_events.py` reads sections 8.1 and 8.2.

These cases are GPU-free: the engine is `tests/protocol/fake_engine.FakeEngine`.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from talkover.config import RealtimeConfig, ServerConfig, TalkoverConfig
from talkover.realtime import events as ev
from talkover.realtime import server as server_module
from talkover.realtime.audio import alaw_encode, encode_base64_pcm16, ulaw_encode
from talkover.realtime.server import create_app
from talkover.realtime.session import (
    DEFAULT_MODEL,
    DEFAULT_VOICE,
    RealtimeSession,
    WebSocketSession,
)

from .fake_engine import FakeEngine, step_event

PROFILE = Path(__file__).resolve().parents[2] / "docs" / "protocol-profile.md"

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class ListSink:
    """An `EventSink` collecting the wire dicts a session emits."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, event: Mapping[str, Any]) -> None:
        json.dumps(event)  # every emitted event must be JSON-serializable
        self.events.append(dict(event))


def make_session(
    engine: FakeEngine | None = None,
    *,
    instructions_cap: int = 256,
    model: str = DEFAULT_MODEL,
    **kwargs: Any,
) -> tuple[RealtimeSession, FakeEngine]:
    engine = engine if engine is not None else FakeEngine()
    config = TalkoverConfig(realtime=RealtimeConfig(memory_slate_max_tokens=instructions_cap))
    session = RealtimeSession(config, engine, sink=ListSink(), model=model, **kwargs)
    return session, engine


def emitted(session: RealtimeSession) -> list[dict[str, Any]]:
    sink = session.sink
    assert isinstance(sink, ListSink)
    return sink.events


def types_of(session: RealtimeSession) -> list[str]:
    return [event["type"] for event in emitted(session)]


def only(session: RealtimeSession, kind: str) -> dict[str, Any]:
    matches = [event for event in emitted(session) if event["type"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, got {types_of(session)}"
    return matches[0]


def drop(session: RealtimeSession) -> None:
    """Forget everything emitted so far, to keep an assertion focused."""
    emitted(session).clear()


async def expect_error(session: RealtimeSession, frame: Mapping[str, Any]) -> dict[str, Any]:
    """Drive one frame that must be rejected and return the `error` payload."""
    before = len(emitted(session))
    try:
        await session.handle_raw(frame)
    except ev.ProtocolError as exc:
        return dict(ev.error_event(exc)["error"])
    new = [event for event in emitted(session)[before:] if event["type"] == "error"]
    assert len(new) == 1, f"expected one error, got {types_of(session)}"
    return dict(new[0]["error"])


def pcm24(seconds: float, value: int = 1000) -> str:
    """Base64 pcm16 at the client rate."""
    samples = np.full(int(24000 * seconds), value, dtype=np.int16)
    return encode_base64_pcm16(samples)


def ulaw8(seconds: float, value: int = 1000) -> str:
    samples = np.full(int(8000 * seconds), value, dtype=np.int16)
    return base64.b64encode(ulaw_encode(samples)).decode("ascii")


def session_update(**session_fields: Any) -> dict[str, Any]:
    return {"type": "session.update", "session": {"type": "realtime", **session_fields}}


# ---------------------------------------------------------------------------
# session.created / conversation.created
# ---------------------------------------------------------------------------


async def test_open_emits_session_created_then_conversation_created() -> None:
    session, _ = make_session(model="gpt-realtime")
    await session.open()

    assert types_of(session) == ["session.created", "conversation.created"]
    created = emitted(session)[0]
    assert created["event_id"].startswith("event_")
    assert created["session"]["id"] == session.session_id
    assert created["session"]["id"].startswith("sess_")
    assert created["session"]["object"] == "realtime.session"
    assert created["session"]["type"] == "realtime"
    assert created["session"]["model"] == "gpt-realtime"
    assert created["session"]["instructions"] == ""
    assert created["session"]["output_modalities"] == ["audio"]
    assert created["session"]["tools"] == []
    assert created["session"]["audio"]["input"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert created["session"]["audio"]["output"]["voice"] == DEFAULT_VOICE

    conversation = emitted(session)[1]["conversation"]
    assert conversation == {"id": session.conversation_id, "object": "realtime.conversation"}
    assert conversation["id"].startswith("conv_")


async def test_default_model_matches_the_server_query_parameter_echo() -> None:
    assert DEFAULT_MODEL == server_module.DEFAULT_MODEL


async def test_session_view_is_a_copy() -> None:
    session, _ = make_session()
    view = session.session_view()
    view["audio"]["input"]["format"]["type"] = "audio/pcmu"
    assert session.session["audio"]["input"]["format"]["type"] == "audio/pcm"


# ---------------------------------------------------------------------------
# session.update
# ---------------------------------------------------------------------------


async def test_session_update_echoes_the_full_effective_session() -> None:
    session, engine = make_session()
    await session.handle_raw(
        session_update(
            model="gpt-realtime",
            tools=[{"type": "function", "name": "transfer_to_human", "parameters": {}}],
            tool_choice="required",
            max_output_tokens=512,
            audio={
                "input": {
                    "format": {"type": "audio/pcmu"},
                    "noise_reduction": {"type": "far_field"},
                },
                "output": {"voice": "verse", "speed": 1.25},
            },
        )
    )

    updated = only(session, "session.updated")["session"]
    assert updated["model"] == "gpt-realtime"
    assert updated["tools"] == [{"type": "function", "name": "transfer_to_human", "parameters": {}}]
    assert updated["tool_choice"] == "required"
    assert updated["max_output_tokens"] == 512
    assert updated["audio"]["input"]["format"] == {"type": "audio/pcmu", "rate": 8000}
    assert updated["audio"]["input"]["noise_reduction"] == {"type": "far_field"}
    assert updated["audio"]["output"]["voice"] == "verse"
    assert updated["audio"]["output"]["speed"] == 1.25
    # Ignored fields keep their documented echo.
    assert updated["truncation"] == "auto"
    assert updated["tracing"] is None
    assert updated["prompt"] is None
    assert updated["id"] == session.session_id
    # No instructions in this update, so the task slate was left alone.
    assert engine.call_names == []


async def test_session_update_merges_field_wise() -> None:
    session, _ = make_session()
    await session.handle_raw(session_update(audio={"output": {"voice": "verse", "speed": 1.25}}))
    drop(session)
    await session.handle_raw(session_update(audio={"output": {"speed": 0.5}}))

    output = only(session, "session.updated")["session"]["audio"]["output"]
    assert output == {
        "format": {"type": "audio/pcm", "rate": 24000},
        "voice": "verse",
        "speed": 0.5,
    }


async def test_session_update_null_clears_a_nullable_field() -> None:
    session, _ = make_session()
    await session.handle_raw(session_update(audio={"input": {"turn_detection": None}}))

    assert only(session, "session.updated")["session"]["audio"]["input"]["turn_detection"] is None
    assert session.turn_detection is None


async def test_instructions_go_to_the_task_slate_and_are_echoed() -> None:
    session, engine = make_session()
    await session.handle_raw(session_update(instructions="Be brief."))

    assert engine.calls == [("set_task_slate", "Be brief.")]
    assert only(session, "session.updated")["session"]["instructions"] == "Be brief."


async def test_long_instructions_are_truncated_before_the_slate_and_the_echo() -> None:
    session, engine = make_session(instructions_cap=4)
    await session.handle_raw(session_update(instructions="x" * 400))

    truncated = engine.argument("set_task_slate")
    assert truncated == "x" * 16  # 4 tokens at four characters per token
    assert ev.estimate_tokens(truncated) <= 4
    assert only(session, "session.updated")["session"]["instructions"] == truncated


async def test_session_update_rejection_keeps_the_session_open() -> None:
    session, engine = make_session()
    error = await expect_error(session, session_update(voice="verse"))

    assert error["code"] == "invalid_value"
    assert error["param"] == "session.voice"
    assert "session.updated" not in types_of(session)
    assert engine.calls == []
    drop(session)
    await session.handle_raw(session_update(instructions="ok"))
    assert only(session, "session.updated")["session"]["instructions"] == "ok"


async def test_error_echoes_the_client_event_id() -> None:
    session, _ = make_session()
    frame = {"type": "session.update", "event_id": "evt-1", "session": {"type": "realtime", "x": 1}}
    error = await expect_error(session, frame)
    assert error["event_id"] == "evt-1"


async def test_changing_the_input_format_restarts_the_input_stream() -> None:
    session, engine = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.5)})
    assert session.buffered_ms == pytest.approx(500.0)

    await session.handle_raw(session_update(audio={"input": {"format": {"type": "audio/pcmu"}}}))
    assert session.input_audio_format == "g711_ulaw"
    assert session.buffered_ms == 0.0

    # 1 s of 8 kHz mu-law still makes exactly one 16 kHz unit.
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": ulaw8(1.0)})
    assert engine.call_names == ["feed_pcm16"]
    assert engine.units[0].size == 16000


# ---------------------------------------------------------------------------
# input_audio_buffer.append / commit / clear
# ---------------------------------------------------------------------------


async def test_append_feeds_whole_units_and_buffers_the_remainder() -> None:
    session, engine = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(1.5)})

    assert engine.call_names == ["feed_pcm16"]
    assert engine.units[0].size == 16000
    assert session.buffered_ms == pytest.approx(1500.0)
    assert types_of(session) == []  # append is not acknowledged


async def test_append_accepts_the_negotiated_alaw_format() -> None:
    session, engine = make_session()
    await session.handle_raw(session_update(audio={"input": {"format": {"type": "audio/pcma"}}}))
    drop(session)
    samples = np.full(8000, 1000, dtype=np.int16)
    payload = base64.b64encode(alaw_encode(samples)).decode("ascii")
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": payload})

    assert engine.call_names == ["feed_pcm16"]
    assert engine.units[0].size == 16000


async def test_append_with_an_undecodable_payload_is_rejected() -> None:
    session, engine = make_session()
    error = await expect_error(
        session,
        {"type": "input_audio_buffer.append", "audio": "AAAAA"},  # odd byte count
    )
    assert error["code"] == "invalid_value"
    assert error["param"] == "audio"
    assert engine.calls == []


async def test_commit_flushes_the_partial_unit_then_flush_pending() -> None:
    session, engine = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.5)})
    await session.handle_raw({"type": "input_audio_buffer.commit"})

    assert engine.call_names == ["feed_pcm16", "flush_pending"]
    assert engine.units[0].size == 16000  # zero padded to a whole unit
    assert session.buffered_ms == 0.0
    assert session.committed_ms == pytest.approx(500.0)

    assert types_of(session) == ["input_audio_buffer.committed", "conversation.item.created"]
    committed = emitted(session)[0]
    created = emitted(session)[1]
    assert committed["item_id"] == session.input_item_id
    assert committed["previous_item_id"] is None
    assert created["item"] == {
        "id": committed["item_id"],
        "object": "realtime.item",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_audio", "transcript": None}],
    }
    assert created["item"]["id"].startswith("item_")


async def test_commit_links_the_previous_item() -> None:
    session, _ = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.2)})
    await session.handle_raw({"type": "input_audio_buffer.commit"})
    first = session.input_item_id
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.2)})
    drop(session)
    await session.handle_raw({"type": "input_audio_buffer.commit"})

    assert emitted(session)[0]["previous_item_id"] == first
    assert session.input_item_id != first


async def test_commit_on_an_empty_buffer_is_rejected() -> None:
    session, engine = make_session()
    error = await expect_error(session, {"type": "input_audio_buffer.commit"})

    assert error["code"] == "invalid_event"
    assert error["param"] is None
    assert engine.calls == []


async def test_clear_drops_the_buffer_and_acknowledges() -> None:
    session, engine = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.5)})
    await session.handle_raw({"type": "input_audio_buffer.clear"})

    assert types_of(session) == ["input_audio_buffer.cleared"]
    assert session.buffered_ms == 0.0
    assert engine.calls == []  # a partial unit never reached the engine
    # The buffer is empty again, so a commit now has nothing to commit.
    error = await expect_error(session, {"type": "input_audio_buffer.commit"})
    assert error["code"] == "invalid_event"


# ---------------------------------------------------------------------------
# response.create / response.cancel
# ---------------------------------------------------------------------------


async def test_response_create_is_a_flush_pending() -> None:
    session, engine = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.25)})
    await session.handle_raw({"type": "response.create"})

    assert engine.call_names == ["feed_pcm16", "flush_pending"]
    # The response itself is opened by the mapping layer when the model stops listening.
    assert types_of(session) == []


async def test_response_create_keeps_the_metadata_for_the_mapping_layer() -> None:
    session, _ = make_session()
    await session.handle_raw(
        {"type": "response.create", "response": {"conversation": "auto", "metadata": {"a": "b"}}}
    )
    assert session.pending_response_metadata == {"a": "b"}


async def test_response_create_while_a_response_is_in_progress_is_rejected() -> None:
    session, engine = make_session()
    session.active_response_id = "resp_1"
    error = await expect_error(session, {"type": "response.create"})

    assert error["code"] == "invalid_event"
    assert engine.calls == []


async def test_response_cancel_interrupts_the_active_response() -> None:
    session, engine = make_session()
    session.active_response_id = "resp_1"
    await session.handle_raw({"type": "response.cancel", "response_id": "resp_1"})

    assert engine.call_names == ["interrupt_output"]
    # `response.done` with status cancelled is emitted by the mapping layer (T2.5).
    assert types_of(session) == []


async def test_response_cancel_without_a_response_is_rejected() -> None:
    session, engine = make_session()
    error = await expect_error(session, {"type": "response.cancel"})

    assert error["code"] == "invalid_event"
    assert engine.calls == []


async def test_response_cancel_with_a_foreign_id_is_rejected() -> None:
    session, engine = make_session()
    session.active_response_id = "resp_1"
    error = await expect_error(session, {"type": "response.cancel", "response_id": "resp_2"})

    assert error["code"] == "invalid_value"
    assert error["param"] == "response_id"
    assert engine.calls == []


# ---------------------------------------------------------------------------
# conversation.item.create / truncate / delete
# ---------------------------------------------------------------------------


def text_item(text: str = "hello", **extra: Any) -> dict[str, Any]:
    item = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}
    item.update(extra)
    return {"type": "conversation.item.create", "item": item}


def output_item(call_id: str = "call_1", output: str = '{"ok": true}', **extra: Any) -> dict:
    item = {"type": "function_call_output", "call_id": call_id, "output": output}
    item.update(extra)
    return {"type": "conversation.item.create", "item": item}


async def test_text_item_becomes_a_text_user_turn() -> None:
    session, engine = make_session()
    await session.handle_raw(text_item("where is my order"))

    assert engine.calls == [("submit_text_turn", "where is my order")]
    created = only(session, "conversation.item.created")
    assert created["previous_item_id"] is None
    assert created["item"] == {
        "id": created["item"]["id"],
        "object": "realtime.item",
        "type": "message",
        "role": "user",
        "status": "completed",
        "content": [{"type": "input_text", "text": "where is my order"}],
    }
    assert created["item"]["id"] in session.items


async def test_client_supplied_item_id_is_used() -> None:
    session, _ = make_session()
    await session.handle_raw(text_item(id="item_client_1"))
    assert only(session, "conversation.item.created")["item"]["id"] == "item_client_1"
    assert "item_client_1" in session.items


async def test_duplicate_item_id_is_rejected() -> None:
    session, engine = make_session()
    await session.handle_raw(text_item(id="item_client_1"))
    drop(session)
    error = await expect_error(session, text_item(id="item_client_1"))

    assert error["code"] == "invalid_value"
    assert error["param"] == "item.id"
    assert engine.call_names == ["submit_text_turn"]  # only the first item reached the engine


async def test_function_call_output_reaches_the_brain_seam() -> None:
    received: list[ev.FunctionCallOutputItem] = []

    async def handler(item: ev.FunctionCallOutputItem) -> None:
        received.append(item)

    session, engine = make_session(on_function_call_output=handler)
    await session.handle_raw(output_item("call_9", '{"status": "shipped"}'))

    created = only(session, "conversation.item.created")
    assert created["item"] == {
        "id": created["item"]["id"],
        "object": "realtime.item",
        "type": "function_call_output",
        "call_id": "call_9",
        "output": '{"status": "shipped"}',
        "status": "completed",
    }
    assert [(item.call_id, item.output) for item in received] == [
        ("call_9", '{"status": "shipped"}')
    ]
    assert engine.calls == []  # a tool result never enters the model's audio channel


async def test_function_call_output_without_a_handler_is_still_acknowledged() -> None:
    session, engine = make_session()
    await session.handle_raw(output_item())
    assert only(session, "conversation.item.created")["item"]["call_id"] == "call_1"
    assert engine.calls == []


async def test_repeated_call_id_is_rejected() -> None:
    session, _ = make_session()
    await session.handle_raw(output_item("call_1"))
    drop(session)
    error = await expect_error(session, output_item("call_1"))

    assert error["code"] == "invalid_value"
    assert error["param"] == "item.call_id"


async def test_truncate_is_acknowledged_only() -> None:
    session, engine = make_session()
    await session.handle_raw(text_item(id="item_1"))
    drop(session)
    await session.handle_raw(
        {
            "type": "conversation.item.truncate",
            "item_id": "item_1",
            "content_index": 0,
            "audio_end_ms": 120,
        }
    )

    truncated = only(session, "conversation.item.truncated")
    assert truncated["item_id"] == "item_1"
    assert truncated["content_index"] == 0
    assert truncated["audio_end_ms"] == 120
    assert engine.call_names == ["submit_text_turn"]  # the model context is not rewound
    assert "item_1" in session.items  # truncate does not remove the item


async def test_truncate_of_an_unknown_item_is_rejected() -> None:
    session, _ = make_session()
    error = await expect_error(
        session,
        {
            "type": "conversation.item.truncate",
            "item_id": "item_nope",
            "content_index": 0,
            "audio_end_ms": 0,
        },
    )
    assert error["code"] == "invalid_value"
    assert error["param"] == "item_id"


async def test_delete_is_acknowledged_and_forgets_the_item() -> None:
    session, engine = make_session()
    await session.handle_raw(text_item(id="item_1"))
    drop(session)
    await session.handle_raw({"type": "conversation.item.delete", "item_id": "item_1"})

    assert only(session, "conversation.item.deleted")["item_id"] == "item_1"
    assert "item_1" not in session.items
    assert session.item_order == []
    assert engine.call_names == ["submit_text_turn"]


async def test_delete_of_an_unknown_item_is_rejected() -> None:
    session, _ = make_session()
    error = await expect_error(
        session, {"type": "conversation.item.delete", "item_id": "item_nope"}
    )
    assert error["code"] == "invalid_value"
    assert error["param"] == "item_id"


async def test_register_item_is_the_mapping_seam() -> None:
    session, _ = make_session()
    session.register_item(
        {"id": "item_assistant", "object": "realtime.item", "type": "message", "role": "assistant"}
    )
    assert session.last_item_id == "item_assistant"
    await session.handle_raw(
        {
            "type": "conversation.item.truncate",
            "item_id": "item_assistant",
            "content_index": 0,
            "audio_end_ms": 40,
        }
    )
    assert only(session, "conversation.item.truncated")["item_id"] == "item_assistant"


# ---------------------------------------------------------------------------
# framing and engine failures
# ---------------------------------------------------------------------------


async def test_a_binary_frame_is_rejected() -> None:
    session, _ = make_session()
    error = await expect_error(session, b'{"type": "input_audio_buffer.clear"}')
    assert error["code"] == "invalid_event"
    assert error["param"] is None


async def test_a_json_text_frame_is_accepted() -> None:
    session, _ = make_session()
    await session.handle_raw(json.dumps({"type": "input_audio_buffer.clear"}))
    assert types_of(session) == ["input_audio_buffer.cleared"]


async def test_an_engine_failure_is_fatal() -> None:
    session, _ = make_session(FakeEngine(fail_on={"submit_text_turn"}))
    with pytest.raises(ev.ProtocolError) as excinfo:
        await session.handle_raw(text_item())
    assert excinfo.value.code == "engine_error"


# ---------------------------------------------------------------------------
# the engine-event pump (the T2.5 / T2.6 seam)
# ---------------------------------------------------------------------------


async def test_pump_hands_every_step_event_to_the_hook() -> None:
    seen: list[int] = []

    async def hook(event: Any) -> None:
        seen.append(event.unit_index)

    engine = FakeEngine()
    session, _ = make_session(engine, on_engine_event=hook)
    task = asyncio.create_task(session.pump_engine_events())
    await engine.push(step_event(1), step_event(2, is_listen=False, text="hi"))
    await engine.stop()
    await task

    assert seen == [1, 2]


async def test_pump_turns_a_terminal_engine_failure_into_engine_error() -> None:
    engine = FakeEngine()
    session, _ = make_session(engine)
    task = asyncio.create_task(session.pump_engine_events())
    await engine.fail_events()
    with pytest.raises(ev.ProtocolError) as excinfo:
        await task
    assert excinfo.value.code == "engine_error"


async def test_pump_returns_when_the_engine_has_no_event_stream() -> None:
    class BareEngine:
        ready = True

    session, _ = make_session(BareEngine())  # type: ignore[arg-type]
    await session.pump_engine_events()


# ---------------------------------------------------------------------------
# profile section 8.3: the state-dependent rejections this layer owns
# ---------------------------------------------------------------------------


async def _state_case(case: str) -> dict[str, Any]:
    if case == "append_audio_payload":
        session, _ = make_session()
        return await expect_error(session, {"type": "input_audio_buffer.append", "audio": "AAAAA"})
    if case == "commit_empty_buffer":
        session, _ = make_session()
        return await expect_error(session, {"type": "input_audio_buffer.commit"})
    if case == "response_create_in_progress":
        session, _ = make_session()
        session.active_response_id = "resp_1"
        return await expect_error(session, {"type": "response.create"})
    if case == "response_cancel_nothing":
        session, _ = make_session()
        return await expect_error(session, {"type": "response.cancel"})
    if case == "response_cancel_foreign_id":
        session, _ = make_session()
        session.active_response_id = "resp_1"
        return await expect_error(session, {"type": "response.cancel", "response_id": "resp_2"})
    if case == "item_id_duplicate":
        session, _ = make_session()
        await session.handle_raw(text_item(id="item_1"))
        return await expect_error(session, text_item(id="item_1"))
    if case == "item_call_id_duplicate":
        session, _ = make_session()
        await session.handle_raw(output_item("call_1"))
        return await expect_error(session, output_item("call_1"))
    if case == "truncate_unknown_item":
        session, _ = make_session()
        return await expect_error(
            session,
            {
                "type": "conversation.item.truncate",
                "item_id": "item_nope",
                "content_index": 0,
                "audio_end_ms": 0,
            },
        )
    if case == "delete_unknown_item":
        session, _ = make_session()
        return await expect_error(
            session, {"type": "conversation.item.delete", "item_id": "item_nope"}
        )
    if case == "engine_call_failed":
        session, _ = make_session(FakeEngine(fail_on={"flush_pending"}))
        await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(0.1)})
        return await expect_error(session, {"type": "response.create"})
    raise AssertionError(f"case {case!r} has no driver in this test module")


STATE_CASES: tuple[str, ...] = (
    "append_audio_payload",
    "commit_empty_buffer",
    "delete_unknown_item",
    "engine_call_failed",
    "item_call_id_duplicate",
    "item_id_duplicate",
    "response_cancel_foreign_id",
    "response_cancel_nothing",
    "response_create_in_progress",
    "truncate_unknown_item",
)


def _profile_state_cases() -> dict[str, tuple[str, str | None]]:
    """case id -> (code, param) as documented in section 8.3."""
    text = PROFILE.read_text(encoding="utf-8")
    start = text.index("### 8.3 State-dependent rejections")
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
    out: dict[str, tuple[str, str | None]] = {}
    for row in rows[1:]:
        case = row[0].strip().strip("`")
        param = row[3].strip().strip("`") or None
        out[case] = (row[2].strip().strip("`"), param)
    return out


@pytest.mark.parametrize("case", STATE_CASES)
async def test_state_rejection_matches_the_profile(case: str) -> None:
    documented = _profile_state_cases()
    assert case in documented, f"case {case!r} is not documented in section 8.3"
    code, param = documented[case]
    error = await _state_case(case)
    assert error["code"] == code
    assert error["param"] == param
    assert error["type"] == ev.ERROR_CODE_TYPES[code]
    assert error["message"]


def test_profile_state_cases_are_covered() -> None:
    assert set(_profile_state_cases()) == set(STATE_CASES)


# ---------------------------------------------------------------------------
# end to end over the WebSocket transport
# ---------------------------------------------------------------------------


def make_client(engine: FakeEngine) -> TestClient:
    config = TalkoverConfig(server=ServerConfig(listen="127.0.0.1:8000", api_key=API_KEY))
    return TestClient(create_app(config, engine, session_factory=WebSocketSession))


def test_websocket_session_drives_the_state_machine() -> None:
    engine = FakeEngine()
    with make_client(engine).websocket_connect(
        "/v1/realtime?model=gpt-realtime", headers=AUTH
    ) as ws:
        assert ws.receive_json()["type"] == "session.created"
        assert ws.receive_json()["type"] == "conversation.created"

        ws.send_json(session_update(instructions="Be brief."))
        updated = ws.receive_json()
        assert updated["type"] == "session.updated"
        assert updated["session"]["model"] == "gpt-realtime"
        assert updated["session"]["instructions"] == "Be brief."

        ws.send_json({"type": "input_audio_buffer.append", "audio": pcm24(1.0)})
        ws.send_json({"type": "input_audio_buffer.commit"})
        assert ws.receive_json()["type"] == "input_audio_buffer.committed"
        assert ws.receive_json()["type"] == "conversation.item.created"

        ws.send_bytes(b"\x00\x01")
        error = ws.receive_json()
        assert error["type"] == "error"
        assert error["error"]["code"] == "invalid_event"

        ws.send_json({"type": "nope"})
        assert ws.receive_json()["error"]["code"] == "invalid_value"

    assert engine.call_names == ["set_task_slate", "feed_pcm16", "flush_pending"]


def test_websocket_session_reports_a_fatal_engine_error_and_closes() -> None:
    engine = FakeEngine(fail_on={"submit_text_turn"})
    with make_client(engine).websocket_connect("/v1/realtime", headers=AUTH) as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_json(text_item("hello"))
        assert ws.receive_json()["type"] == "conversation.item.created"
        error = ws.receive_json()
        assert error["error"]["code"] == "engine_error"
        message = ws.receive()
        assert message["type"] == "websocket.close"
        assert message["code"] == 1011
