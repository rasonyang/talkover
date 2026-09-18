"""Validation, wire shapes and doc/test drift checks for `talkover.realtime.events`.

GPU-free; no pytest marker. The rejection cases are keyed by the case ids in
`docs/protocol-profile.md` section 8.2, and `test_profile_rejection_cases_are_covered`
fails if the document and this file drift apart.
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path
from typing import Any

import pytest

from talkover.realtime import events as ev

PROFILE = Path(__file__).resolve().parents[2] / "docs" / "protocol-profile.md"


# ---------------------------------------------------------------------------
# profile table parsing
# ---------------------------------------------------------------------------


def _table_rows(heading: str) -> list[list[str]]:
    """Return the data rows of the first markdown table after `heading`."""
    text = PROFILE.read_text(encoding="utf-8")
    start = text.index(heading)
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
    return rows[1:]


def _unquote(cell: str) -> str | None:
    value = cell.strip().strip("`")
    return value or None


def profile_codes() -> dict[str, str]:
    """`error.code` → `error.type` as documented in section 8.1."""
    return {
        str(_unquote(row[1])): str(_unquote(row[0])) for row in _table_rows("### 8.1 Rejection")
    }


def profile_cases() -> dict[str, tuple[str, str | None]]:
    """case id → (code, param) as documented in section 8.2."""
    return {
        str(_unquote(row[0])): (str(_unquote(row[2])), _unquote(row[3]))
        for row in _table_rows("### 8.2 Rejection cases")
    }


# ---------------------------------------------------------------------------
# rejection cases
# ---------------------------------------------------------------------------


def _session(**fields: Any) -> dict[str, Any]:
    return {"type": "session.update", "session": {"type": "realtime", **fields}}


def _audio_in(**fields: Any) -> dict[str, Any]:
    return _session(audio={"input": fields})


def _item(item: Any) -> dict[str, Any]:
    return {"type": "conversation.item.create", "item": item}


#: case id → the client frame that must be rejected.
REJECTION_FRAMES: dict[str, Any] = {
    "frame_not_json": "{not json",
    "frame_not_object": "[1, 2]",
    "frame_bad_utf8": b"\xff\xfe\x00",
    "missing_type": {"event_id": "evt_1"},
    "type_not_string": {"type": 5},
    "unknown_type": {"type": "scooby.dooby.doo"},
    "rejected_type_retrieve": {"type": "conversation.item.retrieve", "item_id": "item_1"},
    "rejected_type_output_audio_buffer_clear": {"type": "output_audio_buffer.clear"},
    "rejected_type_transcription_session": {"type": "transcription_session.update"},
    "rejected_type_beta_alias": {"type": "response.audio.delta"},
    "event_id_not_string": {"type": "input_audio_buffer.commit", "event_id": 7},
    "event_id_too_long": {"type": "input_audio_buffer.commit", "event_id": "e" * 513},
    "unknown_top_level_field": {"type": "input_audio_buffer.commit", "garbage": 1},
    "session_missing": {"type": "session.update"},
    "session_not_object": {"type": "session.update", "session": []},
    "session_type_missing": {"type": "session.update", "session": {"model": "x"}},
    "session_type_beta": {"type": "session.update", "session": {"type": "transcription"}},
    "session_unknown_field": _session(nope=1),
    "session_beta_field": _session(voice="alloy"),
    "session_instructions_type": _session(instructions=5),
    "session_modalities_text": _session(output_modalities=["text"]),
    "session_input_format_type": _audio_in(format={"type": "audio/opus"}),
    "session_input_format_rate": _audio_in(format={"type": "audio/pcm", "rate": 16000}),
    "session_output_format_type": _session(audio={"output": {"format": {"type": "audio/opus"}}}),
    "session_transcription_field": _audio_in(transcription={"delay": 1}),
    "session_noise_reduction_type": _audio_in(noise_reduction={"type": "mid_field"}),
    "session_speed_range": _session(audio={"output": {"speed": 3.0}}),
    "session_voice_type": _session(audio={"output": {"voice": 5}}),
    "turn_detection_type": _audio_in(turn_detection={"type": "magic_vad"}),
    "turn_detection_threshold": _audio_in(turn_detection={"type": "server_vad", "threshold": 2.0}),
    "turn_detection_prefix_padding": _audio_in(
        turn_detection={"type": "server_vad", "prefix_padding_ms": -1}
    ),
    "turn_detection_silence_duration": _audio_in(
        turn_detection={"type": "server_vad", "silence_duration_ms": 0}
    ),
    "turn_detection_eagerness": _audio_in(
        turn_detection={"type": "semantic_vad", "eagerness": "urgent"}
    ),
    "turn_detection_foreign_field": _audio_in(
        turn_detection={"type": "semantic_vad", "threshold": 0.5}
    ),
    "turn_detection_idle_timeout": _audio_in(
        turn_detection={"type": "server_vad", "idle_timeout_ms": 5000}
    ),
    "tool_type": _session(tools=[{"type": "mcp", "name": "t"}]),
    "tool_name_missing": _session(tools=[{"type": "function"}]),
    "tool_name_duplicate": _session(
        tools=[{"type": "function", "name": "a"}, {"type": "function", "name": "a"}]
    ),
    "tool_parameters_type": _session(tools=[{"type": "function", "name": "a", "parameters": "{}"}]),
    "tool_choice_value": _session(tool_choice="banana"),
    "tool_choice_unknown_name": _session(tool_choice={"type": "function", "name": "ghost"}),
    "session_max_output_tokens": _session(max_output_tokens=0),
    "session_truncation_value": _session(truncation="disabled"),
    "session_include_value": _session(include=["item.input_audio_transcription.logprobs"]),
    "append_audio_missing": {"type": "input_audio_buffer.append"},
    "append_audio_type": {"type": "input_audio_buffer.append", "audio": 5},
    "response_create_instructions": {"type": "response.create", "response": {"instructions": "x"}},
    "response_create_audio": {
        "type": "response.create",
        "response": {"audio": {"output": {"voice": "alloy"}}},
    },
    "response_create_tools": {"type": "response.create", "response": {"tools": []}},
    "response_create_conversation": {
        "type": "response.create",
        "response": {"conversation": "none"},
    },
    "response_create_metadata_size": {
        "type": "response.create",
        "response": {"metadata": {f"k{i}": "v" for i in range(17)}},
    },
    "response_create_unknown_field": {"type": "response.create", "response": {"nope": 1}},
    "response_cancel_id_type": {"type": "response.cancel", "response_id": 5},
    "item_missing": {"type": "conversation.item.create"},
    "item_type_missing": _item({"role": "user"}),
    "item_type_unknown": _item({"type": "banana"}),
    "item_type_function_call": _item(
        {"type": "function_call", "name": "f", "call_id": "call_1", "arguments": "{}"}
    ),
    "item_role_assistant": _item(
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "x"}]}
    ),
    "item_role_system": _item(
        {"type": "message", "role": "system", "content": [{"type": "input_text", "text": "x"}]}
    ),
    "item_content_multiple": _item(
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
        }
    ),
    "item_content_input_audio": _item(
        {"type": "message", "role": "user", "content": [{"type": "input_audio", "audio": "AA=="}]}
    ),
    "item_content_text_type": _item(
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": 5}]}
    ),
    "item_call_id_missing": _item({"type": "function_call_output", "output": "{}"}),
    "item_output_missing": _item({"type": "function_call_output", "call_id": "call_1"}),
    "truncate_missing_field": {
        "type": "conversation.item.truncate",
        "item_id": "item_1",
        "content_index": 0,
    },
    "truncate_content_index": {
        "type": "conversation.item.truncate",
        "item_id": "item_1",
        "content_index": 1,
        "audio_end_ms": 10,
    },
    "truncate_audio_end_ms": {
        "type": "conversation.item.truncate",
        "item_id": "item_1",
        "content_index": 0,
        "audio_end_ms": -1,
    },
    "delete_item_id_missing": {"type": "conversation.item.delete"},
}

#: case id → a `ProtocolError` raised outside `parse_client_event`.
RUNTIME_REJECTIONS: dict[str, ev.ProtocolError] = {
    "engine_busy": ev.ProtocolError("engine_busy", "A session is already running on this process."),
    "engine_error": ev.ProtocolError("engine_error", "The inference engine failed."),
}


def _raise_case(case: str) -> ev.ProtocolError:
    if case in RUNTIME_REJECTIONS:
        return RUNTIME_REJECTIONS[case]
    frame = REJECTION_FRAMES[case]
    with pytest.raises(ev.ProtocolError) as excinfo:
        ev.parse_client_event(frame)
    return excinfo.value


@pytest.mark.parametrize("case", sorted(set(REJECTION_FRAMES) | set(RUNTIME_REJECTIONS)))
def test_rejection_case_matches_profile(case: str) -> None:
    documented = profile_cases()
    assert case in documented, f"case {case!r} is not documented in section 8.2"
    code, param = documented[case]
    exc = _raise_case(case)
    assert exc.code == code
    assert exc.param == param
    assert exc.message

    wire = ev.error_event(exc)
    assert wire["type"] == "error"
    assert wire["event_id"].startswith("event_")
    assert wire["error"] == {
        "type": ev.ERROR_CODE_TYPES[code],
        "code": code,
        "message": exc.message,
        "param": param,
        "event_id": exc.event_id,
    }
    json.dumps(wire)


def test_profile_rejection_cases_are_covered() -> None:
    tested = set(REJECTION_FRAMES) | set(RUNTIME_REJECTIONS)
    documented = set(profile_cases())
    assert documented == tested


def test_profile_rejection_codes_are_covered() -> None:
    documented = profile_codes()
    assert documented == ev.ERROR_CODE_TYPES
    produced = {code for code, _ in profile_cases().values()}
    assert produced == set(documented)


def test_error_event_carries_the_client_event_id() -> None:
    frame = {"type": "conversation.item.delete", "event_id": "evt_42"}
    with pytest.raises(ev.ProtocolError) as excinfo:
        ev.parse_client_event(frame)
    assert excinfo.value.event_id == "evt_42"
    assert ev.error_event(excinfo.value)["error"]["event_id"] == "evt_42"


def test_unknown_error_code_is_refused() -> None:
    with pytest.raises(ValueError):
        ev.ProtocolError("teapot", "no such code")


# ---------------------------------------------------------------------------
# client event round trips
# ---------------------------------------------------------------------------

AUDIO_B64 = base64.b64encode(b"\x00\x01" * 8).decode()

ROUND_TRIPS: dict[str, dict[str, Any]] = {
    "session.update": {
        "type": "session.update",
        "event_id": "evt_1",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime",
            "instructions": "Be brief.",
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "whisper-1", "language": "zh"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcmu", "rate": 8000},
                    "voice": "alloy",
                    "speed": 1.0,
                },
            },
            "tools": [
                {
                    "type": "function",
                    "name": "transfer_to_human",
                    "description": "Transfer the call.",
                    "parameters": {"type": "object"},
                }
            ],
            "tool_choice": "auto",
            "max_output_tokens": "inf",
            "truncation": "auto",
            "tracing": None,
            "prompt": None,
            "include": [],
        },
    },
    "session.update.minimal": {"type": "session.update", "session": {"type": "realtime"}},
    "input_audio_buffer.append": {"type": "input_audio_buffer.append", "audio": AUDIO_B64},
    "input_audio_buffer.commit": {"type": "input_audio_buffer.commit", "event_id": "evt_2"},
    "input_audio_buffer.clear": {"type": "input_audio_buffer.clear"},
    "response.create": {"type": "response.create"},
    "response.create.options": {
        "type": "response.create",
        "response": {"conversation": "auto", "metadata": {"turn": "3"}},
    },
    "response.cancel": {"type": "response.cancel"},
    "response.cancel.id": {"type": "response.cancel", "response_id": "resp_1"},
    "conversation.item.create.message": {
        "type": "conversation.item.create",
        "item": {
            "id": "item_client_1",
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "查一下我的订单"}],
        },
        "previous_item_id": "item_root",
    },
    "conversation.item.create.function_call_output": {
        "type": "conversation.item.create",
        "item": {
            "type": "function_call_output",
            "call_id": "call_abc",
            "output": '{"status":"transferred"}',
        },
    },
    "conversation.item.truncate": {
        "type": "conversation.item.truncate",
        "item_id": "item_1",
        "content_index": 0,
        "audio_end_ms": 1200,
    },
    "conversation.item.delete": {"type": "conversation.item.delete", "item_id": "item_1"},
}


@pytest.mark.parametrize("name", sorted(ROUND_TRIPS))
def test_client_event_round_trip(name: str) -> None:
    frame = ROUND_TRIPS[name]
    parsed = ev.parse_client_event(frame)
    assert isinstance(parsed, ev.ClientEvent)
    assert parsed.TYPE == frame["type"]
    assert parsed.event_id == frame.get("event_id")
    wire = ev.to_wire(parsed)
    assert wire == frame
    assert ev.to_wire(ev.parse_client_event(json.dumps(frame))) == frame
    assert ev.to_wire(ev.parse_client_event(json.dumps(frame).encode())) == frame


def test_session_update_accessors() -> None:
    parsed = ev.parse_client_event(ROUND_TRIPS["session.update"])
    assert isinstance(parsed, ev.SessionUpdate)
    session = parsed.session
    assert session.model == "gpt-realtime"
    assert session.instructions == "Be brief."
    assert session.output_modalities == ("audio",)
    assert session.voice == "alloy"
    assert session.speed == 1.0
    assert session.transcription == {"model": "whisper-1", "language": "zh"}
    assert session.noise_reduction == {"type": "near_field"}
    assert session.turn_detection is not None
    assert session.turn_detection["type"] == "server_vad"
    assert [tool["name"] for tool in session.tools] == ["transfer_to_human"]
    assert session.tool_choice == "auto"
    assert ev.audio_format_name(session.input_audio_format) == "pcm16"
    assert ev.audio_format_name(session.output_audio_format) == "g711_ulaw"
    assert session.has("instructions")
    assert not session.has("nope")


def test_session_update_minimal_has_no_audio_section() -> None:
    parsed = ev.parse_client_event(ROUND_TRIPS["session.update.minimal"])
    assert isinstance(parsed, ev.SessionUpdate)
    session = parsed.session
    assert session.input_audio_format is None
    assert session.turn_detection is None
    assert session.tools == ()
    assert ev.audio_format_name(None) == "pcm16"


def test_turn_detection_null_is_accepted() -> None:
    parsed = ev.parse_client_event(_audio_in(turn_detection=None))
    assert isinstance(parsed, ev.SessionUpdate)
    assert parsed.session.turn_detection is None
    assert parsed.session.raw["audio"]["input"]["turn_detection"] is None


def test_g711_input_format_is_accepted() -> None:
    parsed = ev.parse_client_event(_audio_in(format={"type": "audio/pcma"}))
    assert isinstance(parsed, ev.SessionUpdate)
    assert parsed.session.input_audio_format == {"type": "audio/pcma", "rate": 8000}
    assert ev.audio_format_name(parsed.session.input_audio_format) == "g711_alaw"


def test_message_item_fields() -> None:
    parsed = ev.parse_client_event(ROUND_TRIPS["conversation.item.create.message"])
    assert isinstance(parsed, ev.ConversationItemCreate)
    item = parsed.item
    assert isinstance(item, ev.MessageItem)
    assert item.role == "user"
    assert item.text == "查一下我的订单"
    assert item.id == "item_client_1"
    assert parsed.previous_item_id == "item_root"


def test_function_call_output_item_fields() -> None:
    parsed = ev.parse_client_event(ROUND_TRIPS["conversation.item.create.function_call_output"])
    assert isinstance(parsed, ev.ConversationItemCreate)
    item = parsed.item
    assert isinstance(item, ev.FunctionCallOutputItem)
    assert item.call_id == "call_abc"
    assert item.output == '{"status":"transferred"}'
    assert item.id is None


def test_item_status_and_object_are_ignored() -> None:
    parsed = ev.parse_client_event(
        _item(
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "ok",
                "status": "completed",
                "object": "realtime.item",
            }
        )
    )
    assert isinstance(parsed, ev.ConversationItemCreate)
    assert dict(parsed.item.raw) == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "ok",
    }


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("factory", "prefix"),
    [
        (ev.new_event_id, "event"),
        (ev.new_item_id, "item"),
        (ev.new_response_id, "resp"),
        (ev.new_session_id, "sess"),
        (ev.new_conversation_id, "conv"),
        (ev.new_call_id, "call"),
    ],
)
def test_id_helpers(factory: Any, prefix: str) -> None:
    value = factory()
    assert re.fullmatch(rf"{prefix}_[a-z0-9]{{16}}", value)
    assert factory() != value


# ---------------------------------------------------------------------------
# instructions cap
# ---------------------------------------------------------------------------


def test_estimate_tokens_counts_cjk_per_character() -> None:
    assert ev.estimate_tokens("") == 0
    assert ev.estimate_tokens("转人工") == 3
    assert ev.estimate_tokens("abcd") == 1
    assert ev.estimate_tokens("转人工abcd") == 4


def test_truncate_instructions_respects_the_cap() -> None:
    assert ev.truncate_instructions("短", 256) == "短"
    long_text = "转" * 400
    truncated = ev.truncate_instructions(long_text, 256)
    assert truncated == "转" * 256
    assert ev.estimate_tokens(truncated) <= 256
    assert ev.truncate_instructions("anything", 0) == ""


def test_truncate_instructions_is_maximal() -> None:
    text = "a" * 100
    truncated = ev.truncate_instructions(text, 10)
    assert ev.estimate_tokens(truncated) <= 10
    assert ev.estimate_tokens(text[: len(truncated) + 1]) > 10


# ---------------------------------------------------------------------------
# server event wire shapes
# ---------------------------------------------------------------------------

ITEM = {
    "id": "item_1",
    "object": "realtime.item",
    "type": "message",
    "role": "assistant",
    "status": "in_progress",
    "content": [{"type": "output_audio", "transcript": ""}],
}
RESPONSE = {
    "id": "resp_1",
    "object": "realtime.response",
    "status": "in_progress",
    "status_details": None,
    "output": [],
    "usage": None,
}

SERVER_EVENTS: list[tuple[ev.ServerEvent, str, dict[str, Any]]] = [
    (ev.SessionCreated(session={"id": "sess_1"}), "session.created", {"session": {"id": "sess_1"}}),
    (ev.SessionUpdated(session={"id": "sess_1"}), "session.updated", {"session": {"id": "sess_1"}}),
    (
        ev.ConversationCreated(conversation={"id": "conv_1", "object": "realtime.conversation"}),
        "conversation.created",
        {"conversation": {"id": "conv_1", "object": "realtime.conversation"}},
    ),
    (
        ev.InputAudioBufferCommitted(item_id="item_1", previous_item_id=None),
        "input_audio_buffer.committed",
        {"item_id": "item_1", "previous_item_id": None},
    ),
    (ev.InputAudioBufferCleared(), "input_audio_buffer.cleared", {}),
    (
        ev.InputAudioBufferSpeechStarted(audio_start_ms=120, item_id="item_1"),
        "input_audio_buffer.speech_started",
        {"audio_start_ms": 120, "item_id": "item_1"},
    ),
    (
        ev.InputAudioBufferSpeechStopped(audio_end_ms=980, item_id="item_1"),
        "input_audio_buffer.speech_stopped",
        {"audio_end_ms": 980, "item_id": "item_1"},
    ),
    (
        ev.ConversationItemCreated(item=ITEM, previous_item_id="item_0"),
        "conversation.item.created",
        {"item": ITEM, "previous_item_id": "item_0"},
    ),
    (
        ev.ConversationItemTruncated(item_id="item_1", content_index=0, audio_end_ms=800),
        "conversation.item.truncated",
        {"item_id": "item_1", "content_index": 0, "audio_end_ms": 800},
    ),
    (
        ev.ConversationItemDeleted(item_id="item_1"),
        "conversation.item.deleted",
        {"item_id": "item_1"},
    ),
    (
        ev.ConversationItemInputAudioTranscriptionCompleted(
            item_id="item_1",
            content_index=0,
            transcript="你好",
            usage={"type": "duration", "seconds": 1},
        ),
        "conversation.item.input_audio_transcription.completed",
        {
            "item_id": "item_1",
            "content_index": 0,
            "transcript": "你好",
            "usage": {"type": "duration", "seconds": 1},
        },
    ),
    (ev.ResponseCreated(response=RESPONSE), "response.created", {"response": RESPONSE}),
    (
        ev.ResponseOutputItemAdded(response_id="resp_1", output_index=0, item=ITEM),
        "response.output_item.added",
        {"response_id": "resp_1", "output_index": 0, "item": ITEM},
    ),
    (
        ev.ResponseOutputItemDone(response_id="resp_1", output_index=0, item=ITEM),
        "response.output_item.done",
        {"response_id": "resp_1", "output_index": 0, "item": ITEM},
    ),
    (
        ev.ResponseContentPartAdded(
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            part={"type": "audio", "transcript": ""},
        ),
        "response.content_part.added",
        {
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "audio", "transcript": ""},
        },
    ),
    (
        ev.ResponseContentPartDone(
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            part={"type": "audio", "transcript": "你好"},
        ),
        "response.content_part.done",
        {
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "audio", "transcript": "你好"},
        },
    ),
    (
        ev.ResponseOutputAudioDelta(
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            delta=AUDIO_B64,
        ),
        "response.output_audio.delta",
        {
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": AUDIO_B64,
        },
    ),
    (
        ev.ResponseOutputAudioDone(
            response_id="resp_1", item_id="item_1", output_index=0, content_index=0
        ),
        "response.output_audio.done",
        {"response_id": "resp_1", "item_id": "item_1", "output_index": 0, "content_index": 0},
    ),
    (
        ev.ResponseOutputAudioTranscriptDelta(
            response_id="resp_1", item_id="item_1", output_index=0, content_index=0, delta="你"
        ),
        "response.output_audio_transcript.delta",
        {
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "delta": "你",
        },
    ),
    (
        ev.ResponseOutputAudioTranscriptDone(
            response_id="resp_1",
            item_id="item_1",
            output_index=0,
            content_index=0,
            transcript="你好",
        ),
        "response.output_audio_transcript.done",
        {
            "response_id": "resp_1",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "transcript": "你好",
        },
    ),
    (
        ev.ResponseFunctionCallArgumentsDelta(
            response_id="resp_1",
            item_id="item_2",
            output_index=1,
            call_id="call_1",
            delta='{"department":"sales"}',
        ),
        "response.function_call_arguments.delta",
        {
            "response_id": "resp_1",
            "item_id": "item_2",
            "output_index": 1,
            "call_id": "call_1",
            "delta": '{"department":"sales"}',
        },
    ),
    (
        ev.ResponseFunctionCallArgumentsDone(
            response_id="resp_1",
            item_id="item_2",
            output_index=1,
            call_id="call_1",
            name="transfer_to_human",
            arguments='{"department":"sales"}',
        ),
        "response.function_call_arguments.done",
        {
            "response_id": "resp_1",
            "item_id": "item_2",
            "output_index": 1,
            "call_id": "call_1",
            "name": "transfer_to_human",
            "arguments": '{"department":"sales"}',
        },
    ),
    (
        ev.ResponseDone(response={**RESPONSE, "status": "cancelled"}),
        "response.done",
        {"response": {**RESPONSE, "status": "cancelled"}},
    ),
    (
        ev.ErrorEvent(error={"type": "server_error", "code": "engine_error"}),
        "error",
        {"error": {"type": "server_error", "code": "engine_error"}},
    ),
]


@pytest.mark.parametrize(
    ("event", "kind", "payload"), SERVER_EVENTS, ids=[case[1] for case in SERVER_EVENTS]
)
def test_server_event_wire_shape(event: ev.ServerEvent, kind: str, payload: dict[str, Any]) -> None:
    wire = ev.to_wire(event)
    assert wire.pop("type") == kind
    assert wire.pop("event_id").startswith("event_")
    assert wire == payload
    json.dumps(ev.to_wire(event))


def test_server_events_use_ga_names_only() -> None:
    names = {kind for _, kind, _ in SERVER_EVENTS}
    assert "response.output_audio.delta" in names
    assert not any(name.startswith(("response.audio.", "response.text.")) for name in names)


def test_every_documented_server_event_is_modelled() -> None:
    documented = {str(_unquote(row[0])) for row in _table_rows("## 9. Server events")}
    # The two `.done` audio events share one documented row.
    documented.discard("response.output_audio.done` / `response.output_audio_transcript.done")
    documented |= {"response.output_audio.done", "response.output_audio_transcript.done"}
    modelled = {kind for _, kind, _ in SERVER_EVENTS}
    assert documented <= modelled


def test_event_ids_are_unique_per_event() -> None:
    first = ev.InputAudioBufferCleared()
    second = ev.InputAudioBufferCleared()
    assert first.event_id != second.event_id
