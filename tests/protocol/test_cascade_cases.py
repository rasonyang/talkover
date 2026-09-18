"""Ported Cascade Realtime client regression cases, driven from the profile (T2.7).

`docs/protocol-profile.md` is the ported form of `cascade-realtime-gateway`'s protocol
profile: documentation and cases only, no code. This module turns it into a regression
suite. Every rule the profile states as a table row — the `session.update` echo rules of
section 3, the audio buffer and response rows of section 2, the `response.create`
overrides of section 4, the item subset of section 5, the turn detection rules of section
7, the error codes of section 8 and the server-event sequences of section 9 — has a case
here, and the cases run end to end over the real application: `create_app` with
`session_factory=WebSocketSession`, a fake engine, and a WebSocket client.

Two things are machine-read from the profile, so the document and the suite cannot drift:

- the effective-session JSON block of section 3, compared against `session.created`;
- the "Never emitted" list of section 9, checked against a full turn;
- section 11.1, the skipped-case table, compared against `SKIPPED_CASES`.

Cases that exist only in Cascade — real VAD boundaries, many sessions per process, the
`conversation.item.added` / `.done` lifecycle, the text-response flow, Cascade's server
error codes — cannot be ported: they are listed in `SKIPPED_CASES` with their reason and
in section 11.1 of the profile.

GPU-free: the engine is `tests/protocol/fake_engine.FakeEngine`.
"""

from __future__ import annotations

import base64
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from talkover.config import TalkoverConfig
from talkover.engine.asr import Transcript
from talkover.engine.protocol import EngineStepEvent
from talkover.realtime import events as ev
from talkover.realtime.audio import ulaw_decode
from talkover.realtime.server import create_app
from talkover.realtime.session import WebSocketSession

from .fake_engine import FakeEngine, step_event
from .test_session import make_session, pcm24, session_update, text_item

PROFILE = Path(__file__).resolve().parents[2] / "docs" / "protocol-profile.md"

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

OPEN = (
    "response.created",
    "response.output_item.added",
    "conversation.item.created",
    "response.content_part.added",
)
CLOSE = (
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.content_part.done",
    "response.output_item.done",
)
TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
AUDIO_DELTA = "response.output_audio.delta"
SPEECH_STARTED = "input_audio_buffer.speech_started"


# ---------------------------------------------------------------------------
# case model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Step:
    """One synchronisation point: send these frames, then read exactly these events.

    Reading a step's events before the next step's frames go out is what makes an
    engine-driven case deterministic: the engine units of step N have reached the client
    before the client event of step N+1 is handled, so a rejection that depends on
    response state (`response.create` while streaming) always sees that state.
    """

    send: tuple[Any, ...] = ()
    expect: tuple[str, ...] = ()


@dataclass(frozen=True)
class Outcome:
    """What a case produced: the handshake, every event read, and the engine."""

    handshake: list[dict[str, Any]]
    events: list[dict[str, Any]]
    engine: FakeEngine

    @property
    def session(self) -> dict[str, Any]:
        """The session object of `session.created`."""
        return dict(self.handshake[0]["session"])

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    def one(self, kind: str) -> dict[str, Any]:
        matches = self.of(kind)
        assert len(matches) == 1, f"expected one {kind}, got {[e['type'] for e in self.events]}"
        return matches[0]

    @property
    def errors(self) -> list[dict[str, Any]]:
        return [dict(event["error"]) for event in self.of("error")]

    @property
    def updated(self) -> dict[str, Any]:
        """The session object of the last `session.updated`."""
        return dict(self.of("session.updated")[-1]["session"])


@dataclass(frozen=True)
class Case:
    """One ported regression case, run over the real WebSocket application."""

    id: str
    ref: str
    steps: tuple[Step, ...] = ()
    engine_calls: tuple[str, ...] | None = None
    check: Callable[[Outcome], None] | None = None
    engine: Callable[[], FakeEngine] | None = None
    query: str = ""
    headers: Mapping[str, str] = field(default_factory=dict)
    xfail: str = ""


def case(
    id: str,
    ref: str,
    *,
    send: Sequence[Any] = (),
    expect: Sequence[str] = (),
    steps: Sequence[Step] = (),
    **kwargs: Any,
) -> Case:
    """Build a `Case`; `send` / `expect` are the common single-step form."""
    if steps:
        assert not send and not expect, "pass either `steps` or `send`/`expect`"
        return Case(id=id, ref=ref, steps=tuple(steps), **kwargs)
    return Case(id=id, ref=ref, steps=(Step(tuple(send), tuple(expect)),), **kwargs)


# ---------------------------------------------------------------------------
# frame and check helpers
# ---------------------------------------------------------------------------


def append(seconds: float = 1.0) -> dict[str, Any]:
    return {"type": "input_audio_buffer.append", "audio": pcm24(seconds)}


COMMIT = {"type": "input_audio_buffer.commit"}
CLEAR = {"type": "input_audio_buffer.clear"}
RESPONSE_CREATE: dict[str, Any] = {"type": "response.create"}


def response_create(**response_fields: Any) -> dict[str, Any]:
    return {"type": "response.create", "response": dict(response_fields)}


def item_create(item: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    return {"type": "conversation.item.create", "item": dict(item), **extra}


def output_item(call_id: str = "call_1", output: str = '{"ok": true}') -> dict[str, Any]:
    return item_create({"type": "function_call_output", "call_id": call_id, "output": output})


def wave(seconds: float = 0.1, value: float = 0.5) -> np.ndarray:
    """One chunk of model audio: float32 at the engine's 24 kHz output rate."""
    return np.full(int(24000 * seconds), value, dtype=np.float32)


def speak(index: int, **overrides: Any) -> EngineStepEvent:
    """A unit the model speaks in (`is_listen: false`)."""
    return step_event(index, is_listen=False, **overrides)


def scripted(**on_call: Sequence[Sequence[EngineStepEvent]]) -> Callable[[], FakeEngine]:
    """An engine factory whose `on_call` script is rebuilt for every run."""
    return lambda: FakeEngine(on_call={name: list(value) for name, value in on_call.items()})


def rejects(code: str, param: str | None = None, *, event_id: str | None = None) -> Any:
    """Check that the case produced exactly one `error` with this code and `param`."""

    def check(outcome: Outcome) -> None:
        assert len(outcome.errors) == 1, outcome.errors
        error = outcome.errors[0]
        assert error["code"] == code, error
        assert error["param"] == param, error
        assert error["type"] == ev.ERROR_CODE_TYPES[code], error
        assert error["message"], error
        assert error["event_id"] == event_id, error

    return check


def echoes(**expected: Any) -> Any:
    """Check dotted paths against the session object of the last `session.updated`."""

    def check(outcome: Outcome) -> None:
        session = outcome.updated
        for path, value in expected.items():
            assert _dig(session, path) == value, (path, session)

    return check


def _dig(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("__"):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Missing)

    __hash__ = None  # type: ignore[assignment]


#: Sentinel for `echoes(...)`: the field must not be present in the echo at all.
_MISSING = _Missing()
MISSING = _MISSING


# ---------------------------------------------------------------------------
# the cases
# ---------------------------------------------------------------------------

TRANSPORT_CASES: tuple[Case, ...] = (
    case(
        "transport_handshake_order",
        "§1 first server events",
        check=lambda o: (
            (
                [e["type"] for e in o.handshake] == ["session.created", "conversation.created"]
                and o.handshake[1]["conversation"]["object"] == "realtime.conversation"
            )
            or _fail(o)
        ),
    ),
    case(
        "transport_model_query_is_echoed",
        "§1 ?model=",
        query="?model=gpt-realtime",
        check=lambda o: _eq(o.session["model"], "gpt-realtime"),
    ),
    case(
        "transport_model_absent_echoes_talkover",
        "§1 ?model=",
        check=lambda o: _eq(o.session["model"], "talkover"),
    ),
    case(
        "transport_openai_beta_header_is_ignored",
        "§1 OpenAI-Beta header",
        headers={"OpenAI-Beta": "realtime=v1"},
        check=lambda o: (
            _eq(o.session["output_modalities"], ["audio"]) and _eq("modalities" in o.session, False)
        ),
    ),
    case(
        "transport_binary_frame_is_rejected",
        "§1 message framing",
        send=(b"\x00\x01\x02",),
        expect=("error",),
        check=rejects("invalid_event", None),
        engine_calls=(),
    ),
    case(
        "transport_frame_not_json",
        "§8.2 frame_not_json",
        send=("{not json",),
        expect=("error",),
        check=rejects("invalid_event", None),
    ),
    case(
        "transport_frame_not_object",
        "§8.2 frame_not_object",
        send=("[1, 2]",),
        expect=("error",),
        check=rejects("invalid_event", None),
    ),
    case(
        "transport_missing_type",
        "§2 missing type",
        send=({"session": {}},),
        expect=("error",),
        check=rejects("invalid_event", None),
    ),
    case(
        "transport_unknown_type",
        "§2 any other type",
        send=({"type": "nope"},),
        expect=("error",),
        check=rejects("invalid_value", "type"),
    ),
    case(
        "transport_event_id_is_echoed_on_the_error",
        "§2 event_id",
        send=({"type": "session.update", "event_id": "e1"},),
        expect=("error",),
        check=rejects("invalid_value", "session", event_id="e1"),
    ),
    case(
        "transport_event_id_too_long",
        "§8.2 event_id_too_long",
        send=({"type": "input_audio_buffer.clear", "event_id": "x" * 513},),
        expect=("error",),
        check=rejects("invalid_value", "event_id"),
    ),
    case(
        "transport_rejection_keeps_the_session_open",
        "§0 rejected",
        steps=(
            Step(("{not json",), ("error",)),
            Step((session_update(instructions="ok"),), ("session.updated",)),
        ),
        check=echoes(instructions="ok"),
    ),
)


CLIENT_EVENT_CASES: tuple[Case, ...] = (
    case(
        "client_session_update_echoes_the_effective_session",
        "§2 session.update",
        send=(session_update(instructions="Be brief."),),
        expect=("session.updated",),
        engine_calls=("set_task_slate",),
        check=echoes(
            type="realtime",
            instructions="Be brief.",
            output_modalities=["audio"],
            tools=[],
            tool_choice="auto",
            max_output_tokens="inf",
            truncation="auto",
            tracing=None,
            prompt=None,
        ),
    ),
    case(
        "client_session_update_rejects_the_whole_event",
        "§2 session.update",
        steps=(
            Step((session_update(instructions="kept", nope=1),), ("error",)),
            Step((session_update(),), ("session.updated",)),
        ),
        engine_calls=(),
        check=lambda o: (
            _eq(o.errors[0]["param"], "session.nope") and _eq(o.updated["instructions"], "")
        ),
    ),
    case(
        "client_append_is_not_acknowledged",
        "§2 input_audio_buffer.append",
        send=(append(0.5),),
        expect=(),
        engine_calls=(),
    ),
    case(
        "client_append_feeds_one_second_units",
        "§2 input_audio_buffer.append",
        send=(append(1.0),),
        expect=(),
        engine_calls=("feed_pcm16",),
    ),
    case(
        "client_append_undecodable_payload",
        "§8.3 append_audio_payload",
        send=({"type": "input_audio_buffer.append", "audio": "AAAAA"},),
        expect=("error",),
        engine_calls=(),
        check=rejects("invalid_value", "audio"),
    ),
    case(
        "client_commit_sequence",
        "§2 input_audio_buffer.commit",
        send=(append(1.0), COMMIT),
        expect=("input_audio_buffer.committed", "conversation.item.created"),
        engine_calls=("feed_pcm16", "flush_pending"),
        check=lambda o: _commit_shape(o),
    ),
    case(
        "client_commit_pads_the_partial_unit",
        "§2 input_audio_buffer.commit",
        send=(append(0.5), COMMIT),
        expect=("input_audio_buffer.committed", "conversation.item.created"),
        engine_calls=("feed_pcm16", "flush_pending"),
    ),
    case(
        "client_commit_on_an_empty_buffer",
        "§8.3 commit_empty_buffer",
        send=(COMMIT,),
        expect=("error",),
        engine_calls=(),
        check=rejects("invalid_event", None),
    ),
    case(
        "client_clear_drops_the_partial_unit",
        "§2 input_audio_buffer.clear",
        steps=(
            Step((append(0.5), CLEAR), ("input_audio_buffer.cleared",)),
            Step((COMMIT,), ("error",)),
        ),
        engine_calls=(),
        check=lambda o: _eq(o.errors[0]["code"], "invalid_event"),
    ),
    case(
        "client_item_create_text_turn",
        "§2 conversation.item.create",
        send=(text_item("hello"),),
        expect=("conversation.item.created",),
        engine_calls=("submit_text_turn",),
        check=lambda o: _eq(o.engine.argument("submit_text_turn"), "hello"),
    ),
    case(
        "client_item_truncate_is_acknowledged_only",
        "§2 conversation.item.truncate",
        send=(
            text_item("hi", id="item_a"),
            {
                "type": "conversation.item.truncate",
                "item_id": "item_a",
                "content_index": 0,
                "audio_end_ms": 120,
            },
        ),
        expect=("conversation.item.created", "conversation.item.truncated"),
        check=lambda o: (
            _eq(o.one("conversation.item.truncated")["audio_end_ms"], 120)
            and _eq(o.one("conversation.item.truncated")["content_index"], 0)
        ),
    ),
    case(
        "client_item_truncate_content_index",
        "§8.2 truncate_content_index",
        send=(
            text_item("hi", id="item_a"),
            {
                "type": "conversation.item.truncate",
                "item_id": "item_a",
                "content_index": 1,
                "audio_end_ms": 0,
            },
        ),
        expect=("conversation.item.created", "error"),
        check=rejects("invalid_value", "content_index"),
    ),
    case(
        "client_item_delete_is_acknowledged",
        "§2 conversation.item.delete",
        send=(
            text_item("hi", id="item_a"),
            {"type": "conversation.item.delete", "item_id": "item_a"},
        ),
        expect=("conversation.item.created", "conversation.item.deleted"),
    ),
    case(
        "client_item_delete_unknown_id",
        "§8.3 delete_unknown_item",
        send=({"type": "conversation.item.delete", "item_id": "item_missing"},),
        expect=("error",),
        check=rejects("invalid_value", "item_id"),
    ),
    case(
        "client_response_create_is_a_flush_pending",
        "§2 response.create",
        send=(append(1.0), RESPONSE_CREATE),
        expect=(),
        engine_calls=("feed_pcm16", "flush_pending"),
    ),
    case(
        "client_response_create_opens_when_the_model_speaks",
        "§2 response.create",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
        check=lambda o: _eq(o.one("response.done")["response"]["status"], "completed"),
    ),
    case(
        "client_response_create_while_streaming",
        "§8.3 response_create_in_progress",
        engine=scripted(flush_pending=[[speak(1, text="Hi")]]),
        steps=(
            Step((append(1.0), RESPONSE_CREATE), (*OPEN, TRANSCRIPT_DELTA)),
            Step((RESPONSE_CREATE,), ("error",)),
        ),
        engine_calls=("feed_pcm16", "flush_pending"),
        check=rejects("invalid_event", None),
    ),
    case(
        "client_response_cancel_flow",
        "§2 response.cancel",
        engine=scripted(
            flush_pending=[[speak(1, text="Hi")]],
            interrupt_output=[[speak(2, interrupted=True)]],
        ),
        steps=(
            Step((append(1.0), RESPONSE_CREATE), (*OPEN, TRANSCRIPT_DELTA)),
            Step(
                ({"type": "response.cancel"},),
                (SPEECH_STARTED, *CLOSE, "conversation.item.truncated", "response.done"),
            ),
        ),
        engine_calls=("feed_pcm16", "flush_pending", "interrupt_output"),
        check=lambda o: _cancelled(o, "client_cancelled"),
    ),
    case(
        "client_response_cancel_with_nothing_to_cancel",
        "§8.3 response_cancel_nothing",
        send=({"type": "response.cancel"},),
        expect=("error",),
        engine_calls=(),
        check=rejects("invalid_event", None),
    ),
    case(
        "client_response_cancel_foreign_id",
        "§8.3 response_cancel_foreign_id",
        engine=scripted(flush_pending=[[speak(1, text="Hi")]]),
        steps=(
            Step((append(1.0), RESPONSE_CREATE), (*OPEN, TRANSCRIPT_DELTA)),
            Step(({"type": "response.cancel", "response_id": "resp_other"},), ("error",)),
        ),
        check=rejects("invalid_value", "response_id"),
    ),
    case(
        "client_item_retrieve_is_rejected",
        "§2 conversation.item.retrieve",
        send=({"type": "conversation.item.retrieve", "item_id": "item_a"},),
        expect=("error",),
        check=rejects("invalid_event", "type"),
    ),
    case(
        "client_output_audio_buffer_clear_is_rejected",
        "§2 output_audio_buffer.clear",
        send=({"type": "output_audio_buffer.clear"},),
        expect=("error",),
        check=rejects("invalid_event", "type"),
    ),
    case(
        "client_transcription_session_update_is_rejected",
        "§2 transcription_session.update",
        send=({"type": "transcription_session.update", "session": {}},),
        expect=("error",),
        check=rejects("invalid_event", "type"),
    ),
    case(
        "client_beta_event_name_is_rejected",
        "§2 beta event names",
        send=({"type": "response.audio.delta"},),
        expect=("error",),
        check=rejects("invalid_event", "type"),
    ),
)


SESSION_FIELD_CASES: tuple[Case, ...] = (
    case(
        "session_type_is_required",
        "§3 type",
        send=({"type": "session.update", "session": {"instructions": "x"}},),
        expect=("error",),
        check=rejects("invalid_value", "session.type"),
    ),
    case(
        "session_type_must_be_realtime",
        "§3 type",
        send=({"type": "session.update", "session": {"type": "transcription"}},),
        expect=("error",),
        check=rejects("invalid_value", "session.type"),
    ),
    case(
        "session_model_is_ignored_but_echoed",
        "§3 model",
        send=(session_update(model="my-model"),),
        expect=("session.updated",),
        check=echoes(model="my-model"),
    ),
    case(
        "session_instructions_are_truncated_and_the_truncated_value_echoed",
        "§3 instructions / §6",
        send=(session_update(instructions="a" * 2000),),
        expect=("session.updated",),
        engine_calls=("set_task_slate",),
        check=lambda o: (
            _eq(len(o.updated["instructions"]), 1024)
            and _eq(o.engine.argument("set_task_slate"), o.updated["instructions"])
        ),
    ),
    case(
        "session_output_modalities_audio_only",
        "§3 output_modalities",
        send=(session_update(output_modalities=["audio"]),),
        expect=("session.updated",),
        check=echoes(output_modalities=["audio"]),
    ),
    case(
        "session_output_modalities_text_is_rejected",
        "§3 output_modalities",
        send=(session_update(output_modalities=["text"]),),
        expect=("error",),
        check=rejects("invalid_value", "session.output_modalities"),
    ),
    case(
        "session_input_format_pcmu",
        "§3 audio.input.format",
        send=(session_update(audio={"input": {"format": {"type": "audio/pcmu", "rate": 8000}}}),),
        expect=("session.updated",),
        check=echoes(audio__input__format={"type": "audio/pcmu", "rate": 8000}),
    ),
    case(
        "session_input_format_wrong_rate",
        "§8.2 session_input_format_rate",
        send=(session_update(audio={"input": {"format": {"type": "audio/pcm", "rate": 8000}}}),),
        expect=("error",),
        check=rejects("invalid_value", "session.audio.input.format.rate"),
    ),
    case(
        "session_input_transcription_object",
        "§3 audio.input.transcription",
        send=(
            session_update(
                audio={"input": {"transcription": {"model": "whisper-1", "language": "zh"}}}
            ),
        ),
        expect=("session.updated",),
        check=echoes(audio__input__transcription={"model": "whisper-1", "language": "zh"}),
    ),
    case(
        "session_input_transcription_null_clears",
        "§3 merge semantics",
        steps=(
            Step(
                (session_update(audio={"input": {"transcription": {"model": "whisper-1"}}}),),
                ("session.updated",),
            ),
            Step((session_update(audio={"input": {"transcription": None}}),), ("session.updated",)),
        ),
        check=echoes(audio__input__transcription=None),
    ),
    case(
        "session_input_transcription_unknown_field",
        "§8.2 session_transcription_field",
        send=(session_update(audio={"input": {"transcription": {"delay": 1}}}),),
        expect=("error",),
        check=rejects("invalid_value", "session.audio.input.transcription.delay"),
    ),
    case(
        "session_noise_reduction_is_ignored_but_echoed",
        "§3 audio.input.noise_reduction",
        send=(session_update(audio={"input": {"noise_reduction": {"type": "near_field"}}}),),
        expect=("session.updated",),
        check=echoes(audio__input__noise_reduction={"type": "near_field"}),
    ),
    case(
        "session_turn_detection_semantic_vad",
        "§3 audio.input.turn_detection / §7",
        send=(
            session_update(
                audio={
                    "input": {
                        "turn_detection": {
                            "type": "semantic_vad",
                            "eagerness": "high",
                            "create_response": True,
                            "interrupt_response": False,
                        }
                    }
                }
            ),
        ),
        expect=("session.updated",),
        check=echoes(
            audio__input__turn_detection={
                "type": "semantic_vad",
                "eagerness": "high",
                "create_response": True,
                "interrupt_response": False,
            }
        ),
    ),
    case(
        "session_turn_detection_null",
        "§3 merge semantics / §7",
        send=(session_update(audio={"input": {"turn_detection": None}}),),
        expect=("session.updated",),
        check=echoes(audio__input__turn_detection=None),
    ),
    case(
        "session_turn_detection_foreign_field",
        "§8.2 turn_detection_foreign_field",
        send=(
            session_update(
                audio={"input": {"turn_detection": {"type": "semantic_vad", "threshold": 0.5}}}
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "session.audio.input.turn_detection.threshold"),
    ),
    case(
        "session_turn_detection_idle_timeout",
        "§7 idle_timeout_ms",
        send=(
            session_update(
                audio={"input": {"turn_detection": {"type": "server_vad", "idle_timeout_ms": 5000}}}
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "session.audio.input.turn_detection.idle_timeout_ms"),
    ),
    case(
        "session_output_format_pcma",
        "§3 audio.output.format",
        send=(session_update(audio={"output": {"format": {"type": "audio/pcma", "rate": 8000}}}),),
        expect=("session.updated",),
        check=echoes(audio__output__format={"type": "audio/pcma", "rate": 8000}),
    ),
    case(
        "session_voice_is_ignored_but_echoed",
        "§3 audio.output.voice",
        send=(session_update(audio={"output": {"voice": "marin"}}),),
        expect=("session.updated",),
        check=echoes(audio__output__voice="marin"),
    ),
    case(
        "session_speed_is_ignored_but_echoed",
        "§3 audio.output.speed",
        send=(session_update(audio={"output": {"speed": 1.25}}),),
        expect=("session.updated",),
        check=echoes(audio__output__speed=1.25),
    ),
    case(
        "session_speed_out_of_range",
        "§8.2 session_speed_range",
        send=(session_update(audio={"output": {"speed": 2.0}}),),
        expect=("error",),
        check=rejects("invalid_value", "session.audio.output.speed"),
    ),
    case(
        "session_max_output_tokens_is_ignored_but_echoed",
        "§3 max_output_tokens",
        send=(session_update(max_output_tokens=512),),
        expect=("session.updated",),
        check=echoes(max_output_tokens=512),
    ),
    case(
        "session_tools_are_echoed",
        "§3 tools",
        send=(
            session_update(
                tools=[{"type": "function", "name": "transfer_to_human", "description": "escalate"}]
            ),
        ),
        expect=("session.updated",),
        check=lambda o: _eq([tool["name"] for tool in o.updated["tools"]], ["transfer_to_human"]),
    ),
    case(
        "session_tools_duplicate_name",
        "§8.2 tool_name_duplicate",
        send=(
            session_update(
                tools=[{"type": "function", "name": "a"}, {"type": "function", "name": "a"}]
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "session.tools[1].name"),
    ),
    case(
        "session_tool_choice_is_ignored_but_echoed",
        "§3 tool_choice",
        send=(session_update(tool_choice="required"),),
        expect=("session.updated",),
        check=echoes(tool_choice="required"),
    ),
    case(
        "session_tool_choice_unknown_function",
        "§8.2 tool_choice_unknown_name",
        send=(session_update(tool_choice={"type": "function", "name": "nope"}),),
        expect=("error",),
        check=rejects("invalid_value", "session.tool_choice.name"),
    ),
    case(
        "session_truncation_auto",
        "§3 truncation",
        send=(session_update(truncation="auto"),),
        expect=("session.updated",),
        check=echoes(truncation="auto"),
    ),
    case(
        "session_truncation_other_value",
        "§8.2 session_truncation_value",
        send=(session_update(truncation={"type": "retention_ratio"}),),
        expect=("error",),
        check=rejects("invalid_value", "session.truncation"),
    ),
    case(
        "session_tracing_null",
        "§3 tracing",
        send=(session_update(tracing=None),),
        expect=("session.updated",),
        check=echoes(tracing=None),
    ),
    case(
        "session_tracing_non_null",
        "§3 tracing",
        send=(session_update(tracing="auto"),),
        expect=("error",),
        check=rejects("invalid_value", "session.tracing"),
    ),
    case(
        "session_prompt_null",
        "§3 prompt",
        send=(session_update(prompt=None),),
        expect=("session.updated",),
        check=echoes(prompt=None),
    ),
    case(
        "session_prompt_non_null",
        "§3 prompt",
        send=(session_update(prompt={"id": "pmpt_1"}),),
        expect=("error",),
        check=rejects("invalid_value", "session.prompt"),
    ),
    case(
        "session_include_empty_is_not_echoed",
        "§3 include",
        send=(session_update(include=[]),),
        expect=("session.updated",),
        check=echoes(include=MISSING),
    ),
    case(
        "session_include_other_value",
        "§8.2 session_include_value",
        send=(session_update(include=["item.input_audio_transcription.logprobs"]),),
        expect=("error",),
        check=rejects("invalid_value", "session.include"),
    ),
    case(
        "session_unknown_field",
        "§3 unknown field",
        send=(session_update(nope=1),),
        expect=("error",),
        check=rejects("invalid_value", "session.nope"),
    ),
    case(
        "session_beta_top_level_field",
        "§3 beta top-level fields",
        send=(session_update(voice="marin"),),
        expect=("error",),
        check=rejects("invalid_value", "session.voice"),
    ),
    case(
        "session_merge_is_field_wise",
        "§3 merge semantics",
        steps=(
            Step(
                (session_update(audio={"output": {"format": {"type": "audio/pcmu"}}}),),
                ("session.updated",),
            ),
            Step((session_update(audio={"output": {"voice": "marin"}}),), ("session.updated",)),
        ),
        check=echoes(
            audio__output__format={"type": "audio/pcmu", "rate": 8000},
            audio__output__voice="marin",
            audio__input__format={"type": "audio/pcm", "rate": 24000},
        ),
    ),
    case(
        "session_input_format_change_drops_the_buffered_unit",
        "§3 changing audio.input.format",
        steps=(
            Step((append(0.5),), ()),
            Step(
                (session_update(audio={"input": {"format": {"type": "audio/pcmu"}}}),),
                ("session.updated",),
            ),
            Step((COMMIT,), ("error",)),
        ),
        engine_calls=(),
        check=rejects("invalid_event", None),
    ),
)


RESPONSE_CREATE_CASES: tuple[Case, ...] = (
    case(
        "response_create_bare_is_the_expected_form",
        "§4 bare response.create",
        send=(append(1.0), RESPONSE_CREATE),
        expect=(),
        engine_calls=("feed_pcm16", "flush_pending"),
    ),
    case(
        "response_create_conversation_auto_is_ignored",
        "§4 conversation",
        send=(append(1.0), response_create(conversation="auto")),
        expect=(),
        engine_calls=("feed_pcm16", "flush_pending"),
    ),
    case(
        "response_create_conversation_none_is_rejected",
        "§8.2 response_create_conversation",
        send=(response_create(conversation="none"),),
        expect=("error",),
        engine_calls=(),
        check=rejects("invalid_value", "response.conversation"),
    ),
    case(
        "response_create_metadata_is_echoed_on_the_response",
        "§4 metadata",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        send=(append(1.0), response_create(metadata={"ticket": "T-1"})),
        expect=(*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
        check=lambda o: (
            _eq(o.one("response.created")["response"]["metadata"], {"ticket": "T-1"})
            and _eq(o.one("response.done")["response"]["metadata"], {"ticket": "T-1"})
        ),
    ),
    case(
        "response_create_metadata_too_many_entries",
        "§8.2 response_create_metadata_size",
        send=(response_create(metadata={f"k{i}": "v" for i in range(17)}),),
        expect=("error",),
        check=rejects("invalid_value", "response.metadata"),
    ),
    case(
        "response_create_instructions_is_rejected",
        "§4 instructions",
        send=(response_create(instructions="be terse"),),
        expect=("error",),
        check=rejects("invalid_value", "response.instructions"),
    ),
    case(
        "response_create_output_modalities_is_rejected",
        "§4 output_modalities",
        send=(response_create(output_modalities=["audio"]),),
        expect=("error",),
        check=rejects("invalid_value", "response.output_modalities"),
    ),
    case(
        "response_create_max_output_tokens_is_rejected",
        "§4 max_output_tokens",
        send=(response_create(max_output_tokens=100),),
        expect=("error",),
        check=rejects("invalid_value", "response.max_output_tokens"),
    ),
    case(
        "response_create_audio_is_rejected",
        "§4 audio",
        send=(response_create(audio={"output": {"voice": "marin"}}),),
        expect=("error",),
        check=rejects("invalid_value", "response.audio"),
    ),
    case(
        "response_create_input_is_rejected",
        "§4 input",
        send=(response_create(input=[]),),
        expect=("error",),
        check=rejects("invalid_value", "response.input"),
    ),
    case(
        "response_create_tools_is_rejected",
        "§4 tools",
        send=(response_create(tools=[]),),
        expect=("error",),
        check=rejects("invalid_value", "response.tools"),
    ),
    case(
        "response_create_tool_choice_is_rejected",
        "§4 tool_choice",
        send=(response_create(tool_choice="auto"),),
        expect=("error",),
        check=rejects("invalid_value", "response.tool_choice"),
    ),
    case(
        "response_create_unknown_field_is_rejected",
        "§8.2 response_create_unknown_field",
        send=(response_create(nope=1),),
        expect=("error",),
        check=rejects("invalid_value", "response.nope"),
    ),
)


ITEM_CASES: tuple[Case, ...] = (
    case(
        "item_user_input_text_wire_shape",
        "§5 item shape",
        send=(text_item("hello"),),
        expect=("conversation.item.created",),
        check=lambda o: _eq(
            _without_id(o.one("conversation.item.created")["item"]),
            {
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ),
    ),
    case(
        "item_function_call_output_wire_shape",
        "§5 function_call_output",
        send=(output_item("call_1", '{"ok": true}'),),
        expect=("conversation.item.created",),
        check=lambda o: _eq(
            _without_id(o.one("conversation.item.created")["item"]),
            {
                "object": "realtime.item",
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"ok": true}',
                "status": "completed",
            },
        ),
    ),
    case(
        "item_function_call_output_answered_twice",
        "§8.3 item_call_id_duplicate",
        send=(output_item("call_1"), output_item("call_1")),
        expect=("conversation.item.created", "error"),
        check=rejects("invalid_value", "item.call_id"),
    ),
    case(
        "item_function_call_output_without_call_id",
        "§8.2 item_call_id_missing",
        send=(item_create({"type": "function_call_output", "output": "{}"}),),
        expect=("error",),
        check=rejects("invalid_value", "item.call_id"),
    ),
    case(
        "item_function_call_output_without_output",
        "§8.2 item_output_missing",
        send=(item_create({"type": "function_call_output", "call_id": "call_1"}),),
        expect=("error",),
        check=rejects("invalid_value", "item.output"),
    ),
    case(
        "item_role_system_is_rejected",
        "§5 role system",
        send=(
            item_create(
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "x"}],
                }
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "item.role"),
    ),
    case(
        "item_role_assistant_is_rejected",
        "§5 role assistant",
        send=(
            item_create(
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "input_text", "text": "x"}],
                }
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "item.role"),
    ),
    case(
        "item_input_audio_content_is_rejected",
        "§5 input_audio content",
        send=(
            item_create(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_audio", "audio": "AAAA"}],
                }
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "item.content[0].type"),
    ),
    case(
        "item_multiple_content_parts_are_rejected",
        "§5 multiple content parts",
        send=(
            item_create(
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "a"},
                        {"type": "input_text", "text": "b"},
                    ],
                }
            ),
        ),
        expect=("error",),
        check=rejects("invalid_value", "item.content"),
    ),
    case(
        "item_function_call_type_is_rejected",
        "§5 function_call",
        send=(
            item_create({"type": "function_call", "name": "n", "call_id": "c", "arguments": "{}"}),
        ),
        expect=("error",),
        check=rejects("invalid_value", "item.type"),
    ),
    case(
        "item_reference_type_is_rejected",
        "§5 item_reference",
        send=(item_create({"type": "item_reference", "id": "item_a"}),),
        expect=("error",),
        check=rejects("invalid_value", "item.type"),
    ),
    case(
        "item_client_supplied_id_is_used",
        "§5 client-supplied id",
        send=(text_item("hi", id="item_client"),),
        expect=("conversation.item.created",),
        check=lambda o: _eq(o.one("conversation.item.created")["item"]["id"], "item_client"),
    ),
    case(
        "item_client_supplied_id_must_be_unique",
        "§8.3 item_id_duplicate",
        send=(text_item("a", id="item_client"), text_item("b", id="item_client")),
        expect=("conversation.item.created", "error"),
        check=rejects("invalid_value", "item.id"),
    ),
    case(
        "item_status_and_object_are_ignored",
        "§5 status / object",
        send=(text_item("hi", status="incomplete", object="realtime.item"),),
        expect=("conversation.item.created",),
        check=lambda o: _eq(o.one("conversation.item.created")["item"]["status"], "completed"),
    ),
    case(
        "item_previous_item_id_is_ordering_only",
        "§5 previous_item_id",
        send=(
            text_item("a", id="item_a"),
            item_create(
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "b"}],
                },
                previous_item_id="item_a",
            ),
        ),
        expect=("conversation.item.created", "conversation.item.created"),
        check=lambda o: _eq(o.of("conversation.item.created")[1]["previous_item_id"], "item_a"),
    ),
)


TURN_DETECTION_CASES: tuple[Case, ...] = (
    case(
        "turn_detection_retroactive_speech_started",
        "§7 speech_started",
        engine=scripted(flush_pending=[[speak(3, interrupted=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(
            SPEECH_STARTED,
            *OPEN,
            *CLOSE,
            "conversation.item.truncated",
            "response.done",
        ),
        check=lambda o: (
            _eq(o.one(SPEECH_STARTED)["audio_start_ms"], 2000) and _cancelled(o, "turn_detected")
        ),
    ),
    case(
        "turn_detection_null_synthesizes_nothing",
        "§7 turn_detection null",
        engine=scripted(flush_pending=[[speak(3, interrupted=True)]]),
        steps=(
            Step(
                (session_update(audio={"input": {"turn_detection": None}}),),
                ("session.updated",),
            ),
            Step(
                (append(1.0), RESPONSE_CREATE),
                (*OPEN, *CLOSE, "conversation.item.truncated", "response.done"),
            ),
        ),
        check=lambda o: _eq(o.of(SPEECH_STARTED), []),
    ),
    case(
        "turn_detection_create_response_does_not_change_behaviour",
        "§7 create_response / interrupt_response",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        steps=(
            Step(
                (
                    session_update(
                        audio={
                            "input": {
                                "turn_detection": {
                                    "type": "server_vad",
                                    "create_response": False,
                                    "interrupt_response": False,
                                }
                            }
                        }
                    ),
                ),
                ("session.updated",),
            ),
            Step(
                (append(1.0), RESPONSE_CREATE),
                (*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
            ),
        ),
        check=lambda o: _eq(o.one("response.done")["response"]["status"], "completed"),
    ),
    case(
        "turn_detection_audio_chunk_never_opens_a_span",
        "§7 is_audio_chunk",
        engine=scripted(flush_pending=[[step_event(4, is_audio_chunk=True, interrupted=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(),
    ),
)


SERVER_EVENT_CASES: tuple[Case, ...] = (
    case(
        "server_full_turn_sequence",
        "§9 response lifecycle",
        engine=scripted(
            flush_pending=[
                [
                    speak(1, text="Hello", audio_waveform=wave()),
                    speak(2, text=" world", end_of_turn=True),
                ]
            ]
        ),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(
            *OPEN,
            TRANSCRIPT_DELTA,
            AUDIO_DELTA,
            TRANSCRIPT_DELTA,
            *CLOSE,
            "response.done",
        ),
        check=lambda o: _full_turn(o),
    ),
    case(
        "server_response_created_fields",
        "§9 response object",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
        check=lambda o: _created_fields(o),
    ),
    case(
        "server_output_audio_delta_is_g711_when_negotiated",
        "§9 response.output_audio.delta",
        engine=scripted(flush_pending=[[speak(1, audio_waveform=wave(0.1), end_of_turn=True)]]),
        steps=(
            Step(
                (session_update(audio={"output": {"format": {"type": "audio/pcmu"}}}),),
                ("session.updated",),
            ),
            Step((append(1.0), RESPONSE_CREATE), (*OPEN, AUDIO_DELTA, *CLOSE, "response.done")),
        ),
        check=lambda o: _g711_delta(o),
    ),
    case(
        "server_content_index_and_output_index",
        "§9 content_index / output_index",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
        check=lambda o: _indices(o),
    ),
    case(
        "server_a_unit_without_audio_emits_no_delta",
        "§9 response.output_audio.delta",
        engine=scripted(flush_pending=[[speak(1, text="Hi", end_of_turn=True)]]),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(*OPEN, TRANSCRIPT_DELTA, *CLOSE, "response.done"),
        check=lambda o: _eq(o.of(AUDIO_DELTA), []),
    ),
    case(
        "server_every_event_carries_a_unique_event_id",
        "§9 field details",
        engine=scripted(
            flush_pending=[[speak(1, text="Hi", audio_waveform=wave()), speak(2, end_of_turn=True)]]
        ),
        send=(append(1.0), RESPONSE_CREATE),
        expect=(*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE, "response.done"),
        check=lambda o: _unique_event_ids(o),
    ),
)


ALL_CASES: tuple[Case, ...] = (
    *TRANSPORT_CASES,
    *CLIENT_EVENT_CASES,
    *SESSION_FIELD_CASES,
    *RESPONSE_CREATE_CASES,
    *ITEM_CASES,
    *TURN_DETECTION_CASES,
    *SERVER_EVENT_CASES,
)


# ---------------------------------------------------------------------------
# checks used by the cases
# ---------------------------------------------------------------------------


def _fail(outcome: Outcome) -> bool:  # pragma: no cover - only on a failing case
    raise AssertionError(outcome)


def _eq(actual: Any, expected: Any) -> bool:
    assert actual == expected, f"{actual!r} != {expected!r}"
    return True


def _without_id(item: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in item.items() if key != "id"}


def _commit_shape(outcome: Outcome) -> bool:
    committed = outcome.one("input_audio_buffer.committed")
    created = outcome.one("conversation.item.created")
    assert committed["item_id"] == created["item"]["id"]
    assert committed["previous_item_id"] is None
    assert created["item"]["content"] == [{"type": "input_audio", "transcript": None}]
    return True


def _cancelled(outcome: Outcome, reason: str) -> bool:
    response = outcome.one("response.done")["response"]
    assert response["status"] == "cancelled", response
    assert response["status_details"] == {"type": "cancelled", "reason": reason}
    assert outcome.one("response.output_item.done")["item"]["status"] == "incomplete"
    assert outcome.one("conversation.item.truncated")["item_id"] == response["output"][0]["id"]
    return True


def _full_turn(outcome: Outcome) -> bool:
    deltas = [event["delta"] for event in outcome.of(TRANSCRIPT_DELTA)]
    assert deltas == ["Hello", " world"]
    done = outcome.one("response.output_audio_transcript.done")
    assert done["transcript"] == "Hello world"
    assert outcome.one("response.content_part.done")["part"] == {
        "type": "audio",
        "transcript": "Hello world",
    }
    item = outcome.one("response.output_item.done")["item"]
    assert item["status"] == "completed"
    assert item["content"] == [{"type": "output_audio", "transcript": "Hello world"}]
    response = outcome.one("response.done")["response"]
    assert response["status"] == "completed"
    assert response["status_details"] is None
    assert response["output"] == [item]
    usage = response["usage"]
    assert usage["input_tokens"] == 0
    assert usage["output_tokens"] == usage["total_tokens"] > 0
    assert usage["output_token_details"]["audio_tokens"] == 0
    return True


def _created_fields(outcome: Outcome) -> bool:
    response = outcome.one("response.created")["response"]
    assert response["object"] == "realtime.response"
    assert response["id"].startswith("resp_")
    assert response["status"] == "in_progress"
    assert response["status_details"] is None
    assert response["output"] == []
    assert response["usage"] is None
    assert response["output_modalities"] == ["audio"]
    assert response["audio"]["output"]["voice"] == "gander"
    assert response["conversation_id"].startswith("conv_")
    assert outcome.one("response.content_part.added")["part"] == {
        "type": "audio",
        "transcript": "",
    }
    return True


def _g711_delta(outcome: Outcome) -> bool:
    delta = base64.b64decode(outcome.one(AUDIO_DELTA)["delta"])
    # 0.1 s of 24 kHz model audio is 800 g711 bytes at 8 kHz, one byte per sample.
    assert 700 <= len(delta) <= 900, len(delta)
    decoded = ulaw_decode(delta)
    assert decoded.dtype == np.int16
    assert int(np.abs(decoded).max()) > 0
    return True


def _indices(outcome: Outcome) -> bool:
    for kind in (TRANSCRIPT_DELTA, "response.content_part.added", "response.content_part.done"):
        for event in outcome.of(kind):
            assert event["content_index"] == 0, event
            assert event["output_index"] == 0, event
    for kind in ("response.output_item.added", "response.output_item.done"):
        for event in outcome.of(kind):
            assert event["output_index"] == 0, event
            assert "content_index" not in event, event
    return True


def _first(events: Sequence[Mapping[str, Any]], kind: str) -> dict[str, Any]:
    """The first event of this type; the session-mode cases read a list sink directly."""
    return next(dict(event) for event in events if event["type"] == kind)


def _unique_event_ids(outcome: Outcome) -> bool:
    ids = [event["event_id"] for event in outcome.handshake + outcome.events]
    assert all(value.startswith("event_") for value in ids), ids
    assert len(set(ids)) == len(ids), ids
    return True


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def _run_case(case_: Case, connect: Callable[..., Any]) -> Outcome:
    engine = case_.engine() if case_.engine is not None else FakeEngine()
    collected: list[dict[str, Any]] = []
    with connect(engine, query=case_.query, headers=case_.headers) as client:
        for index, step in enumerate(case_.steps):
            client.send_all(step.send)
            got = client.receive_many(len(step.expect))
            assert [event["type"] for event in got] == list(step.expect), (
                f"step {index}: {[event['type'] for event in got]}"
            )
            collected.extend(got)
        extra = client.drain()
        assert extra == [], f"unexpected trailing events: {[e['type'] for e in extra]}"
        handshake = list(client.handshake)
    outcome = Outcome(handshake, collected, engine)
    if case_.engine_calls is not None:
        assert engine.call_names == list(case_.engine_calls)
    if case_.check is not None:
        case_.check(outcome)
    return outcome


def _params() -> list[Any]:
    out = []
    for item in ALL_CASES:
        marks = [pytest.mark.xfail(strict=True, reason=item.xfail)] if item.xfail else []
        out.append(pytest.param(item, id=item.id, marks=marks))
    return out


@pytest.mark.parametrize("cascade_case", _params())
def test_cascade_case(cascade_case: Case, realtime_connect: Callable[..., Any]) -> None:
    """Run one ported case end to end over `/v1/realtime`."""
    _run_case(cascade_case, realtime_connect)


def test_case_ids_are_unique() -> None:
    ids = [item.id for item in ALL_CASES]
    assert len(set(ids)) == len(ids)


def test_every_case_names_the_profile_rule_it_came_from() -> None:
    assert all(item.ref.startswith("§") for item in ALL_CASES)


# ---------------------------------------------------------------------------
# §1 transport: auth and the single-session guard (not frame-driven)
# ---------------------------------------------------------------------------


def test_missing_key_is_refused_before_the_upgrade(realtime_client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        realtime_client.websocket_connect("/v1/realtime"),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_wrong_key_is_refused_before_the_upgrade(realtime_client: TestClient) -> None:
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        realtime_client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer nope"}),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_a_second_connection_is_engine_busy_and_closed(realtime_client: TestClient) -> None:
    with realtime_client.websocket_connect("/v1/realtime", headers=AUTH) as first:
        first.receive_json()
        first.receive_json()
        with realtime_client.websocket_connect("/v1/realtime", headers=AUTH) as second:
            error = second.receive_json()
            assert error["type"] == "error"
            assert error["error"]["code"] == "engine_busy"
            assert error["error"]["type"] == "server_error"
            # A server-originated error carries no client `event_id` (§8).
            assert error["error"]["event_id"] is None
            assert error["error"]["param"] is None
            closed = second.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 1011
        # The first session is unaffected by the rejected one.
        first.send_json({"type": "input_audio_buffer.clear"})
        assert first.receive_json()["type"] == "input_audio_buffer.cleared"
        assert realtime_client.get("/health").json()["status"] == "busy"


def test_health_reports_readiness(realtime_client: TestClient) -> None:
    body = realtime_client.get("/health").json()
    assert body == {
        "status": "idle",
        "engine": True,
        "asr": None,
        "brain": None,
        "session_active": False,
    }


def test_a_fatal_engine_error_closes_with_1011(
    realtime_config: TalkoverConfig,
) -> None:
    engine = FakeEngine(fail_on={"submit_text_turn"})
    app = create_app(realtime_config, engine, session_factory=WebSocketSession)
    with TestClient(app).websocket_connect("/v1/realtime", headers=AUTH) as ws:
        ws.receive_json()
        ws.receive_json()
        ws.send_json(text_item("hello"))
        assert ws.receive_json()["type"] == "conversation.item.created"
        error = ws.receive_json()
        assert error["error"]["code"] == "engine_error"
        assert error["error"]["type"] == "server_error"
        closed = ws.receive()
        assert closed["type"] == "websocket.close"
        assert closed["code"] == 1011


# ---------------------------------------------------------------------------
# §9 cases that have no client-initiated path (the Brain and ASR side channels)
# ---------------------------------------------------------------------------


async def test_function_call_joins_the_streaming_response() -> None:
    """§9: a `function_call` is the response's next output item."""
    session, _ = make_session()
    await session.handle_engine_event(speak(1, text="One moment"))
    await session.emit_function_call("call_1", "transfer_to_human", '{"reason": "refund"}')
    await session.handle_engine_event(speak(2, end_of_turn=True))

    types = [event["type"] for event in session.sink.events]
    assert types == [
        *OPEN,
        TRANSCRIPT_DELTA,
        "response.output_item.added",
        "conversation.item.created",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        *CLOSE,
        "response.done",
    ]
    delta = _first(session.sink.events, "response.function_call_arguments.delta")
    assert delta["delta"] == '{"reason": "refund"}'
    assert delta["output_index"] == 1
    done = _first(session.sink.events, "response.done")["response"]
    assert [item["type"] for item in done["output"]] == ["message", "function_call"]


async def test_an_intercepted_transfer_opens_a_response_of_its_own() -> None:
    """§9: no message item, `output_index` 0, closed right after `.done`."""
    session, _ = make_session()
    await session.emit_function_call("call_1", "transfer_to_human", "{}")

    types = [event["type"] for event in session.sink.events]
    assert types == [
        "response.created",
        "response.output_item.added",
        "conversation.item.created",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.done",
    ]
    added = _first(session.sink.events, "response.output_item.added")
    assert added["output_index"] == 0
    response = _first(session.sink.events, "response.done")["response"]
    assert response["status"] == "completed"
    assert [item["type"] for item in response["output"]] == ["function_call"]


async def test_an_asr_segment_completes_the_committed_item() -> None:
    """§9: the transcription event targets the last committed input item."""
    session, _ = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(1.0)})
    await session.handle_raw({"type": "input_audio_buffer.commit"})
    item_id = session.input_item_id
    session.sink.events.clear()

    await session.on_asr_event(Transcript(text="查一下订单", start_ms=200, end_ms=1400))

    events = session.sink.events
    assert [event["type"] for event in events] == [
        "conversation.item.input_audio_transcription.completed"
    ]
    assert events[0]["item_id"] == item_id
    assert events[0]["transcript"] == "查一下订单"
    assert events[0]["usage"] == {"type": "duration", "seconds": 1.2}
    assert session.items[item_id]["content"][0]["transcript"] == "查一下订单"


async def test_a_segment_before_any_commit_is_dropped() -> None:
    """§9: a transcript with no item to name is dropped."""
    session, _ = make_session()
    await session.on_asr_event(Transcript(text="hello", start_ms=0, end_ms=100))
    assert session.sink.events == []


# ---------------------------------------------------------------------------
# the profile is the source: machine-read checks
# ---------------------------------------------------------------------------


def _profile_text() -> str:
    return PROFILE.read_text(encoding="utf-8")


def _default_session_block() -> dict[str, Any]:
    """The effective-session JSON block of section 3."""
    text = _profile_text()
    start = text.index("Effective values before the first `session.update`")
    block = text[start:].split("```json", 1)[1].split("```", 1)[0]
    return json.loads(block)


def test_session_created_matches_the_documented_defaults(
    realtime_connect: Callable[..., Any],
) -> None:
    with realtime_connect() as client:
        session = client.session
    documented = _default_session_block()
    effective = {key: value for key, value in session.items() if key in documented}
    assert effective == documented
    assert session["id"].startswith("sess_")
    assert session["object"] == "realtime.session"
    assert set(session) - set(documented) == {"id", "object", "model"}


def _never_emitted() -> list[str]:
    """The `Never emitted:` list of section 9, as event-type prefixes."""
    text = _profile_text()
    start = text.index("Never emitted:")
    block = text[start : text.index("Field details:")]
    return sorted({name.strip("`*.") for name in re.findall(r"`([^`]+)`", block)})


def test_a_full_turn_emits_none_of_the_never_emitted_events(
    realtime_connect: Callable[..., Any],
) -> None:
    forbidden = _never_emitted()
    assert forbidden, "section 9 must list the never-emitted events"
    engine = FakeEngine(
        on_call={
            "flush_pending": [
                [
                    speak(1, text="Hello", audio_waveform=wave()),
                    speak(2, text=" world", end_of_turn=True),
                ]
            ]
        }
    )
    with realtime_connect(engine) as client:
        client.send_all((append(1.0), COMMIT))
        seen = [event["type"] for event in client.receive_many(2)]
        client.send(RESPONSE_CREATE)
        seen += [event["type"] for event in client.receive_many(12)]
        seen += [event["type"] for event in client.drain()]
    for name in forbidden:
        assert not any(kind == name or kind.startswith(name) for kind in seen), name


# ---------------------------------------------------------------------------
# Cascade-only cases: skipped, and documented in section 11.1
# ---------------------------------------------------------------------------

#: `case id -> reason`. Every entry is also a row of section 11.1 of the profile.
SKIPPED_CASES: dict[str, str] = {
    "vad_threshold_boundary": (
        "Cascade tunes a real VAD; Talkover's turn detection is approximate (§7, §11) and "
        "threshold only reaches the ASR side channel."
    ),
    "vad_prefix_padding_trims_the_committed_audio": (
        "Cascade's VAD rewinds the committed buffer by prefix_padding_ms; Talkover validates "
        "and echoes the field without acting on it (§7)."
    ),
    "semantic_vad_transcript_gated_response_created": (
        "Cascade orders response.created after a final transcript; Talkover starts a response "
        "when the model stops listening (§11, deliberately not adopted)."
    ),
    "idle_timeout_triggered": (
        "input_audio_buffer.timeout_triggered is in the never-emitted list of §9; "
        "idle_timeout_ms is rejected outright (§7)."
    ),
    "conversation_item_added_then_done": (
        "Cascade's item lifecycle is .added then .done; Talkover emits conversation.item.created "
        "only (§11)."
    ),
    "input_audio_transcription_delta": (
        "Cascade streams transcription deltas; Talkover emits .completed only, one per ASR "
        "segment (§11)."
    ),
    "response_output_text_flow": (
        "Cascade serves output_modalities ['text']; Talkover has no text-only response path "
        "(§3, §11)."
    ),
    "rate_limits_updated": "rate_limits.updated is in the never-emitted list of §9.",
    "conversation_item_retrieved": (
        "conversation.item.retrieve is rejected because Talkover retains no input audio (§2), "
        "so the .retrieved event has no case."
    ),
    "output_audio_buffer_events": (
        "output_audio_buffer.* is WebRTC and SIP only; the WebSocket profile rejects the client "
        "event and never emits the server ones (§2, §9)."
    ),
    "per_response_voice_override": (
        "Cascade supports a per-response voice and the 'no voice change after the first audio' "
        "rule; the Talker voice is fixed by the checkpoint (§11)."
    ),
    "parallel_tool_calls": "Not adopted: tools are session-level and the Brain owns selection (§11).",
    "mcp_tool_events": "response.mcp_call* and mcp_list_tools.* are never emitted (§9, §11).",
    "truncate_rewinds_the_model_context": (
        "Cascade rewinds assistant audio and text on conversation.item.truncate; Talkover "
        "acknowledges only (§2, §11)."
    ),
    "multi_session_concurrency": (
        "Cascade serves many sessions per process; Talkover is one session per process and a "
        "second connection gets engine_busy (§1, §10, §11)."
    ),
    "per_tenant_api_keys": (
        "Cascade authenticates per tenant; Talkover has a single server.api_key (§1, §11)."
    ),
    "cascade_server_error_codes": (
        "provider_error, transcript_timeout, input_audio_buffer_overflow and "
        "input_queue_overflow do not exist in Talkover; the codes are engine_busy and "
        "engine_error (§8.1, §11)."
    ),
    "transcription_session_updated": (
        "Talkover serves no transcription sessions: the client event is rejected and "
        "transcription_session.updated is never emitted (§2, §9)."
    ),
}


@pytest.mark.parametrize("case_id", sorted(SKIPPED_CASES))
def test_cascade_only_case(case_id: str) -> None:
    """A Cascade case Talkover does not implement; see section 11.1 of the profile."""
    pytest.skip(SKIPPED_CASES[case_id])


def _profile_skipped_cases() -> dict[str, str]:
    """case id -> reason, as documented in section 11.1."""
    text = _profile_text()
    start = text.index("### 11.1 Skipped Cascade cases")
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
    return {row[0].strip("`"): row[1] for row in rows[1:]}


def test_skipped_cases_are_documented() -> None:
    assert set(_profile_skipped_cases()) == set(SKIPPED_CASES)


def test_documented_skip_reasons_are_not_empty() -> None:
    assert all(reason for reason in _profile_skipped_cases().values())
