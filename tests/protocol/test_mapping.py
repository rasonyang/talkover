"""`EngineStepEvent` to Realtime server event mapping tests (T2.5).

The cases drive the right column of the event mapping table in DESIGN.md 5.3 and the
response half of `docs/protocol-profile.md` section 9: a scripted unit sequence from
`tests/protocol/fake_engine.FakeEngine` must yield exactly the client events the profile
documents, in that order, with the documented fields.

GPU-free: no model and no ASR backend, only the event types of `talkover.engine.asr`.
"""

from __future__ import annotations

import asyncio
import base64
from typing import Any

import numpy as np
import pytest

from talkover.engine.asr import Transcript
from talkover.engine.protocol import EngineStepEvent
from talkover.realtime.audio import alaw_decode, float_to_pcm16, ulaw_decode
from talkover.realtime.mapping import ResponseMapper

from .fake_engine import FakeEngine, step_event
from .test_session import drop, emitted, make_session, pcm24, session_update, types_of

OPEN = [
    "response.created",
    "response.output_item.added",
    "conversation.item.created",
    "response.content_part.added",
]
CLOSE = [
    "response.output_audio.done",
    "response.output_audio_transcript.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.done",
]
TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
AUDIO_DELTA = "response.output_audio.delta"
TRANSCRIPTION_DONE = "conversation.item.input_audio_transcription.completed"


def wave(seconds: float = 0.1, value: float = 0.5) -> np.ndarray:
    """One chunk of model audio: float32 at the engine's 24 kHz output rate."""
    return np.full(int(24000 * seconds), value, dtype=np.float32)


def only_event(session: Any, kind: str) -> dict[str, Any]:
    matches = [event for event in emitted(session) if event["type"] == kind]
    assert len(matches) == 1, f"expected exactly one {kind}, got {types_of(session)}"
    return matches[0]


def events_of(session: Any, kind: str) -> list[dict[str, Any]]:
    return [event for event in emitted(session) if event["type"] == kind]


async def run(session: Any, *steps: EngineStepEvent) -> None:
    """Hand a scripted unit sequence to the session exactly as the pump does."""
    for step in steps:
        await session.handle_engine_event(step)


async def pump(session: Any, engine: FakeEngine, *steps: EngineStepEvent) -> None:
    """Drive the same sequence through `events()` and `pump_engine_events`."""
    await engine.push(*steps)
    await engine.stop()
    await session.pump_engine_events()


def speak(index: int, **overrides: Any) -> EngineStepEvent:
    """A unit the model speaks in (`is_listen: false`)."""
    return step_event(index, is_listen=False, **overrides)


# ---------------------------------------------------------------------------
# a normal turn
# ---------------------------------------------------------------------------


async def test_a_listening_unit_produces_nothing() -> None:
    session, _ = make_session()
    await run(session, step_event(1), step_event(2))
    assert types_of(session) == []
    assert session.active_response_id is None


async def test_a_full_turn_yields_the_documented_event_list() -> None:
    session, _ = make_session()
    await run(
        session,
        step_event(1),
        speak(2, text="Hello", audio_waveform=wave()),
        speak(3, text=" there", audio_waveform=wave()),
        speak(4, end_of_turn=True),
    )
    assert types_of(session) == [
        *OPEN,
        TRANSCRIPT_DELTA,
        AUDIO_DELTA,
        TRANSCRIPT_DELTA,
        AUDIO_DELTA,
        *CLOSE,
    ]
    assert session.active_response_id is None


async def test_a_scripted_sequence_through_the_pump_matches() -> None:
    """The acceptance case: fake engine in, exactly the profile's event list out."""
    session, engine = make_session()
    await pump(
        session,
        engine,
        step_event(1),
        speak(2, text="Hi", audio_waveform=wave()),
        speak(3, end_of_turn=True),
    )
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE]


async def test_the_response_is_opened_once_for_the_whole_turn() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="a"), speak(2, text="b"), speak(3, text="c"))
    assert types_of(session).count("response.created") == 1
    assert len(events_of(session, TRANSCRIPT_DELTA)) == 3


async def test_the_opening_events_describe_the_assistant_item() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="Hello"))

    created = only_event(session, "response.created")["response"]
    response_id = created["id"]
    assert response_id.startswith("resp_")
    assert created["object"] == "realtime.response"
    assert created["status"] == "in_progress"
    assert created["status_details"] is None
    assert created["output"] == []
    assert created["usage"] is None
    assert created["conversation_id"] == session.conversation_id
    assert created["output_modalities"] == ["audio"]
    assert created["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}
    assert created["metadata"] is None

    added = only_event(session, "response.output_item.added")
    item = added["item"]
    assert added["response_id"] == response_id
    assert added["output_index"] == 0
    assert item["type"] == "message"
    assert item["role"] == "assistant"
    assert item["status"] == "in_progress"
    assert item["id"].startswith("item_")

    # The same item is announced on the conversation and is retained for truncate/delete.
    assert only_event(session, "conversation.item.created")["item"]["id"] == item["id"]
    assert session.items[item["id"]]["role"] == "assistant"

    part = only_event(session, "response.content_part.added")
    assert part["item_id"] == item["id"]
    assert part["content_index"] == 0
    assert part["part"] == {"type": "audio", "transcript": ""}

    delta = only_event(session, TRANSCRIPT_DELTA)
    assert delta == {
        "event_id": delta["event_id"],
        "type": TRANSCRIPT_DELTA,
        "response_id": response_id,
        "item_id": item["id"],
        "output_index": 0,
        "content_index": 0,
        "delta": "Hello",
    }


async def test_end_of_turn_closes_with_the_accumulated_transcript() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="Hello"), speak(2, text=" world", end_of_turn=True))

    item_id = only_event(session, "response.output_item.added")["item"]["id"]
    assert only_event(session, "response.output_audio.done")["item_id"] == item_id
    assert only_event(session, "response.output_audio_transcript.done")["transcript"] == (
        "Hello world"
    )
    assert only_event(session, "response.content_part.done")["part"] == {
        "type": "audio",
        "transcript": "Hello world",
    }
    done_item = only_event(session, "response.output_item.done")["item"]
    assert done_item["status"] == "completed"
    assert done_item["content"] == [{"type": "output_audio", "transcript": "Hello world"}]

    response = only_event(session, "response.done")["response"]
    assert response["status"] == "completed"
    assert response["status_details"] is None
    assert response["output"] == [done_item]
    assert response["usage"]["output_token_details"]["audio_tokens"] == 0
    assert response["usage"]["output_tokens"] > 0
    # The stored item is the final one, so a later `conversation.item.truncate` sees it.
    assert session.items[item_id]["status"] == "completed"


async def test_a_second_turn_opens_a_new_response() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="one", end_of_turn=True))
    first = only_event(session, "response.created")["response"]["id"]
    drop(session)
    await run(session, step_event(2), speak(3, text="two", end_of_turn=True))
    second = only_event(session, "response.created")["response"]["id"]
    assert first != second
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, *CLOSE]


async def test_response_create_metadata_is_echoed_on_the_response() -> None:
    session, _ = make_session()
    await session.handle_raw({"type": "response.create", "response": {"metadata": {"call": "42"}}})
    drop(session)
    await run(session, speak(1, text="hi", end_of_turn=True))
    assert only_event(session, "response.created")["response"]["metadata"] == {"call": "42"}
    assert only_event(session, "response.done")["response"]["metadata"] == {"call": "42"}
    assert session.pending_response_metadata is None


# ---------------------------------------------------------------------------
# audio deltas
# ---------------------------------------------------------------------------


async def test_audio_is_base64_pcm16_at_24_khz_by_default() -> None:
    session, _ = make_session()
    samples = wave(0.1)
    await run(session, speak(1, audio_waveform=samples))
    delta = only_event(session, AUDIO_DELTA)["delta"]
    decoded = np.frombuffer(base64.b64decode(delta), dtype="<i2")
    assert decoded.size == samples.size
    assert np.array_equal(decoded, float_to_pcm16(samples))


async def test_a_silent_unit_emits_no_audio_delta() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="only text"))
    assert events_of(session, AUDIO_DELTA) == []


async def test_g711_output_is_encoded_at_8_khz() -> None:
    session, _ = make_session()
    await session.handle_raw(
        session_update(audio={"output": {"format": {"type": "audio/pcmu", "rate": 8000}}})
    )
    drop(session)
    await run(session, speak(1, audio_waveform=wave(0.5)))

    payload = base64.b64decode(only_event(session, AUDIO_DELTA)["delta"])
    # 0.5 s at 24 kHz resampled to 8 kHz is one byte per sample, minus the filter warm-up.
    assert 3900 <= len(payload) <= 4000
    decoded = ulaw_decode(payload)
    assert np.allclose(decoded[100:] / 32768.0, 0.5, atol=0.02)


async def test_a_g711_response_reuses_one_resampler_across_chunks() -> None:
    session, _ = make_session()
    await session.handle_raw(
        session_update(audio={"output": {"format": {"type": "audio/pcma", "rate": 8000}}})
    )
    drop(session)
    chunk = wave(0.3)
    await run(session, speak(1, audio_waveform=chunk), speak(2, audio_waveform=chunk))
    payloads = [base64.b64decode(event["delta"]) for event in events_of(session, AUDIO_DELTA)]
    assert [len(payload) for payload in payloads] == [2400, 2400]
    first, second = (alaw_decode(payload) / 32768.0 for payload in payloads)
    # The response keeps one resampler, so the second chunk continues at the steady-state
    # amplitude instead of replaying the filter warm-up the first chunk paid for.
    assert abs(first[0]) < 0.1
    assert np.allclose(second[:50], 0.5, atol=0.02)


async def test_engine_audio_at_another_rate_is_resampled_to_the_client_rate() -> None:
    session, _ = make_session()
    await run(
        session, speak(1, audio_waveform=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    )
    decoded = np.frombuffer(base64.b64decode(only_event(session, AUDIO_DELTA)["delta"]), "<i2")
    assert decoded.size == pytest.approx(24000, abs=2)


async def test_a_detached_talker_audio_chunk_only_extends_the_open_response() -> None:
    session, _ = make_session()
    # An `is_audio_chunk` event carries no text, no `end_of_turn` and no `is_listen`
    # meaning: on its own it may neither open nor close a response.
    await run(session, step_event(1, is_audio_chunk=True, audio_waveform=wave()))
    assert types_of(session) == []
    await run(
        session,
        speak(2, text="hi"),
        step_event(3, is_audio_chunk=True, audio_waveform=wave(), end_of_turn=True),
    )
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA]
    assert session.active_response_id is not None


# ---------------------------------------------------------------------------
# interruption
# ---------------------------------------------------------------------------


async def test_an_interrupted_turn_is_cancelled_and_truncated() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Let me check", audio_waveform=wave(0.5)),
        speak(2, interrupted=True),
    )
    assert types_of(session) == [
        *OPEN,
        TRANSCRIPT_DELTA,
        AUDIO_DELTA,
        "input_audio_buffer.speech_started",  # the retroactive start of T2.6
        "response.output_audio.done",
        "response.output_audio_transcript.done",
        "response.content_part.done",
        "response.output_item.done",
        "conversation.item.truncated",
        "response.done",
    ]
    item_id = only_event(session, "response.output_item.added")["item"]["id"]
    assert only_event(session, "response.output_item.done")["item"]["status"] == "incomplete"

    truncated = only_event(session, "conversation.item.truncated")
    assert truncated["item_id"] == item_id
    assert truncated["content_index"] == 0
    assert truncated["audio_end_ms"] == 500  # exactly the audio already sent

    response = only_event(session, "response.done")["response"]
    assert response["status"] == "cancelled"
    assert response["status_details"] == {"type": "cancelled", "reason": "turn_detected"}
    assert session.active_response_id is None


async def test_a_client_cancel_is_reported_as_client_cancelled() -> None:
    session, engine = make_session()
    await run(session, speak(1, text="Let me check", audio_waveform=wave(0.25)))
    await session.handle_raw({"type": "response.cancel"})
    assert engine.call_names[-1] == "interrupt_output"
    drop(session)
    await run(session, speak(2, interrupted=True))

    response = only_event(session, "response.done")["response"]
    assert response["status"] == "cancelled"
    assert response["status_details"] == {"type": "cancelled", "reason": "client_cancelled"}
    assert only_event(session, "conversation.item.truncated")["audio_end_ms"] == 250
    assert session.client_cancel_requested is False


async def test_a_later_barge_in_is_no_longer_client_cancelled() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="one"))
    await session.handle_raw({"type": "response.cancel"})
    await run(session, speak(2, interrupted=True))
    drop(session)
    await run(session, speak(3, text="two"), speak(4, interrupted=True))
    assert only_event(session, "response.done")["response"]["status_details"] == {
        "type": "cancelled",
        "reason": "turn_detected",
    }


async def test_an_interruption_without_a_response_is_ignored() -> None:
    session, _ = make_session()
    await run(session, step_event(1, end_of_turn=True))
    assert types_of(session) == []


# ---------------------------------------------------------------------------
# the ASR side channel
# ---------------------------------------------------------------------------


async def test_an_asr_segment_completes_the_committed_input_item() -> None:
    session, _ = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(1.0)})
    await session.handle_raw({"type": "input_audio_buffer.commit"})
    item_id = session.input_item_id
    assert only_event(session, "conversation.item.created")["item"]["content"] == [
        {"type": "input_audio", "transcript": None}
    ]
    drop(session)

    await session.on_asr_event(Transcript(text="查一下订单", start_ms=200, end_ms=1400))
    completed = only_event(session, TRANSCRIPTION_DONE)
    assert completed["item_id"] == item_id
    assert completed["content_index"] == 0
    assert completed["transcript"] == "查一下订单"
    assert completed["usage"] == {"type": "duration", "seconds": 1.2}
    # The stored item now carries the transcript the profile promises (§2).
    assert session.items[item_id]["content"][0]["transcript"] == "查一下订单"


async def test_an_asr_segment_without_a_committed_item_emits_nothing() -> None:
    session, _ = make_session()
    assert await session.mapper.on_asr_transcript(Transcript("hi", 0, 100)) is False
    assert types_of(session) == []


async def test_an_asr_segment_can_name_the_item_explicitly() -> None:
    session, _ = make_session()
    assert await session.mapper.on_asr_transcript("hello", item_id="item_x") is True
    assert only_event(session, TRANSCRIPTION_DONE)["item_id"] == "item_x"


# ---------------------------------------------------------------------------
# the Brain function call (the T3.7 seam)
# ---------------------------------------------------------------------------


async def test_a_function_call_during_a_turn_is_one_delta_and_its_done() -> None:
    session, _ = make_session()
    await run(session, speak(1, text="Transferring you now"))
    drop(session)
    arguments = '{"reason": "customer asked for an agent"}'
    item_id = await session.emit_function_call("call_1", "transfer_to_human", arguments)

    assert types_of(session) == [
        "response.output_item.added",
        "conversation.item.created",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
    ]
    added = only_event(session, "response.output_item.added")
    assert added["output_index"] == 1  # the message item holds index 0 (profile §9)
    assert added["item"] == {
        "id": item_id,
        "object": "realtime.item",
        "type": "function_call",
        "name": "transfer_to_human",
        "call_id": "call_1",
        "arguments": arguments,
        "status": "in_progress",
    }
    delta = only_event(session, "response.function_call_arguments.delta")
    assert delta["delta"] == arguments  # complete in one delta
    assert delta["call_id"] == "call_1"
    assert delta["item_id"] == item_id
    done = only_event(session, "response.function_call_arguments.done")
    assert done["arguments"] == arguments
    assert done["name"] == "transfer_to_human"
    assert done["output_index"] == 1

    drop(session)
    await run(session, speak(2, end_of_turn=True))
    output = only_event(session, "response.done")["response"]["output"]
    assert [entry["type"] for entry in output] == ["message", "function_call"]
    assert output[1]["status"] == "completed"


async def test_a_function_call_outside_a_turn_opens_a_response_of_its_own() -> None:
    session, _ = make_session()
    item_id = await session.emit_function_call("call_2", "transfer_to_human", "{}")
    assert types_of(session) == [
        "response.created",
        "response.output_item.added",
        "conversation.item.created",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.done",
    ]
    # No assistant message item, so the function call is the first output item.
    assert only_event(session, "response.output_item.added")["output_index"] == 0
    response = only_event(session, "response.done")["response"]
    assert response["status"] == "completed"
    assert [entry["id"] for entry in response["output"]] == [item_id]
    assert session.active_response_id is None


async def test_the_client_may_answer_the_call_id_talkover_emitted() -> None:
    session, _ = make_session()
    await session.emit_function_call("call_3", "transfer_to_human", "{}")
    drop(session)
    await session.handle_raw(
        {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": "call_3", "output": "ok"},
        }
    )
    assert types_of(session) == ["conversation.item.created"]


# ---------------------------------------------------------------------------
# the mapper on its own
# ---------------------------------------------------------------------------


async def test_the_session_owns_one_mapper_wired_into_the_pump() -> None:
    session, engine = make_session()
    assert isinstance(session.mapper, ResponseMapper)
    assert session.mapper.session is session
    await pump(session, engine, speak(1, text="hi", end_of_turn=True))
    assert session.mapper.active is None


async def test_the_extra_engine_hook_still_sees_every_unit() -> None:
    seen: list[int] = []

    async def hook(step: EngineStepEvent) -> None:
        seen.append(step.unit_index)

    session, engine = make_session(on_engine_event=hook)
    await pump(session, engine, step_event(1), speak(2, text="hi", end_of_turn=True))
    assert seen == [1, 2]
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, *CLOSE]


# ---------------------------------------------------------------------------
# holding the close chain for the threaded Talker (T1.5 / DESIGN.md 5.3, 11)
# ---------------------------------------------------------------------------

#: What upstream `DuplexLiveSession.talker_state()` reports with a `TalkerThread` attached.
DETACHED = {"talker": {"mode": "detached", "active": True, "drained": False}}


def chunk(index: int, **overrides: Any) -> EngineStepEvent:
    """One late Talker waveform, as `EngineSession.audio_event` builds it."""
    fields: dict[str, Any] = {
        "is_listen": False,
        "is_audio_chunk": True,
        "audio_waveform": wave(),
    }
    fields.update(overrides)
    return step_event(index, **fields)


def drained(index: int, generation_id: int = 0, unit_id: int = 1) -> EngineStepEvent:
    """The Talker's drained marker, as `EngineSession.drained_event` builds it."""
    return step_event(
        index,
        is_listen=False,
        is_audio_chunk=True,
        talker_done=True,
        unit_id=unit_id,
        generation_id=generation_id,
    )


async def test_the_in_step_talker_still_closes_on_end_of_turn() -> None:
    """The T1.4 default is unchanged: no chunks, so nothing is held."""
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Hi", audio_waveform=wave()),
        speak(2, end_of_turn=True),
    )
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE]
    assert session.mapper.talker_threaded is False
    assert session.active_response_id is None


async def test_end_of_turn_is_held_until_the_talker_reports_the_turn_drained() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Hi", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, text=" there", end_of_turn=True, metrics=DETACHED),
    )
    # The close chain is held: the turn's own waveform has not been vocoded yet.
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, TRANSCRIPT_DELTA]
    assert session.active_response_id is not None

    # A chunk arriving inside the hold is delivered normally, ...
    await run(session, chunk(2, unit_id=2))
    assert types_of(session)[-1] == AUDIO_DELTA
    assert session.active_response_id is not None

    # ... and the drained marker releases the chain.
    await run(session, drained(2, unit_id=2))
    assert types_of(session) == [
        *OPEN,
        TRANSCRIPT_DELTA,
        AUDIO_DELTA,
        TRANSCRIPT_DELTA,
        AUDIO_DELTA,
        *CLOSE,
    ]
    assert session.active_response_id is None
    done = only_event(session, "response.done")
    assert done["response"]["status"] == "completed"
    assert session.mapper.dropped_audio_chunks == 0


async def test_the_first_turn_is_held_on_the_metrics_alone() -> None:
    """A one-unit first turn has no chunk yet; `metrics["talker"]["mode"]` still holds it."""
    session, _ = make_session()
    await run(session, speak(1, text="Hi", end_of_turn=True, metrics=DETACHED))
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA]
    assert session.mapper.talker_threaded is True

    await run(session, chunk(1, unit_id=1), drained(1, unit_id=1))
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE]


async def test_the_held_audio_reaches_the_client_instead_of_being_dropped() -> None:
    """The DESIGN.md section 11 risk, as a regression: the turn's tail is not lost."""
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Order", metrics=DETACHED),
        speak(2, text=" 12345", end_of_turn=True, metrics=DETACHED),
        chunk(2, unit_id=1),
        chunk(2, unit_id=2),
        drained(2, unit_id=2),
    )
    assert len(events_of(session, AUDIO_DELTA)) == 2
    assert types_of(session).index("response.done") == len(types_of(session)) - 1


async def test_an_interrupt_closes_immediately_and_later_chunks_are_counted() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Let me", audio_waveform=None, metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, interrupted=True, metrics=DETACHED),
    )
    assert types_of(session)[-1] == "response.done"
    assert only_event(session, "response.done")["response"]["status"] == "cancelled"
    assert session.active_response_id is None

    before = len(types_of(session))
    await run(session, chunk(2, unit_id=2), drained(2, unit_id=2))
    assert len(types_of(session)) == before
    assert session.mapper.dropped_audio_chunks == 1


async def test_an_interrupt_while_the_close_is_held_cancels_the_response() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Hi", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, end_of_turn=True, metrics=DETACHED),
    )
    assert session.active_response_id is not None
    await run(session, speak(3, interrupted=True, metrics=DETACHED))
    assert only_event(session, "response.done")["response"]["status"] == "cancelled"
    assert session.active_response_id is None


async def test_the_next_turn_flushes_a_close_the_talker_never_released() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="one", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, end_of_turn=True, metrics=DETACHED),
        speak(3, text="two", metrics=DETACHED),
    )
    types = types_of(session)
    # The first response is closed before the second one opens, so the second unit's
    # transcript never joins the finished response.
    assert types == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE, *OPEN, TRANSCRIPT_DELTA]
    responses = {event["response"]["id"] for event in events_of(session, "response.created")}
    assert len(responses) == 2


async def test_the_drain_timeout_closes_a_response_the_talker_abandoned() -> None:
    """A Talker failure costs the tail of one turn, never the response."""
    session, _ = make_session()
    session.mapper.talker_drain_timeout_sec = 0.02
    await run(
        session,
        speak(1, text="Hi", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, end_of_turn=True, metrics=DETACHED),
    )
    assert session.active_response_id is not None
    await asyncio.sleep(0.1)
    assert types_of(session) == [*OPEN, TRANSCRIPT_DELTA, AUDIO_DELTA, *CLOSE]
    assert only_event(session, "response.done")["response"]["status"] == "completed"
    assert session.active_response_id is None


async def test_the_drain_timer_does_not_survive_the_response_it_guarded() -> None:
    session, _ = make_session()
    session.mapper.talker_drain_timeout_sec = 0.02
    await run(
        session,
        speak(1, text="Hi", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, end_of_turn=True, metrics=DETACHED),
        drained(2, unit_id=2),
        speak(3, text="next", metrics=DETACHED),
    )
    await asyncio.sleep(0.1)
    # The second response is still open: the first response's timer was cancelled.
    assert session.active_response_id is not None
    assert types_of(session).count("response.done") == 1


async def test_cancel_pending_close_drops_the_hold_without_emitting_it() -> None:
    session, _ = make_session()
    await run(
        session,
        speak(1, text="Hi", metrics=DETACHED),
        chunk(1, unit_id=1),
        speak(2, end_of_turn=True, metrics=DETACHED),
    )
    session.mapper.cancel_pending_close()
    await run(session, drained(2, unit_id=2))
    assert "response.done" not in types_of(session)
    assert session.active_response_id is not None
