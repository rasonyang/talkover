"""Approximate turn detection tests (T2.6).

Three trigger paths produce the two speech events (`docs/protocol-profile.md` section 7):
the ASR VAD's `SpeechStarted`, a retroactive start when an engine unit reports
`interrupted: true`, and the stop at the end of an ASR segment. `turn_detection: null`
disables both. The cases cover `TurnDetector` on its own and through `RealtimeSession`,
which is where the sink, the item ids and the engine pump come in.

GPU-free: no ASR backend is loaded, only the event types of `talkover.engine.asr`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from talkover.engine.asr import SpeechStarted, SpeechStopped, Transcript
from talkover.realtime import events as ev
from talkover.realtime.session import DEFAULT_TURN_DETECTION
from talkover.realtime.turn_detection import TURN_DETECTION_TYPES, TurnDetector, TurnSettings

from .fake_engine import FakeEngine, step_event
from .test_session import drop, emitted, make_session, only, pcm24, session_update, types_of

PROFILE = Path(__file__).resolve().parents[2] / "docs" / "protocol-profile.md"

SPEECH_STARTED = "input_audio_buffer.speech_started"
SPEECH_STOPPED = "input_audio_buffer.speech_stopped"


def wire(produced: list[ev.ServerEvent]) -> list[dict[str, Any]]:
    """Render what a detector call returned, dropping the per-event ids."""
    return [{k: v for k, v in event.to_wire().items() if k != "event_id"} for event in produced]


def speech_events(session: Any) -> list[dict[str, Any]]:
    return [
        event for event in emitted(session) if event["type"] in (SPEECH_STARTED, SPEECH_STOPPED)
    ]


# ---------------------------------------------------------------------------
# settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", TURN_DETECTION_TYPES)
def test_both_vad_types_are_one_implementation(kind: str) -> None:
    detector = TurnDetector({"type": kind})
    assert detector.enabled
    assert detector.settings is not None
    assert detector.settings.type == kind
    started = detector.on_asr_event(SpeechStarted(at_ms=400))
    assert [event.TYPE for event in started] == [SPEECH_STARTED]


def test_settings_carry_the_echoed_fields_without_acting_on_them() -> None:
    settings = TurnSettings.from_session(
        {
            "type": "server_vad",
            "threshold": 0.7,
            "prefix_padding_ms": 120,
            "silence_duration_ms": 900,
            "create_response": False,
            "interrupt_response": False,
            "idle_timeout_ms": 5000,
        }
    )
    assert settings == TurnSettings(
        type="server_vad",
        threshold=0.7,
        prefix_padding_ms=120,
        silence_duration_ms=900,
        create_response=False,
        interrupt_response=False,
        idle_timeout_ms=5000,
    )
    # `create_response: false` and `interrupt_response: false` are ignored: the detector
    # still reports the same boundaries, because it never drives a response.
    detector = TurnDetector(
        {"type": "server_vad", "create_response": False, "interrupt_response": False}
    )
    assert [event.TYPE for event in detector.on_asr_event(SpeechStarted(at_ms=0))] == [
        SPEECH_STARTED
    ]
    assert [event.TYPE for event in detector.on_asr_event(SpeechStopped(at_ms=800))] == [
        SPEECH_STOPPED
    ]


def test_semantic_vad_eagerness_is_carried() -> None:
    settings = TurnSettings.from_session({"type": "semantic_vad", "eagerness": "high"})
    assert settings is not None
    assert (settings.type, settings.eagerness) == ("semantic_vad", "high")


# ---------------------------------------------------------------------------
# path 1: the ASR VAD signal
# ---------------------------------------------------------------------------


def test_asr_speech_started_emits_speech_started() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION, new_item_id=lambda: "item_speech")
    produced = detector.on_asr_event(SpeechStarted(at_ms=1240))
    assert wire(produced) == [
        {"type": SPEECH_STARTED, "audio_start_ms": 1240, "item_id": "item_speech"}
    ]
    assert detector.in_speech


def test_a_second_speech_started_does_not_repeat_the_event() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    assert detector.on_asr_event(SpeechStarted(at_ms=100))
    assert detector.on_asr_event(SpeechStarted(at_ms=300)) == []


async def test_session_emits_speech_started_from_the_asr_signal() -> None:
    session, _ = make_session()
    drop(session)
    await session.on_asr_event(SpeechStarted(at_ms=640))
    assert types_of(session) == [SPEECH_STARTED]
    event = speech_events(session)[0]
    assert event["audio_start_ms"] == 640
    assert event["item_id"].startswith("item_")


# ---------------------------------------------------------------------------
# path 2: retroactive start on `interrupted: true`
# ---------------------------------------------------------------------------


def test_interrupted_unit_opens_the_span_at_the_start_of_that_unit() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION, new_item_id=lambda: "item_bargein")
    produced = detector.on_engine_event(step_event(4, is_listen=False, interrupted=True))
    assert wire(produced) == [
        {"type": SPEECH_STARTED, "audio_start_ms": 3000, "item_id": "item_bargein"}
    ]
    assert detector.in_speech


def test_a_unit_without_interrupted_produces_nothing() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    assert detector.on_engine_event(step_event(2, is_listen=False, text="hello")) == []
    assert detector.on_engine_event(step_event(3, end_of_turn=True)) == []
    assert not detector.in_speech


def test_a_late_talker_audio_chunk_is_no_anchor() -> None:
    # `is_audio_chunk` events carry audio for an earlier unit; their flags say nothing.
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    chunk = step_event(7, is_audio_chunk=True, interrupted=True)
    assert detector.on_engine_event(chunk) == []
    assert not detector.in_speech


def test_the_first_unit_cannot_anchor_before_zero() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    (started,) = detector.on_engine_event(step_event(1, is_listen=False, interrupted=True))
    assert started.to_wire()["audio_start_ms"] == 0


def test_a_barge_in_during_detected_speech_does_not_repeat_the_start() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    assert detector.on_asr_event(SpeechStarted(at_ms=2000))
    assert detector.on_engine_event(step_event(3, is_listen=False, interrupted=True)) == []


async def test_session_pump_emits_the_retroactive_start_before_the_mapping_sees_the_unit() -> None:
    seen: list[str] = []

    async def record(step: Any) -> None:
        seen.append(f"handler:{step.unit_index}")

    engine = FakeEngine()
    session, _ = make_session(engine, on_engine_event=record)
    drop(session)
    await engine.push(step_event(5, is_listen=False, interrupted=True))
    await engine.stop()
    await session.pump_engine_events()

    # The mapper (T2.5) turns the same unit into the `response.*` events that follow; what
    # this case pins down is that the retroactive start comes before all of them.
    assert types_of(session)[0] == SPEECH_STARTED
    assert seen == ["handler:5"]
    assert speech_events(session)[0]["audio_start_ms"] == 4000


# ---------------------------------------------------------------------------
# path 3: the stop at the end of an ASR segment
# ---------------------------------------------------------------------------


def test_asr_speech_stopped_closes_the_span_with_the_same_item_id() -> None:
    ids = iter(["item_one", "item_two"])
    detector = TurnDetector(DEFAULT_TURN_DETECTION, new_item_id=lambda: next(ids))
    detector.on_asr_event(SpeechStarted(at_ms=500))
    produced = detector.on_asr_event(SpeechStopped(at_ms=2100))
    assert wire(produced) == [{"type": SPEECH_STOPPED, "audio_end_ms": 2100, "item_id": "item_one"}]
    assert not detector.in_speech


def test_a_stop_without_a_start_produces_nothing() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    assert detector.on_asr_event(SpeechStopped(at_ms=900)) == []


def test_a_transcript_closes_a_span_the_backend_left_open() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    detector.on_asr_event(SpeechStarted(at_ms=0))
    (stopped,) = detector.on_asr_event(Transcript(text="hello", start_ms=0, end_ms=1500))
    assert stopped.to_wire()["audio_end_ms"] == 1500
    # The usual order is stop-then-transcript, and the transcript adds nothing.
    detector.on_asr_event(SpeechStarted(at_ms=2000))
    detector.on_asr_event(SpeechStopped(at_ms=3000))
    assert detector.on_asr_event(Transcript(text="bye", start_ms=2000, end_ms=3000)) == []


def test_a_stop_is_never_reported_before_its_retroactive_start() -> None:
    # The engine unit clock and the ASR timeline are two approximations of one stream.
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    detector.on_engine_event(step_event(9, is_listen=False, interrupted=True))
    (stopped,) = detector.on_asr_event(SpeechStopped(at_ms=1200))
    assert stopped.to_wire()["audio_end_ms"] == 8000


async def test_session_emits_a_full_span_around_a_barge_in() -> None:
    engine = FakeEngine()
    session, _ = make_session(engine)
    drop(session)
    await session.detect_turn(step_event(2, is_listen=False, interrupted=True))
    await session.on_asr_event(SpeechStopped(at_ms=2400))
    started, stopped = speech_events(session)
    assert started["type"] == SPEECH_STARTED and started["audio_start_ms"] == 1000
    assert stopped["type"] == SPEECH_STOPPED and stopped["audio_end_ms"] == 2400
    assert started["item_id"] == stopped["item_id"]


# ---------------------------------------------------------------------------
# item ids
# ---------------------------------------------------------------------------


def test_take_item_id_hands_the_span_id_over_once() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION, new_item_id=lambda: "item_span")
    assert detector.take_item_id() is None
    detector.on_asr_event(SpeechStarted(at_ms=0))
    assert detector.take_item_id() == "item_span"
    assert detector.take_item_id() is None


async def test_the_committed_item_carries_the_detected_span_id() -> None:
    session, _ = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(1.0)})
    await session.on_asr_event(SpeechStarted(at_ms=0))
    await session.on_asr_event(SpeechStopped(at_ms=1000))
    drop(session)
    await session.handle_raw({"type": "input_audio_buffer.commit"})

    committed = only(session, "input_audio_buffer.committed")
    assert committed["item_id"] == session.input_item_id
    assert session.turn_detector.take_item_id() is None


async def test_a_commit_without_a_detected_span_still_gets_an_item_id() -> None:
    session, _ = make_session()
    await session.handle_raw({"type": "input_audio_buffer.append", "audio": pcm24(1.0)})
    await session.handle_raw({"type": "input_audio_buffer.commit"})
    assert session.input_item_id is not None
    assert session.input_item_id.startswith("item_")


# ---------------------------------------------------------------------------
# `turn_detection: null`
# ---------------------------------------------------------------------------


def test_null_turn_detection_disables_both_events() -> None:
    detector = TurnDetector(None)
    assert not detector.enabled
    assert detector.on_asr_event(SpeechStarted(at_ms=100)) == []
    assert detector.on_asr_event(SpeechStopped(at_ms=900)) == []
    assert detector.on_engine_event(step_event(3, is_listen=False, interrupted=True)) == []
    assert not detector.in_speech
    assert detector.take_item_id() is None


def test_clearing_turn_detection_drops_an_open_span() -> None:
    detector = TurnDetector(DEFAULT_TURN_DETECTION)
    detector.on_asr_event(SpeechStarted(at_ms=0))
    detector.configure(None)
    assert not detector.in_speech
    assert detector.on_asr_event(SpeechStopped(at_ms=800)) == []
    detector.configure({"type": "semantic_vad"})
    assert [e.TYPE for e in detector.on_asr_event(SpeechStarted(at_ms=1000))] == [SPEECH_STARTED]


async def test_session_update_null_silences_every_trigger() -> None:
    engine = FakeEngine()
    session, _ = make_session(engine)
    await session.handle_raw(session_update(audio={"input": {"turn_detection": None}}))
    assert session.turn_detection is None
    drop(session)

    await session.on_asr_event(SpeechStarted(at_ms=200))
    await session.on_asr_event(SpeechStopped(at_ms=1200))
    await session.detect_turn(step_event(4, is_listen=False, interrupted=True))
    assert types_of(session) == []


async def test_turn_detection_is_on_by_default_and_can_be_restored() -> None:
    session, _ = make_session()
    assert session.turn_detector.enabled
    await session.handle_raw(session_update(audio={"input": {"turn_detection": None}}))
    assert not session.turn_detector.enabled
    await session.handle_raw(
        session_update(audio={"input": {"turn_detection": {"type": "semantic_vad"}}})
    )
    assert session.turn_detector.enabled
    drop(session)
    await session.on_asr_event(SpeechStarted(at_ms=0))
    assert types_of(session) == [SPEECH_STARTED]


# ---------------------------------------------------------------------------
# the documented approximation
# ---------------------------------------------------------------------------


def test_profile_documents_the_three_trigger_paths() -> None:
    text = PROFILE.read_text(encoding="utf-8")
    section = text.split("## 7. Turn detection", 1)[1].split("\n## ", 1)[0]
    for phrase in (
        "`input_audio_buffer.speech_started`",
        "`input_audio_buffer.speech_stopped`",
        "retroactively",
        "`turn_detection: null`",
        "approximate",
    ):
        assert phrase in section, f"profile section 7 no longer documents {phrase}"
