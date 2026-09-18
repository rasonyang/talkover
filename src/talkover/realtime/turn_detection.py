"""Approximate turn detection: synthesizing `input_audio_buffer.speech_started` / `_stopped`.

Gander has no explicit VAD boundary (DESIGN.md 5.4), so the two speech events of the
Realtime protocol cannot be derived from the model. `docs/protocol-profile.md` section 7
documents what Talkover does instead, and this module implements exactly that:

- `server_vad` and `semantic_vad` select the same implementation. Their tuning fields
  (`threshold`, `prefix_padding_ms`, `silence_duration_ms`, `eagerness`) are validated and
  echoed by `events.py`; only the ASR side channel's VAD can act on them, so they are
  carried here for inspection and never change this module's behaviour.
- `create_response`, `interrupt_response` and `idle_timeout_ms` are accepted and ignored:
  the model decides on its own when to speak and when to stop, and Talkover never emits
  `input_audio_buffer.timeout_triggered`.
- `turn_detection: null` disables both events.

Three signals reach the detector, and every one of them is pushed in by
`talkover.realtime.session`:

1. `SpeechStarted` from the ASR backend's `EnergyVad` (T1.6) opens a speech span.
2. An `EngineStepEvent` with `interrupted: true` opens one retroactively, because a
   barge-in is the model telling us, after the fact, that the user had started talking.
   The anchor is the start of the interrupted unit, `(unit_index - 1) * UNIT_MS`.
3. `SpeechStopped` (or, defensively, the `Transcript` that follows it) closes the span at
   the end of the ASR segment.

The detector is pure: it holds no session state, performs no I/O and returns the server
events it wants emitted rather than sending them. `RealtimeSession` owns the sink.

`item_id`: the Realtime protocol expects both speech events to name the user item the
audio will end up in. Talkover creates that item at `input_audio_buffer.commit`, which
happens after the speech span, so the detector mints the id up front and the session takes
it for the committed item (:meth:`TurnDetector.take_item_id`). A span that is never
committed simply leaves an unused id behind, and when two spans precede one commit the
committed item carries the last span's id.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from talkover.engine.asr import AsrEvent, SpeechStarted, SpeechStopped, Transcript
from talkover.engine.protocol import UNIT_MS, EngineStepEvent
from talkover.realtime import events as ev

__all__ = ["TURN_DETECTION_TYPES", "TurnDetector", "TurnSettings"]

#: The two `turn_detection.type` values the profile accepts; both behave identically.
TURN_DETECTION_TYPES = ("server_vad", "semantic_vad")


@dataclass(frozen=True, slots=True)
class TurnSettings:
    """The effective `audio.input.turn_detection`, already validated by `events.py`.

    Every field is kept for inspection only. The detector reads none of them: with no VAD
    of its own it has nothing to tune, and the ASR backend owns the thresholds.
    """

    type: str = "server_vad"
    threshold: float | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    eagerness: str | None = None
    create_response: bool | None = None
    interrupt_response: bool | None = None
    idle_timeout_ms: int | None = None

    @classmethod
    def from_session(cls, value: Mapping[str, Any] | None) -> TurnSettings | None:
        """Build settings from `session.audio.input.turn_detection`; `None` stays `None`."""
        if value is None:
            return None
        return cls(
            type=str(value.get("type", "server_vad")),
            threshold=_opt_float(value.get("threshold")),
            prefix_padding_ms=_opt_int(value.get("prefix_padding_ms")),
            silence_duration_ms=_opt_int(value.get("silence_duration_ms")),
            eagerness=_opt_str(value.get("eagerness")),
            create_response=_opt_bool(value.get("create_response")),
            interrupt_response=_opt_bool(value.get("interrupt_response")),
            idle_timeout_ms=_opt_int(value.get("idle_timeout_ms")),
        )


class TurnDetector:
    """Turn ASR VAD signals and model barge-ins into the two speech events.

    Construct one per session, keep it in step with `session.update` through
    :meth:`configure`, and feed it with :meth:`on_asr_event` and :meth:`on_engine_event`.
    Both return the server events to emit, in order, and both return nothing while turn
    detection is disabled.
    """

    def __init__(
        self,
        turn_detection: Mapping[str, Any] | None = None,
        *,
        new_item_id: Callable[[], str] = ev.new_item_id,
    ) -> None:
        self._new_item_id = new_item_id
        self.settings = TurnSettings.from_session(turn_detection)
        self._in_speech = False
        self._item_id: str | None = None
        self._pending_commit_id: str | None = None
        self._started_ms = 0

    # -- state -------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        """Whether `turn_detection` is configured; `null` disables both events."""
        return self.settings is not None

    @property
    def in_speech(self) -> bool:
        """Whether a speech span is currently open."""
        return self._in_speech

    @property
    def item_id(self) -> str | None:
        """The item id of the open speech span, or `None` when no span is open."""
        return self._item_id

    def configure(self, turn_detection: Mapping[str, Any] | None) -> None:
        """Apply a `session.update`; clearing `turn_detection` also drops the open span."""
        self.settings = TurnSettings.from_session(turn_detection)
        if self.settings is None:
            self.reset()

    def reset(self) -> None:
        """Forget the open span and any item id that was minted for it."""
        self._in_speech = False
        self._item_id = None
        self._pending_commit_id = None
        self._started_ms = 0

    def take_item_id(self) -> str | None:
        """Hand the id minted for the latest speech span to the committed audio item.

        Returns `None` when no span has been detected since the last commit, which is when
        the session mints its own id. Consumed once, so two commits never share an id.
        """
        item_id, self._pending_commit_id = self._pending_commit_id, None
        return item_id

    # -- signals -----------------------------------------------------------

    def on_asr_event(self, event: AsrEvent) -> list[ev.ServerEvent]:
        """Map one ASR stream event to the speech events it implies."""
        if isinstance(event, SpeechStarted):
            return self._start(event.at_ms)
        if isinstance(event, SpeechStopped):
            return self._stop(event.at_ms)
        if isinstance(event, Transcript):
            # `AsrStream` emits `SpeechStopped` before the `Transcript`, so the span is
            # normally already closed; this only catches a backend that skips the stop.
            return self._stop(event.end_ms)
        return []

    def on_engine_event(self, step: EngineStepEvent) -> list[ev.ServerEvent]:
        """Open a speech span retroactively when the model reports a barge-in.

        An `is_audio_chunk` event is late Talker audio for an earlier unit, so its unit
        index is no anchor and its flags carry no information; it is ignored.
        """
        if step.is_audio_chunk or not step.interrupted:
            return []
        return self._start(max(0, (step.unit_index - 1) * UNIT_MS))

    # -- internals ---------------------------------------------------------

    def _start(self, at_ms: int) -> list[ev.ServerEvent]:
        if self.settings is None or self._in_speech:
            return []
        self._in_speech = True
        self._started_ms = max(0, int(at_ms))
        self._item_id = self._new_item_id()
        self._pending_commit_id = self._item_id
        return [
            ev.InputAudioBufferSpeechStarted(audio_start_ms=self._started_ms, item_id=self._item_id)
        ]

    def _stop(self, at_ms: int) -> list[ev.ServerEvent]:
        if self.settings is None or not self._in_speech:
            return []
        self._in_speech = False
        item_id = self._item_id or self._new_item_id()
        self._item_id = None
        # The ASR timeline and the engine's unit clock are two approximations of the same
        # audio, so a retroactive start can sit after the stop it is paired with.
        end_ms = max(self._started_ms, int(at_ms))
        return [ev.InputAudioBufferSpeechStopped(audio_end_ms=end_ms, item_id=item_id)]


def _opt_int(value: Any) -> int | None:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _opt_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None
