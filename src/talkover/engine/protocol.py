"""The asyncio-facing engine interface, shared by the real engine and its fakes.

This module is deliberately import-light: it pulls in nothing but the standard library
and numpy. The realtime layer (and the fake engine the protocol tests drive) import it,
and neither may pay for torch, mcpmft or a model load just to type-check a call.

:class:`EngineProtocol` is the whole contract between ``talkover.realtime`` and
``talkover.engine``; :class:`EngineStepEvent` is the only value that crosses back. The
event mirrors the upstream ``DuplexStepEvent`` fields the mapping in DESIGN.md 5.3 needs,
with the conversion (``EngineSession.from_upstream``) living in ``session.py`` so that
nothing here has to know what upstream looks like.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "INPUT_SAMPLE_RATE",
    "OUTPUT_SAMPLE_RATE",
    "UNIT_BYTES",
    "UNIT_MS",
    "UNIT_SAMPLES",
    "WORKER_DELIVERY_TOPICS",
    "WORKER_DELIVERY_TYPE",
    "EngineError",
    "EngineProtocol",
    "EngineStepEvent",
]

#: The model consumes audio at 16 kHz; the realtime layer resamples 24 kHz input to it.
INPUT_SAMPLE_RATE = 16000
#: token2wav emits 24 kHz, which is also the external Realtime pcm16 rate.
OUTPUT_SAMPLE_RATE = 24000
#: One causal unit is one second of audio.
UNIT_MS = 1000
#: Samples in one unit at :data:`INPUT_SAMPLE_RATE`.
UNIT_SAMPLES = INPUT_SAMPLE_RATE * UNIT_MS // 1000
#: Bytes in one unit of little-endian mono pcm16.
UNIT_BYTES = UNIT_SAMPLES * 2

#: The ``type`` every runtime event carrying a Brain delivery must have.
#:
#: Upstream ``mcpmft.data.frontbrain_training`` recognises a worker delivery by this
#: value, and ``gander_runtime.duplex_bridge`` refuses a runtime event without it.
WORKER_DELIVERY_TYPE = "worker_delivery"

#: The ``topic`` values the model was trained on.
#:
#: ``DeliveryRecord.topic`` (``gander_runtime/coordination.py``) is this closed set, and
#: ``worker_delivery_response`` (``gander_runtime/lean_realtime.py``) copies it verbatim
#: into the envelope. ``final`` is the terminal one and is the only topic that may carry a
#: ``status``.
WORKER_DELIVERY_TOPICS = frozenset({"milestone", "interaction", "final", "risk", "aggregate"})


class EngineError(RuntimeError):
    """Raised when the engine cannot serve a call, or has failed terminally.

    A terminal failure on the inference thread is re-raised from
    :meth:`EngineProtocol.events` and leaves :attr:`EngineProtocol.ready` ``False``.
    """


@dataclass(frozen=True, slots=True)
class EngineStepEvent:
    """One model unit, as the realtime layer sees it.

    The fields mirror ``mcpmft.infer.realtime.DuplexStepEvent`` (upstream ``cf43838``),
    renamed only where upstream's name is ambiguous: upstream ``index`` becomes
    :attr:`unit_index`. ``audio_waveform`` is always a 1-D ``float32`` numpy array in
    ``[-1, 1]`` at :attr:`sample_rate`, or ``None`` when the unit produced no audio
    (every ``is_listen`` unit does, unless upstream is asked for listen audio).
    """

    #: Upstream ``index``: a 1-based counter over every step of the session.
    unit_index: int
    #: The model chose to keep listening rather than speak during this unit.
    is_listen: bool
    #: Incremental assistant text for this unit; maps to the transcript delta.
    text: str
    #: The assistant turn ended with this unit.
    end_of_turn: bool
    #: The model cut its own output short (barge-in).
    interrupted: bool
    #: 1-D float32 waveform for this unit, or ``None``.
    audio_waveform: np.ndarray | None = None
    #: Sample rate of :attr:`audio_waveform`.
    sample_rate: int = OUTPUT_SAMPLE_RATE
    #: Upstream ``current_time``: the model's own clock reading for the unit.
    current_time: int | None = None
    #: Upstream ``unit_id``: counts output (non-listen) units only; ``None`` while listening.
    unit_id: int | None = None
    #: Upstream ``generation_id``: bumped by the detached Talker; 0 in-process.
    generation_id: int = 0
    #: The unit carried a native tool call instead of speech.
    is_tool_call: bool = False
    #: Parsed native tool calls, when :attr:`is_tool_call`.
    tool_calls: tuple[Mapping[str, Any], ...] = ()
    #: Why a tool call could not be used, when :attr:`is_tool_call`.
    tool_error: str | None = None
    #: The model expects a tool response before the next unit.
    tool_response_expected: bool = False
    #: Upstream per-stage metrics (``cost_llm``, ``cost_tts``, ``talker`` state, ...).
    metrics: Mapping[str, Any] = field(default_factory=dict)
    #: This event carries nothing but Talker audio, produced for an earlier unit.
    #:
    #: The in-process Talker thread (T1.5) vocodes a unit after the Thinker step that
    #: produced it has already been reported, so its waveform arrives on its own event:
    #: :attr:`unit_id` and :attr:`generation_id` say which unit it belongs to, while
    #: :attr:`text`, :attr:`end_of_turn` and :attr:`is_listen` carry no information. Always
    #: ``False`` while the Talker runs inside the Thinker step, where the waveform rides
    #: the unit's own event.
    is_audio_chunk: bool = False
    #: This event is the Talker's drained marker for :attr:`generation_id`.
    #:
    #: The threaded Talker (T1.5) vocodes a unit after the Thinker step that produced it
    #: has been reported, so the waveform of the unit that ended a turn arrives *after*
    #: that turn's ``end_of_turn``. This marker says the Talker has no more audio for the
    #: turn: it is published once the Talker finishes the request carrying ``end_of_turn``,
    #: and once for a generation that was cancelled. It is always carried on an
    #: :attr:`is_audio_chunk` event with no waveform and no text; :attr:`generation_id` and
    #: :attr:`unit_id` name the turn. ``talkover.realtime.mapping`` holds the response close
    #: chain until it arrives (DESIGN.md 5.3). Never set while the Talker runs in step.
    talker_done: bool = False
    #: Wall time the engine spent producing this unit, in seconds (T1.8 reads it).
    step_wall_time_sec: float = 0.0

    @property
    def has_audio(self) -> bool:
        """Whether this unit carries a non-empty waveform."""
        return self.audio_waveform is not None and self.audio_waveform.size > 0


@runtime_checkable
class EngineProtocol(Protocol):
    """What ``talkover.realtime`` may call on an engine.

    Every method is a coroutine: the real engine hands the work to a dedicated inference
    thread and never blocks the event loop. Calls are serialized in submission order, so
    a ``feed_pcm16`` that has returned has already queued its :class:`EngineStepEvent`.
    """

    @property
    def ready(self) -> bool:
        """Whether the engine is loaded and able to accept calls (``GET /health``)."""
        ...

    async def start(self) -> None:
        """Load the model and start the inference thread. Idempotent."""
        ...

    async def feed_pcm16(self, pcm16_16k_1s: bytes | np.ndarray) -> None:
        """Feed exactly one 1 s unit of 16 kHz mono pcm16.

        Accepts :data:`UNIT_BYTES` bytes, an ``int16`` array of :data:`UNIT_SAMPLES`
        samples, or a ``float32`` array of :data:`UNIT_SAMPLES` samples in ``[-1, 1]``.
        Any other size raises :class:`ValueError`; partial units are the realtime
        layer's business, not the engine's.
        """
        ...

    async def flush_pending(self) -> None:
        """Run any audio left inside the engine, zero-padded to a whole unit."""
        ...

    async def interrupt_output(self) -> None:
        """End the current assistant output without pausing future units."""
        ...

    async def set_task_slate(self, text: str) -> None:
        """Replace the pinned task slate (DESIGN.md 5.5)."""
        ...

    async def submit_text_turn(self, text: str) -> None:
        """Queue a text user turn, delivered with the next audio unit."""
        ...

    async def feed_tool_response(self, response: Mapping[str, Any]) -> None:
        """Answer the native tool call the model is blocked on (DESIGN.md 4.6).

        Mirrors upstream ``DuplexLiveSession.feed_tool_response(response)``: the payload
        is the bounded front-brain result shape that
        ``gander_runtime.lean_realtime.control_tool_response`` builds (``status``,
        ``task_ids``, and optionally ``reason`` / ``content``), and it is prefilled inside
        ``<tool_response>`` markers with the next unit's microphone audio. The model stops
        producing units until it arrives, so it is only ever sent for a unit whose
        :attr:`EngineStepEvent.tool_response_expected` is set.

        Raises:
            EngineError: if upstream refuses the payload — no tool call is pending, or the
                response is longer than ``max_tool_response_tokens``. The refusal is not
                terminal: the session keeps running without the answer.
        """
        ...

    async def feed_worker_delivery(self, delivery: Mapping[str, Any]) -> None:
        """Hand the Cerebellum one Brain delivery to phrase in its own words.

        Mirrors upstream ``DuplexLiveSession.feed_runtime_event(event)`` with the
        ``worker_delivery`` envelope, which uses a slot separate from the synchronous tool
        response above. ``delivery`` is the payload
        ``gander_runtime.lean_realtime.worker_delivery_response`` produces —
        :data:`WORKER_DELIVERY_TYPE` as ``type``, the semantic ``task_name`` the model
        named in its ``task_start``, a ``topic`` from :data:`WORKER_DELIVERY_TOPICS`, the
        spoken hint in ``content``, and a terminal ``status`` only when ``topic`` is
        ``final``. ``type`` may be omitted and is filled in.

        Raises:
            ValueError: if ``type`` is present and is not :data:`WORKER_DELIVERY_TYPE`.
            EngineError: if upstream refuses the payload — a tool response is still
                pending, or the envelope exceeds ``max_tool_response_tokens``. Not
                terminal, same as above.
        """
        ...

    def events(self) -> AsyncIterator[EngineStepEvent]:
        """Iterate model units until :meth:`stop`; re-raises a terminal engine failure."""
        ...

    async def stop(self) -> None:
        """Close the session, join the inference thread and release the model."""
        ...
