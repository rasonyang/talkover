"""Wraps the upstream ``DuplexLiveSession`` and exposes the Talkover engine interface.

Two things live here.

**The import-order hook (T1.3).** Importing ``talkover.engine.session`` applies the
upstream monkeypatches (``talkover.engine.patches``) before anything imports
``mcpmft.infer``. Any module that touches upstream inference code must therefore import
this module first. The YAML configuration is not available at import time, so the backend
used for that first call is a provisional one: ``TALKOVER_DEVICE`` when it is set (``mps``,
``cuda``, ``cuda:<n>`` or ``cpu``), otherwise the first available device in the order
``mps``, ``cuda``, ``cpu``. :class:`EngineSession` calls ``apply_patches`` again with the
configured backend before it loads the model, which retargets the installed patches.

**The engine session (T1.4).** :class:`EngineSession` implements
:class:`talkover.engine.protocol.EngineProtocol`. It builds the upstream settings from the
typed :class:`~talkover.config.TalkoverConfig` directly — ``gander_runtime/cli.py`` is not
used (DESIGN.md 4.1) — and runs every model call on one dedicated inference thread:

- the event loop puts commands on a ``queue.Queue`` and awaits an ``asyncio.Future``;
- the thread runs the upstream call and pushes :class:`EngineStepEvent` values back
  through ``loop.call_soon_threadsafe`` into an ``asyncio.Queue``;
- nothing that touches the model ever runs on the event loop.

Commands are served in submission order, so a ``feed_pcm16`` that has returned has already
queued its event. Any exception on the inference thread is terminal: it fails the pending
call, is re-raised from :meth:`EngineSession.events`, leaves ``ready`` ``False``, and the
thread closes the upstream session and exits.

**The Cerebellum channel (T1.4).** ``feed_tool_response`` and ``feed_worker_delivery`` are
the way back into the model for what must never reach the Realtime client: the bounded
answer to a native tool call the model is blocked on, and a Brain delivery for it to phrase
itself (DESIGN.md 4.6). They are ordinary commands on the same thread, with one exception
to the rule below: neither runs the model, so upstream refusing one is reported to the
caller and logged instead of ending the session.

**The Talker seam (T1.5).** ``talker_factory`` moves the Talker and token2wav onto their
own thread (:mod:`talkover.engine.talker`). The inference thread then only hands over speak
tokens, and the waveforms come back asynchronously as ``is_audio_chunk`` events, drained at
each step boundary and once more at shutdown. Without the factory the Talker keeps running
inside the upstream step, which is what T1.4 validated.

Device rules (DESIGN.md 4.2): this module names no device-specific torch attribute. All
device work goes through :class:`~talkover.engine.backend.DeviceBackend` — ``device()`` for
placement and ``synchronize()`` at the step boundary where the per-unit timing is read.
"""

from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from talkover.config import TalkoverConfig
from talkover.engine.backend import DeviceBackend, get_backend
from talkover.engine.patches import apply_patches, speech_worker_override
from talkover.engine.protocol import (
    OUTPUT_SAMPLE_RATE,
    UNIT_BYTES,
    UNIT_SAMPLES,
    WORKER_DELIVERY_TYPE,
    EngineError,
    EngineStepEvent,
)
from talkover.engine.talker import (
    TalkerChunk,
    TalkerDone,
    TalkerFailed,
    TalkerInterrupted,
    TalkerThread,
)

__all__ = [
    "DEVICE_ENV_VAR",
    "EngineSession",
    "build_duplex_params",
    "build_live_config",
    "build_live_session",
    "build_model_arguments",
    "default_backend",
    "default_token2wav_dir",
]

LOGGER = logging.getLogger(__name__)

DEVICE_ENV_VAR = "TALKOVER_DEVICE"
_AUTO_DETECT_ORDER = ("mps", "cuda", "cpu")

#: ``engine.media_mode`` uses the Talkover spelling; upstream calls the video mode "omni".
_MEDIA_MODES = {"voice": "voice", "video": "omni"}

#: token2wav assets ship inside the MiniCPM-o base model checkpoint.
_TOKEN2WAV_SUBDIR = ("assets", "token2wav")

_STOP_JOIN_TIMEOUT_SEC = 30.0

#: Commands whose failure is reported to the caller but does not end the session.
#:
#: Both only queue a payload for the next unit: upstream validates and tokenizes it and
#: never touches the model, so a refusal ("no pending tool call", or an envelope past
#: ``max_tool_response_tokens``) says nothing about the model's health. Dropping one Brain
#: delivery is a smaller loss than dropping the call (DESIGN.md 4.6).
_REFUSABLE_COMMANDS = frozenset({"tool_response", "worker_delivery"})


def default_backend() -> DeviceBackend:
    """The backend to patch against before the configuration is known.

    ``TALKOVER_DEVICE`` wins when set; otherwise the first available device in
    ``mps``, ``cuda``, ``cpu`` is used, with ``cpu`` as the final fallback.

    Raises:
        ValueError: if ``TALKOVER_DEVICE`` names an unknown device.
    """
    requested = os.environ.get(DEVICE_ENV_VAR, "").strip()
    if requested:
        return get_backend(requested)

    for name in _AUTO_DETECT_ORDER:
        backend = get_backend(name)
        try:
            available = backend.is_available()
        except Exception:  # noqa: BLE001 - a probe must never break the import
            available = False
        if available:
            return backend
    return get_backend("cpu")


apply_patches(default_backend())


# --------------------------------------------------------------------------------------
# Upstream settings, built from the typed config (never through gander_runtime/cli.py)
# --------------------------------------------------------------------------------------


def default_token2wav_dir(config: TalkoverConfig) -> str | None:
    """Where the token2wav assets sit, given the configuration.

    ``engine.token2wav_dir`` wins when it is set. It is null by default, because the
    directory is not a separate download: it is ``<base_model>/assets/token2wav`` in the
    MiniCPM-o checkpoint. Upstream's ``ModelArguments.token2wav_dir`` has no default and
    the Talker stays silent without it. Returns ``None`` when the resolved directory is
    absent, which is also what the caller passes to skip ``init_tts``.
    """
    engine = config.engine
    if engine.token2wav_dir:
        candidate = Path(engine.token2wav_dir).expanduser()
    elif engine.base_model:
        candidate = Path(engine.base_model).expanduser().joinpath(*_TOKEN2WAV_SUBDIR)
    else:
        return None
    return str(candidate) if candidate.is_dir() else None


def build_model_arguments(config: TalkoverConfig, *, token2wav_dir: str | None = None) -> Any:
    """Build upstream ``mcpmft.args.ModelArguments`` from the Talkover config.

    ``init_vision`` is forced off under ``media_mode: voice`` (DESIGN.md 4.5 counts the
    vision encoder only when it is loaded), ``device_map`` stays ``None`` so upstream's
    single ``model.to(...)`` — retargeted by the T1.3 patch — does the placement, and
    ``attn_implementation`` and ``torch_dtype`` come straight from the config.
    """
    from mcpmft.args import ModelArguments

    engine = config.engine
    voice_only = engine.media_mode == "voice"
    return ModelArguments(
        model_name_or_path=engine.base_model,
        trust_remote_code=True,
        torch_dtype=engine.dtype,
        attn_implementation=engine.attn_implementation,
        init_vision=False if voice_only else engine.init_vision,
        init_audio=True,
        init_tts=True,
        token2wav_dir=token2wav_dir,
        device_map=None,
    )


def build_duplex_params(config: TalkoverConfig) -> Any:
    """Build upstream ``mcpmft.infer.online.DuplexParams`` from the Talkover config.

    Only the knobs Talkover exposes are overridden; everything else keeps the upstream
    default, which is what the checkpoints were trained with. ``generate_audio`` is on, so
    the Talker runs inside the step; when a Talker thread is given (T1.5), upstream turns
    it off itself and takes the speak tokens out through that thread's queue instead.
    """
    from mcpmft.infer.online import DuplexParams

    return DuplexParams(
        generate_audio=True,
        sliding_window_mode=config.engine.sliding_window_mode,
        context_max_units=config.engine.context_max_units,
        memory_slate_max_tokens=config.realtime.memory_slate_max_tokens,
    )


def build_live_config(config: TalkoverConfig) -> Any:
    """Build upstream ``mcpmft.infer.realtime.DuplexLiveConfig`` from the Talkover config.

    ``stop_on_turn_end`` is disabled: a Realtime session is driven by the client's audio
    stream and must not close itself when the model finishes a turn.
    """
    from mcpmft.infer.realtime import DuplexLiveConfig

    return DuplexLiveConfig(
        trailing_silence_sec=config.realtime.trailing_silence_sec,
        stop_on_turn_end=False,
        send_listen_audio=False,
        media_mode=_MEDIA_MODES[config.engine.media_mode],
    )


def build_live_session(
    config: TalkoverConfig,
    backend: DeviceBackend,
    *,
    system_prompt: str | None = None,
    tools: Sequence[Mapping[str, Any]] | None = None,
    ref_audio_path: str | None = None,
    token2wav_dir: str | None = None,
    talker_factory: Callable[[Any, DeviceBackend], TalkerThread] | None = None,
) -> Any:
    """Load the checkpoints and construct an upstream ``DuplexLiveSession``.

    This runs on the inference thread and is the only place that loads weights. It calls
    ``apply_patches(backend)`` first so the T1.3 patches target the configured device
    rather than the one auto-detected at import time, then ``load_for_infer`` (which the
    patch places on ``backend.device()``), then builds the session.

    Without ``talker_factory`` the Talker and token2wav run inside the Thinker step, on
    the inference thread, and the waveform comes back on the step event (upstream
    ``generate_audio=True``, ``detached_talker=None``).

    With ``talker_factory`` (T1.5) the Talker runs on its own thread: the factory is called
    here, on the inference thread, with the loaded Thinker model and the backend, and
    returns a :class:`~talkover.engine.talker.TalkerThread`. Upstream then switches to its
    detached path — speak tokens are handed over through the thread's queue instead of
    being vocoded inline — and :func:`talkover.engine.patches.speech_worker_override` makes
    it adopt the already-running thread rather than building one of its own.

    ``engine.talker_device`` naming a different device is the M4 branch and is refused by
    :class:`EngineSession` and by ``talkover.engine.talker`` before this function is
    reached.
    """
    apply_patches(backend)

    from mcpmft.infer.common import load_for_infer
    from mcpmft.infer.realtime import DuplexLiveSession
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT

    engine = config.engine
    if token2wav_dir is None:
        token2wav_dir = default_token2wav_dir(config)
    if token2wav_dir is None:
        LOGGER.warning(
            "no token2wav assets at %s; the Talker will not produce audio",
            engine.token2wav_dir or f"{engine.base_model}/{'/'.join(_TOKEN2WAV_SUBDIR)}",
        )

    model_args = build_model_arguments(config, token2wav_dir=token2wav_dir)
    bundle = load_for_infer(
        model_args,
        checkpoint=engine.thinker_checkpoint or None,
        talker_checkpoint=engine.talker_checkpoint or None,
        init_token2wav=token2wav_dir is not None,
    )
    kwargs: dict[str, Any] = {
        "params": build_duplex_params(config),
        "system_prompt": system_prompt or GANDER_DUPLEX_SYSTEM_PROMPT,
        "ref_audio_path": ref_audio_path,
        "config": build_live_config(config),
        "tools": tools,
    }
    if talker_factory is None:
        return DuplexLiveSession(bundle, detached_talker=None, **kwargs)

    talker = talker_factory(bundle.model, backend)
    try:
        with speech_worker_override(talker):
            return DuplexLiveSession(bundle, detached_talker=talker.runtime, **kwargs)
    except BaseException:
        talker.close()
        raise


# --------------------------------------------------------------------------------------
# Input normalization
# --------------------------------------------------------------------------------------


def _unit_to_pcm16_bytes(unit: bytes | bytearray | memoryview | np.ndarray) -> bytes:
    """Validate one 1 s unit and return it as little-endian mono pcm16 bytes.

    Raises:
        TypeError: if ``unit`` is neither a bytes-like object nor a numpy array.
        ValueError: if it does not hold exactly one unit, or has an unusable dtype.
    """
    if isinstance(unit, np.ndarray):
        if unit.ndim != 1:
            raise ValueError(f"audio unit must be 1-D, got shape {unit.shape}")
        if unit.size != UNIT_SAMPLES:
            raise ValueError(
                f"audio unit must hold exactly {UNIT_SAMPLES} samples of 16 kHz audio, "
                f"got {unit.size}"
            )
        if unit.dtype == np.int16:
            return unit.astype("<i2", copy=False).tobytes()
        if unit.dtype.kind == "f":
            scaled = np.clip(unit.astype(np.float32, copy=False), -1.0, 1.0)
            pcm = np.where(scaled < 0, scaled * 32768.0, scaled * 32767.0)
            return pcm.astype("<i2").tobytes()
        raise ValueError(f"audio unit must be int16 or floating point, got dtype {unit.dtype}")

    if isinstance(unit, (bytes, bytearray, memoryview)):
        raw = bytes(unit)
        if len(raw) != UNIT_BYTES:
            raise ValueError(
                f"audio unit must hold exactly {UNIT_BYTES} bytes of 16 kHz pcm16, got {len(raw)}"
            )
        return raw

    raise TypeError(f"audio unit must be bytes or a numpy array, got {type(unit).__name__}")


def _as_float32_waveform(value: Any) -> np.ndarray | None:
    """Normalize an upstream waveform (numpy, torch tensor or sequence) to 1-D float32."""
    if value is None:
        return None
    detach = getattr(value, "detach", None)
    if callable(detach):  # a torch tensor, without naming torch here
        value = detach().to("cpu").numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return array if array.size else None


# --------------------------------------------------------------------------------------
# The command plumbing between the event loop and the inference thread
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Command:
    """One unit of work for the inference thread, with the future awaiting it."""

    kind: str
    payload: Any = None
    future: asyncio.Future[None] | None = None


class _Sentinel:
    """Marks the end of the event stream."""


_EVENTS_DONE = _Sentinel()


class EngineSession:
    """The Gander inference session, implementing :class:`EngineProtocol`.

    Args:
        config: the loaded Talkover configuration.
        backend: the device backend; defaults to ``get_backend(config.engine.device)``.
        session_factory: builds the upstream session. Injected by the tests so no weights
            are loaded; the default is :func:`build_live_session` with this config.
        system_prompt: overrides the upstream Gander duplex system prompt.
        tools: native tool schemas passed to upstream ``DuplexLiveSession``.
        ref_audio_path: reference wav for the Talker voice; overrides
            ``engine.ref_audio_path``. Upstream keeps the checkpoint's own voice when both
            are ``None``.
        token2wav_dir: overrides ``engine.token2wav_dir`` and, through it,
            :func:`default_token2wav_dir`.
        talker_factory: T1.5. When given, it is called on the inference thread with the
            loaded Thinker model and the backend and returns a
            :class:`~talkover.engine.talker.TalkerThread`, which then runs the Talker and
            token2wav off the Thinker's thread;
            :func:`talkover.engine.talker.talker_factory_from_config` builds the default
            one. When ``None`` (still the default this phase) the Talker runs inside the
            upstream step and its waveform rides the unit's own event.
    """

    def __init__(
        self,
        config: TalkoverConfig,
        *,
        backend: DeviceBackend | None = None,
        session_factory: Callable[[], Any] | None = None,
        system_prompt: str | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
        ref_audio_path: str | None = None,
        token2wav_dir: str | None = None,
        talker_factory: Callable[[Any, DeviceBackend], TalkerThread] | None = None,
    ) -> None:
        engine = config.engine
        if engine.talker_device is not None and engine.talker_device != engine.device:
            raise NotImplementedError(
                "engine.talker_device pointing at a second device needs the upstream "
                f"detached Talker, which is the M4 multi-GPU branch (T1.5); got "
                f"{engine.talker_device!r} with engine.device={engine.device!r}"
            )
        if ref_audio_path is None and engine.ref_audio_path:
            ref_audio_path = str(Path(engine.ref_audio_path).expanduser())

        self._config = config
        self._backend = backend if backend is not None else get_backend(engine.device)
        self._session_factory = session_factory or (
            lambda: build_live_session(
                config,
                self._backend,
                system_prompt=system_prompt,
                tools=tools,
                ref_audio_path=ref_audio_path,
                token2wav_dir=token2wav_dir,
                talker_factory=talker_factory,
            )
        )

        self._commands: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._live: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._events: asyncio.Queue[EngineStepEvent | BaseException | _Sentinel] | None = None
        self._start_future: asyncio.Future[None] | None = None
        self._started = False
        self._stopping = False
        self._error: BaseException | None = None

    # -- introspection -------------------------------------------------------------

    @property
    def backend(self) -> DeviceBackend:
        """The device backend this session runs on."""
        return self._backend

    @property
    def ready(self) -> bool:
        """Whether the model is loaded and the inference thread is serving commands."""
        return self._started and self._error is None and not self._stopping

    # -- lifecycle -----------------------------------------------------------------

    async def start(self) -> None:
        """Load the model on the inference thread and wait until it is ready.

        Calling it again once the session is running is a no-op; a concurrent call waits
        for the same load.

        Raises:
            EngineError: if the session has already been stopped or has failed.
            Exception: whatever the model load raised.
        """
        if self._error is not None:
            raise EngineError("engine session has failed") from self._error
        if self._stopping:
            raise EngineError("engine session has been stopped")
        if self._start_future is not None:
            await asyncio.shield(self._start_future)
            return

        loop = asyncio.get_running_loop()
        self._loop = loop
        self._events = asyncio.Queue()
        self._start_future = loop.create_future()
        self._thread = threading.Thread(
            target=self._thread_main, name="talkover-engine", daemon=True
        )
        self._thread.start()
        await asyncio.shield(self._start_future)

    async def stop(self) -> None:
        """Stop the inference thread, close the upstream session and release the model.

        Safe to call more than once and safe to call before :meth:`start`. It never
        raises the terminal error; :meth:`events` is where that surfaces.
        """
        if self._stopping:
            return
        self._stopping = True
        thread = self._thread
        if thread is None:
            self._publish(_EVENTS_DONE)
            return
        self._commands.put(_Command("stop"))
        await asyncio.to_thread(thread.join, _STOP_JOIN_TIMEOUT_SEC)
        if thread.is_alive():  # pragma: no cover - only on a wedged model call
            LOGGER.error(
                "engine inference thread did not stop within %.0f s", _STOP_JOIN_TIMEOUT_SEC
            )
        self._thread = None

    # -- commands ------------------------------------------------------------------

    async def feed_pcm16(self, pcm16_16k_1s: bytes | np.ndarray) -> None:
        """Feed exactly one 1 s unit of 16 kHz mono pcm16 and run the model on it.

        Returns once the unit's events have been queued, so a caller that awaits this and
        then drains :meth:`events` sees them in order.

        Raises:
            TypeError, ValueError: if the unit is not exactly one 1 s pcm16 unit.
            EngineError: if the session is not running.
        """
        payload = _unit_to_pcm16_bytes(pcm16_16k_1s)
        await self._submit("feed", payload)

    async def flush_pending(self) -> None:
        """Run whatever partial unit upstream still holds, zero-padded to a full unit."""
        await self._submit("flush")

    async def interrupt_output(self) -> None:
        """End the current assistant output without pausing future model units.

        Commands are serialized, so this takes effect after the unit already in flight —
        under the real-time budget, within one unit.
        """
        await self._submit("interrupt")

    async def set_task_slate(self, text: str) -> None:
        """Replace the pinned task slate (DESIGN.md 5.5).

        Upstream ignores the slate unless ``sliding_window_mode`` installs a pinned
        context; that is logged, not raised.
        """
        await self._submit("slate", text)

    async def submit_text_turn(self, text: str) -> None:
        """Queue a text user turn, delivered to the model with the next audio unit.

        Upstream has no dedicated text-turn channel, so the text rides the runtime-event
        envelope (``DuplexLiveSession.feed_runtime_event``), which is prefilled inside
        ``<tool_response>`` markers together with the next unit's microphone audio.
        """
        await self._submit("text_turn", text)

    async def feed_tool_response(self, response: Mapping[str, Any]) -> None:
        """Answer the native tool call the model is blocked on.

        Calls upstream ``DuplexLiveSession.feed_tool_response``, which queues the payload
        for the next unit's prefill. The bridge (``talkover.realtime.brain_bridge``) is the
        only caller; see DESIGN.md 4.6 for the payload shape.

        Raises:
            EngineError: if upstream refuses the payload. The session survives it.
        """
        await self._submit("tool_response", dict(response))

    async def feed_worker_delivery(self, delivery: Mapping[str, Any]) -> None:
        """Queue one Brain delivery as an upstream ``worker_delivery`` runtime event.

        Raises:
            ValueError: if ``type`` is present and names something other than
                ``worker_delivery``.
            EngineError: if upstream refuses the payload. The session survives it.
        """
        payload = dict(delivery)
        kind = payload.setdefault("type", WORKER_DELIVERY_TYPE)
        if kind != WORKER_DELIVERY_TYPE:
            raise ValueError(
                f"a worker delivery must have type {WORKER_DELIVERY_TYPE!r}, not {kind!r}"
            )
        await self._submit("worker_delivery", payload)

    # -- events --------------------------------------------------------------------

    async def events(self) -> AsyncIterator[EngineStepEvent]:
        """Yield model units until the session stops.

        Raises:
            BaseException: the terminal failure from the inference thread, re-raised.
        """
        if self._events is None:
            raise EngineError("engine session has not been started")
        while True:
            item = await self._events.get()
            if isinstance(item, _Sentinel):
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    # -- conversion ----------------------------------------------------------------

    @staticmethod
    def from_upstream(step_event: Any, *, step_wall_time_sec: float = 0.0) -> EngineStepEvent:
        """Convert an upstream ``DuplexStepEvent`` into an :class:`EngineStepEvent`.

        Upstream's ``index`` becomes ``unit_index``; the waveform is normalized to a 1-D
        float32 numpy array (upstream hands out a torch tensor when the Talker ran), and
        an empty waveform becomes ``None`` so the mapping in DESIGN.md 5.3 has a single
        "no audio" case.
        """
        metrics = getattr(step_event, "metrics", None)
        return EngineStepEvent(
            unit_index=int(step_event.index),
            is_listen=bool(step_event.is_listen),
            text=str(step_event.text or ""),
            end_of_turn=bool(step_event.end_of_turn),
            interrupted=bool(getattr(step_event, "interrupted", False)),
            audio_waveform=_as_float32_waveform(getattr(step_event, "audio_waveform", None)),
            sample_rate=OUTPUT_SAMPLE_RATE,
            current_time=getattr(step_event, "current_time", None),
            unit_id=getattr(step_event, "unit_id", None),
            generation_id=int(getattr(step_event, "generation_id", 0) or 0),
            is_tool_call=bool(getattr(step_event, "is_tool_call", False)),
            tool_calls=tuple(getattr(step_event, "tool_calls", ()) or ()),
            tool_error=getattr(step_event, "tool_error", None),
            tool_response_expected=bool(getattr(step_event, "tool_response_expected", False)),
            metrics=dict(metrics) if isinstance(metrics, Mapping) else {},
            step_wall_time_sec=step_wall_time_sec,
        )

    @staticmethod
    def audio_event(chunk: TalkerChunk, *, unit_index: int) -> EngineStepEvent:
        """Wrap one Talker waveform chunk as an audio-only :class:`EngineStepEvent`.

        ``unit_index`` is the step the chunk was drained after, so the event stream stays
        monotonic; the unit the audio belongs to is :attr:`EngineStepEvent.unit_id`.
        ``end_of_turn`` stays ``False``: the Thinker's own event is what ends the turn.
        """
        return EngineStepEvent(
            unit_index=unit_index,
            is_listen=False,
            text="",
            end_of_turn=False,
            interrupted=False,
            audio_waveform=chunk.waveform,
            sample_rate=chunk.sample_rate,
            current_time=chunk.current_time,
            unit_id=chunk.unit_id,
            generation_id=chunk.generation_id,
            metrics=dict(chunk.metrics),
            is_audio_chunk=True,
        )

    # -- event-loop side plumbing --------------------------------------------------

    async def _submit(self, kind: str, payload: Any = None) -> None:
        """Queue one command for the inference thread and wait for it to finish."""
        if self._error is not None:
            raise EngineError(f"engine session has failed; cannot {kind}") from self._error
        if self._stopping:
            raise EngineError(f"engine session has been stopped; cannot {kind}")
        if not self._started or self._loop is None:
            raise EngineError(f"engine session has not been started; cannot {kind}")
        future: asyncio.Future[None] = self._loop.create_future()
        self._commands.put(_Command(kind, payload, future))
        await future

    def _publish(self, item: EngineStepEvent | BaseException | _Sentinel) -> None:
        """Hand one item to the event queue from whichever thread produced it."""
        events = self._events
        loop = self._loop
        if events is None or loop is None:
            return
        try:
            loop.call_soon_threadsafe(events.put_nowait, item)
        except RuntimeError:  # pragma: no cover - the loop closed under us
            LOGGER.debug("event loop closed while publishing an engine event")

    def _settle(self, future: asyncio.Future[None] | None, error: BaseException | None) -> None:
        """Complete a command's future from the inference thread."""
        loop = self._loop
        if future is None or loop is None:
            return

        def complete() -> None:
            if future.done():
                return
            if error is None:
                future.set_result(None)
            else:
                future.set_exception(error)

        try:
            loop.call_soon_threadsafe(complete)
        except RuntimeError:  # pragma: no cover - the loop closed under us
            LOGGER.debug("event loop closed while completing an engine command")

    # -- inference thread ----------------------------------------------------------

    def _thread_main(self) -> None:
        """The inference thread: load the model, then serve commands until stopped."""
        try:
            self._live = self._session_factory()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller verbatim
            self._error = exc
            self._settle(self._start_future, exc)
            self._publish(exc)
            return
        self._started = True
        self._settle(self._start_future, None)

        try:
            self._serve_commands()
        finally:
            self._close_live()
            self._publish(_EVENTS_DONE)

    def _serve_commands(self) -> None:
        """Pop commands until a stop arrives or one of them fails terminally."""
        while True:
            command = self._commands.get()
            if command.kind == "stop":
                self._settle(command.future, None)
                return
            try:
                self._run_command(command)
            except BaseException as exc:  # a model error of any kind is terminal
                if isinstance(exc, Exception) and command.kind in _REFUSABLE_COMMANDS:
                    LOGGER.warning("upstream refused the %s: %s", command.kind, exc)
                    refused = EngineError(f"upstream refused the {command.kind}: {exc}")
                    refused.__cause__ = exc
                    self._settle(command.future, refused)
                    continue
                LOGGER.exception("engine inference thread failed on %s", command.kind)
                self._error = exc
                self._settle(command.future, exc)
                self._publish(exc)
                self._drain_commands(exc)
                return
            self._settle(command.future, None)

    def _run_command(self, command: _Command) -> None:
        """Execute one command against the upstream session."""
        live = self._live
        if command.kind == "feed":
            self._emit(lambda: live.feed_pcm16(command.payload))
        elif command.kind == "flush":
            self._emit(live.flush_pending)
        elif command.kind == "interrupt":
            live.interrupt_output()
        elif command.kind == "slate":
            if not live.set_task_slate(command.payload):
                LOGGER.info("task slate ignored: no pinned context in this window mode")
        elif command.kind == "text_turn":
            live.feed_runtime_event({"type": "user_text", "content": command.payload})
        elif command.kind == "tool_response":
            live.feed_tool_response(command.payload)
        elif command.kind == "worker_delivery":
            live.feed_runtime_event(command.payload)
        else:  # pragma: no cover - only reachable through a coding error
            raise EngineError(f"unknown engine command {command.kind!r}")

    def _emit(self, call: Callable[[], Sequence[Any]]) -> None:
        """Run one stepping call, time it against the device, and publish its events."""
        started = time.perf_counter()
        step_events = call()
        # The device queue must be empty before the clock is read, or the measurement
        # records submission time rather than compute time (DESIGN.md 4.4, T1.8).
        self._backend.synchronize()
        elapsed = time.perf_counter() - started
        count = len(step_events) or 1
        for step_event in step_events:
            self._publish(self.from_upstream(step_event, step_wall_time_sec=elapsed / count))
        self._publish_talker_audio()

    def _talker(self) -> TalkerThread | None:
        """The Talker thread, when one runs off the inference thread (T1.5)."""
        worker = getattr(self._live, "speech_worker", None)
        return worker if isinstance(worker, TalkerThread) else None

    def _publish_talker_audio(self) -> None:
        """Publish whatever the Talker thread has finished since the last step.

        The Talker vocodes a unit while the Thinker is already working on the next one, so
        its waveform arrives after that unit's own event and is published on its own
        ``is_audio_chunk`` event. A failed unit is logged and leaves a gap in the audio: it
        does not end the session, because the Talker thread stays usable and one silent
        unit is a smaller loss than a dropped call.
        """
        talker = self._talker()
        if talker is None:
            return
        unit_index = int(getattr(self._live, "step_index", 0) or 0)
        for output in talker.drain_outputs():
            if isinstance(output, TalkerChunk):
                self._publish(self.audio_event(output, unit_index=unit_index))
            elif isinstance(output, TalkerFailed):
                LOGGER.error(
                    "talker failed on unit %d (generation %d): %s",
                    output.unit_id,
                    output.generation_id,
                    output.message,
                )
            elif isinstance(output, (TalkerDone, TalkerInterrupted)):
                LOGGER.debug("talker %s", output)

    def _drain_commands(self, error: BaseException) -> None:
        """Fail every queued command after a terminal error, so no caller hangs."""
        wrapped = EngineError("engine session has failed")
        wrapped.__cause__ = error
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._settle(command.future, None if command.kind == "stop" else wrapped)

    def _close_live(self) -> None:
        """Close the upstream session and drop the reference to the model."""
        live = self._live
        self._started = False
        if live is None:
            return
        # Whatever the Talker finished before the stop still belongs to the caller.
        try:
            self._publish_talker_audio()
        except Exception:  # pragma: no cover - draining must not mask the shutdown
            LOGGER.exception("draining the Talker thread during shutdown failed")
        self._live = None
        try:
            live.close()
        except Exception:  # closing must not mask the original failure
            LOGGER.exception("closing the upstream duplex session failed")
