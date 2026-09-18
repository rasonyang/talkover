"""Process composition: engine <-> realtime <-> Brain (DESIGN.md section 3).

This module is the one place where the three packages meet. It owns no protocol logic, no
inference and no business rules; it only builds the objects and hands each one the seams
the others expose:

===============================  =====================================================
Object                           Wired to
===============================  =====================================================
`BusinessProvider` / project      one per process, from `TalkoverConfig.brain` (T3.4)
`BrainBridge`                     one per Realtime session, holding that project and
                                  the engine it answers the Cerebellum through (4.6)
`WebSocketSession`                `on_engine_event` and `on_function_call_output` of
                                  that bridge (T2.4 / T3.7)
`AsrSideChannel`                  the units fed to the engine, fanned out to both
                                  `WebSocketSession.session.on_asr_event` and
                                  `BrainBridge.on_asr_event` (T1.6, DESIGN.md 4.3)
`create_app`                      the session factory above, the lifespan that starts
                                  and stops the engine, and `on_release`, which closes
                                  the bridge and resets the engine (T2.8 / T2.9)
===============================  =====================================================

**Lifecycle (T2.9).** `build_app` installs a FastAPI lifespan: the engine (and the ASR
side channel) start before the first request is served and stop on shutdown, so
`talkover serve` is nothing but "load the config, build these objects, run uvicorn". The
engine is *reset*, never stopped, when a session is released: the process serves the next
call and `GET /health` has to go back to `idle` (DESIGN.md 5.7).

Everything in this module stays import-light: it may not pull in torch, mcpmft or a model,
because `tests/protocol` drives the whole composition against a fake engine.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gander_runtime.contracts import new_id
from gander_runtime.coordination import ProjectRecord

from talkover.brain.provider import BusinessProject, BusinessProvider
from talkover.config import TalkoverConfig
from talkover.engine.memory import estimate_memory, format_estimate
from talkover.realtime.brain_bridge import BrainBridge
from talkover.realtime.server import CallLater, create_app, default_model
from talkover.realtime.session import WebSocketSession

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from talkover.engine.asr import AsrEvent, AsrStream
    from talkover.engine.protocol import EngineProtocol

__all__ = [
    "ASR_QUEUE_UNITS",
    "DEFAULT_PROJECT_LABEL",
    "DEFAULT_PROJECT_OWNER",
    "AsrConsumer",
    "AsrSideChannel",
    "TalkoverApp",
    "brain_ready",
    "build_app",
    "reset_engine",
    "startup_summary",
]

LOGGER = logging.getLogger(__name__)

#: One process serves one deployment, so the project record is a constant (DESIGN.md 5.7).
DEFAULT_PROJECT_OWNER = "talkover"
DEFAULT_PROJECT_LABEL = "customer service"

#: How many 1 s units the ASR side channel may queue before it starts dropping the oldest.
#:
#: ASR is a side channel: when transcription falls behind the call, losing transcripts is
#: the right failure, and blocking the audio path into the model is not (DESIGN.md 4.3).
ASR_QUEUE_UNITS = 32

#: What `AsrSideChannel` hands each finished ASR event to.
AsrConsumer = Callable[["AsrEvent"], Awaitable[None]]


# --------------------------------------------------------------------------------------
# ASR side channel
# --------------------------------------------------------------------------------------


class AsrSideChannel:
    """The ASR side channel: tap the units fed to the engine, fan the events out.

    The model consumes the microphone audio itself; ASR runs beside it and produces the
    transcript the client sees and the trusted text Brain reasons over (DESIGN.md 4.3).
    Both consumers take the same `AsrEvent`, so one stream feeds them both:

    - `RealtimeSession.on_asr_event` — approximate turn detection and the transcription
      events of the protocol (T2.6);
    - `BrainBridge.on_asr_event` — the trusted text a `task_start` is about (T3.7).

    Audio arrives through :meth:`feed`, which never blocks the caller: units go into a
    bounded queue and a worker task transcribes them, with the blocking model call moved
    off the event loop by `AsrStream.afeed`. Past :data:`ASR_QUEUE_UNITS` the oldest unit
    is dropped, counted in :attr:`dropped` and logged — never silently.

    The stream itself is built lazily by `factory`, on a worker thread inside
    :meth:`start`, so building the application loads no ASR model: `talkover serve
    --check-only` and every protocol test stop before that point.
    """

    def __init__(
        self,
        factory: Callable[[], AsrStream],
        *,
        queue_units: int = ASR_QUEUE_UNITS,
    ) -> None:
        self._factory = factory
        self._stream: AsrStream | None = None
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=max(1, queue_units))
        self._worker: asyncio.Task[None] | None = None
        self._consumers: tuple[AsrConsumer, ...] = ()
        #: Units dropped because the worker fell behind.
        self.dropped = 0

    @property
    def ready(self) -> bool:
        """`GET /health` reads this: whether a stream is loaded and its worker is running."""
        worker = self._worker
        return self._stream is not None and worker is not None and not worker.done()

    @property
    def stream(self) -> AsrStream | None:
        """The loaded stream, or `None` before :meth:`start`."""
        return self._stream

    # -- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Load the ASR backend off the event loop and start the worker. Idempotent."""
        if self._worker is not None:
            return
        self._stream = await asyncio.to_thread(self._factory)
        self._worker = asyncio.create_task(self._run(), name="talkover-asr")

    async def aclose(self) -> None:
        """Stop the worker and forget the stream."""
        self._consumers = ()
        worker, self._worker = self._worker, None
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._stream = None

    # -- wiring ------------------------------------------------------------

    def bind(self, *consumers: AsrConsumer) -> None:
        """Send every event to `consumers`, replacing whoever was bound before."""
        self._consumers = tuple(consumers)

    def unbind(self) -> None:
        """Stop sending events anywhere; the session they belonged to is gone."""
        self._consumers = ()

    @property
    def consumers(self) -> tuple[AsrConsumer, ...]:
        """Who the events currently go to."""
        return self._consumers

    # -- audio in ----------------------------------------------------------

    def feed(self, unit: Any) -> None:
        """Queue one unit of 16 kHz pcm16; never blocks and never raises."""
        self._offer("audio", unit)

    def request_flush(self) -> None:
        """Close an open utterance, e.g. when the client committed its input buffer."""
        self._offer("flush", None)

    async def reset(self) -> None:
        """Drop queued audio and the stream's own VAD state, between two calls."""
        self._drain()
        if self._worker is not None:
            self._offer("reset", None)

    def _offer(self, kind: str, payload: Any) -> None:
        if self._worker is None:
            return
        try:
            self._queue.put_nowait((kind, payload))
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
            self.dropped += 1
            LOGGER.warning(
                "the ASR side channel is behind; dropped the oldest unit (%d in total)",
                self.dropped,
            )
            with contextlib.suppress(asyncio.QueueFull):
                self._queue.put_nowait((kind, payload))

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    # -- the worker --------------------------------------------------------

    async def _run(self) -> None:
        while True:
            kind, payload = await self._queue.get()
            stream = self._stream
            if stream is None:  # pragma: no cover - only between aclose and cancellation
                continue
            try:
                events = await self._transcribe(stream, kind, payload)
            except Exception:  # a side channel never ends the call
                LOGGER.exception("the ASR side channel failed on a %r item", kind)
                continue
            for event in events:
                await self._dispatch(event)

    async def _transcribe(self, stream: AsrStream, kind: str, payload: Any) -> Sequence[AsrEvent]:
        if kind == "audio":
            return await stream.afeed(payload)
        if kind == "flush":
            return await asyncio.to_thread(stream.flush)
        stream.reset()
        return ()

    async def _dispatch(self, event: AsrEvent) -> None:
        for consumer in self._consumers:
            try:
                await consumer(event)
            except Exception:  # one consumer must not starve the other
                LOGGER.exception("an ASR consumer failed on %r", event)


class _TappedEngine:
    """`engine`, with every unit it is fed copied into the ASR side channel.

    The realtime layer feeds the model 1 s units of 16 kHz pcm16, which is exactly what
    ASR wants, so the side channel is a tap on that call rather than a second audio path.
    Everything else is delegation: the sessions, the mapper and `BrainBridge` see an
    ordinary `EngineProtocol`.
    """

    def __init__(self, engine: EngineProtocol, asr: AsrSideChannel) -> None:
        self._engine = engine
        self._asr = asr

    @property
    def engine(self) -> EngineProtocol:
        """The engine underneath the tap."""
        return self._engine

    @property
    def ready(self) -> bool:
        return self._engine.ready

    async def start(self) -> None:
        await self._engine.start()

    async def feed_pcm16(self, pcm16_16k_1s: Any) -> None:
        self._asr.feed(pcm16_16k_1s)
        await self._engine.feed_pcm16(pcm16_16k_1s)

    async def flush_pending(self) -> None:
        self._asr.request_flush()
        await self._engine.flush_pending()

    async def interrupt_output(self) -> None:
        await self._engine.interrupt_output()

    async def set_task_slate(self, text: str) -> None:
        await self._engine.set_task_slate(text)

    async def submit_text_turn(self, text: str) -> None:
        await self._engine.submit_text_turn(text)

    async def feed_tool_response(self, response: Any) -> None:
        await self._engine.feed_tool_response(response)

    async def feed_worker_delivery(self, delivery: Any) -> None:
        await self._engine.feed_worker_delivery(delivery)

    def events(self) -> AsyncIterator[Any]:
        return self._engine.events()

    async def stop(self) -> None:
        await self._engine.stop()


# --------------------------------------------------------------------------------------
# readiness and reset
# --------------------------------------------------------------------------------------


def brain_ready(
    config: TalkoverConfig, provider: BusinessProvider | None, *, needs_key: bool
) -> bool:
    """The `brain` field of `GET /health`: is there a provider, and can it reach an LLM?

    Deliberately cheap — `/health` is polled by a scheduler, so it may not call the LLM.
    A provider built here from `config.brain` can only answer once its credential is
    present, which is the one failure that is both common and invisible otherwise; a
    provider injected by the caller (`tests/brain`'s `FakeLLM`, a future in-process LLM)
    carries its own client and is taken at its word.
    """
    if provider is None:
        return False
    if not needs_key:
        return True
    return bool(config.brain.llm.api_key.strip())


async def reset_engine(engine: Any) -> None:
    """Return the engine to an idle state for the next call, without unloading the model.

    `SessionSlot` awaits this through `create_app(on_release=...)` when the single session
    slot frees. It deliberately does **not** call `engine.stop()`: the process goes on
    serving, and `GET /health` has to report `idle` rather than `not_ready` (DESIGN.md 5.7,
    `docs/protocol-profile.md` §10.1).

    Upstream `DuplexLiveSession` has no "start a new conversation" call short of rebuilding
    the session, which would reload the weights, so the reset is best effort and layered:
    an engine offering `reset()` is asked for one, and any other engine is told to end the
    output it still has in flight. A failure is logged, never raised — the release path
    must not take the process down.
    """
    reset = getattr(engine, "reset", None)
    try:
        if callable(reset):
            await reset()
        else:
            await engine.interrupt_output()
    except Exception:  # releasing a session must always finish
        LOGGER.exception("resetting the engine after the session was released failed")


# --------------------------------------------------------------------------------------
# startup report
# --------------------------------------------------------------------------------------


def startup_summary(config: TalkoverConfig) -> str:
    """The block `talkover serve` logs before it binds the port.

    Everything a support ticket needs and nothing that requires loading anything: where
    the service listens, which device and precision the model runs on, the four checkpoint
    paths, the resolved ASR backend, the Brain's LLM, and the static memory estimate from
    `talkover.engine.memory` (DESIGN.md 4.5).
    """
    engine = config.engine
    asr = config.asr
    brain = config.brain
    rows: list[tuple[str, str]] = [
        ("listen", config.server.listen),
        ("device", f"{engine.device} ({engine.dtype})"),
        ("media_mode", engine.media_mode),
        ("base_model", engine.base_model or "<unset>"),
        ("thinker", engine.thinker_checkpoint or "<unset>"),
        ("talker", engine.talker_checkpoint or "<unset>"),
        ("token2wav", engine.token2wav_dir or f"{engine.base_model}/assets/token2wav"),
        ("ref_audio", engine.ref_audio_path or "<checkpoint voice>"),
        ("asr", f"{asr.backend} {asr.model} on {asr.device} ({asr.compute_type})"),
        ("brain", f"{brain.provider} via {brain.llm.kind} {brain.llm.model}"),
        ("brain.llm.api_key", "set" if brain.llm.api_key.strip() else "MISSING"),
        ("model name", default_model(config)),
    ]
    width = max(len(name) for name, _ in rows)
    lines = ["talkover serve"] + [f"  {name.ljust(width)}  {value}" for name, value in rows]
    lines.append(format_estimate(estimate_memory(config)))
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# composition
# --------------------------------------------------------------------------------------


class _BrainState:
    """What the session factory writes and `GET /health` reads.

    `create_app` needs both before the application exists, so they live in this small
    mutable object rather than on :class:`TalkoverApp`, which is built afterwards.
    """

    __slots__ = ("bridge", "closed", "config", "needs_key", "provider", "runner")

    def __init__(
        self,
        config: TalkoverConfig,
        provider: BusinessProvider | None = None,
        *,
        needs_key: bool = True,
    ) -> None:
        #: The bridge of the session currently holding the single slot.
        self.bridge: BrainBridge | None = None
        #: The runner of that session, so the ASR tap knows who to feed.
        self.runner: WebSocketSession | None = None
        self.config = config
        self.provider = provider
        self.needs_key = needs_key
        self.closed = False

    @property
    def ready(self) -> bool:
        """`server._readiness` reads this attribute off the `brain` component."""
        if self.closed:
            return False
        return brain_ready(self.config, self.provider, needs_key=self.needs_key)


@dataclass(slots=True)
class TalkoverApp:
    """The composed application and the objects it owns."""

    app: FastAPI
    config: TalkoverConfig
    engine: EngineProtocol
    provider: BusinessProvider
    project: BusinessProject
    #: Whether this object built the provider and must therefore close it.
    owns_provider: bool = True
    #: The ASR side channel, when one was given (`None` in the protocol tests).
    asr: AsrSideChannel | None = None
    _state: _BrainState | None = None

    @property
    def bridge(self) -> BrainBridge | None:
        """The Brain bridge of the running session, or `None` when the slot is free."""
        return self._state.bridge if self._state is not None else None

    @property
    def runner(self) -> WebSocketSession | None:
        """The Realtime session holding the single slot, or `None` when it is free."""
        return self._state.runner if self._state is not None else None

    async def aclose(self) -> None:
        """Release the Brain side; the lifespan is what stops the engine (T2.9)."""
        state = self._state
        if state is None:  # pragma: no cover - defensive; build_app always sets it
            return
        bridge, state.bridge = state.bridge, None
        state.runner = None
        if bridge is not None:
            await bridge.aclose()
        state.closed = True
        if self.owns_provider:
            await self.provider.close()


async def build_app(
    config: TalkoverConfig,
    engine: EngineProtocol,
    *,
    provider: BusinessProvider | None = None,
    asr: object | None = None,
    call_later: CallLater | None = None,
    manage_engine: bool = True,
) -> TalkoverApp:
    """Compose the ASGI application around `engine` and a business Brain provider.

    `provider` is injectable so that a test drives the whole chain against `FakeLLM`;
    when it is omitted one is built from `config.brain` and closed by
    :meth:`TalkoverApp.aclose`. The engine is also the bridge's channel back to the model
    (`feed_tool_response` / `feed_worker_delivery`, DESIGN.md 4.6), so no separate
    Cerebellum object is passed around.

    `asr` is an :class:`AsrSideChannel` in a serving process: its units are tapped off
    `feed_pcm16` and its events reach both the Realtime session and the Brain bridge. Any
    other object is only read for its `ready` attribute by `GET /health`, which is what the
    protocol tests pass.

    `manage_engine` installs the lifespan that starts the engine before the first request
    and stops it (with the ASR channel and the Brain) on shutdown. Turning it off leaves
    the whole lifecycle to the caller; nothing else changes.

    Sessions are `WebSocketSession`, always: `create_app`'s `DefaultSession` is the
    placeholder for a bare server with no application around it (T2.3).
    """
    owns_provider = provider is None
    brain_provider = provider if provider is not None else BusinessProvider(config.brain)
    project = await brain_provider.open_project(
        ProjectRecord(
            project_id=new_id("project"),
            owner_id=DEFAULT_PROJECT_OWNER,
            label=DEFAULT_PROJECT_LABEL,
            provider_name=brain_provider.name,
        )
    )
    state = _BrainState(config, brain_provider, needs_key=owns_provider)
    side_channel = asr if isinstance(asr, AsrSideChannel) else None
    session_engine: Any = engine if side_channel is None else _TappedEngine(engine, side_channel)

    def session_factory(
        websocket: Any, session_config: TalkoverConfig, engine_for_session: Any, model: str
    ) -> WebSocketSession:
        """Build one Realtime session with its own Brain bridge."""
        bridge = BrainBridge(project, engine=engine_for_session)
        runner = WebSocketSession(
            websocket,
            session_config,
            engine_for_session,
            model,
            on_engine_event=bridge.on_engine_event,
            on_function_call_output=bridge.on_function_call_output,
        )
        bridge.bind(runner.session)
        state.bridge = bridge
        state.runner = runner
        if side_channel is not None:
            # One ASR stream, both consumers: turn detection and transcripts on the
            # protocol side, trusted text on the Brain side (DESIGN.md 4.3).
            side_channel.bind(runner.session.on_asr_event, bridge.on_asr_event)
        return runner

    async def on_release() -> None:
        """Give the process back to the next call when the single slot frees.

        Closes the Brain side of the session that has just ended, forgets its ASR state,
        and resets the engine — never stops it, so `/health` reports `idle` again (T2.8).
        """
        bridge, state.bridge = state.bridge, None
        state.runner = None
        if bridge is not None:
            await bridge.aclose()
        if side_channel is not None:
            side_channel.unbind()
            await side_channel.reset()
        await reset_engine(engine)

    composed: TalkoverApp | None = None

    @contextlib.asynccontextmanager
    async def lifespan(_: Any) -> AsyncIterator[None]:
        """Start the engine before the first connection, stop it on shutdown."""
        await engine.start()
        if side_channel is not None:
            try:
                await side_channel.start()
            except Exception:  # ASR is a side channel, not the call
                LOGGER.exception("the ASR backend could not be loaded; running without it")
        LOGGER.info("talkover is ready on %s", config.server.listen)
        try:
            yield
        finally:
            if composed is not None:
                await composed.aclose()
            if side_channel is not None:
                await side_channel.aclose()
            await engine.stop()

    app = create_app(
        config,
        session_engine,
        asr=asr,
        brain=state,
        session_factory=session_factory,
        on_release=on_release,
        call_later=call_later,
        lifespan=lifespan if manage_engine else None,
    )
    composed = TalkoverApp(
        app=app,
        config=config,
        engine=engine,
        provider=brain_provider,
        project=project,
        owns_provider=owns_provider,
        asr=side_channel,
        _state=state,
    )
    return composed
