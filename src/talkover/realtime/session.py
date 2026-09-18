"""The per-connection Realtime session state machine (client -> server half).

This module implements the left column of the event mapping table in DESIGN.md 5.3:
every client event is validated by `talkover.realtime.events`, applied to the session
state, turned into the engine call the table prescribes, and acknowledged with the
server event `docs/protocol-profile.md` section 9 documents.

Layering:

- `RealtimeSession` is transport-free. It writes server events to an `EventSink` and
  drives the engine through `talkover.engine.protocol.EngineProtocol`, so the protocol
  tests run it with a list sink and a fake engine, without a WebSocket and without a GPU.
- `WebSocketSession` is the thin adapter with the signature
  `session_factory(websocket, config, engine, model)` that `talkover.realtime.server`
  expects: it wraps the WebSocket in a `WebSocketSink`, runs the receive loop and pumps
  engine events.

Seams left for the tasks that build on this one:

- `RealtimeSession.sink` is the single outbound path. T2.5 (`mapping.py`) and T2.6
  (`turn_detection.py`) emit through it and never touch the WebSocket.
- `RealtimeSession.mapper` is T2.5's `ResponseMapper`: `handle_engine_event` hands it every
  unit and it owns the whole `response.*` lifecycle. `RealtimeSession.on_engine_event` stays
  the extra consumer hook `pump_engine_events` calls after the mapper.
- `RealtimeSession.active_response_id`, `pending_response_metadata`, `client_cancel_requested`
  and `register_item` are the response/conversation bookkeeping the mapper writes to; this
  module only reads them for the state-dependent rejections in section 8.3 of the profile.
- `RealtimeSession.on_function_call_output` is the T3.7 seam: a
  `conversation.item.create` carrying a `function_call_output` is acknowledged here and
  handed to that callback, which T3.7 implements as the Brain's `TaskInteractionReply`
  (DESIGN.md 5.3, 6.3). The other half of that seam is `emit_function_call`, which T3.7
  calls when the Brain hands over `transfer_to_human`. Nothing about the Brain is
  implemented in this module.

- `RealtimeSession.turn_detector` is T2.6's `TurnDetector`: `on_asr_event` feeds it the ASR
  VAD signals, `detect_turn` feeds it every engine unit so a barge-in can open a speech span
  retroactively, and both emit through the sink. The detector decides nothing about
  responses; it only synthesizes `input_audio_buffer.speech_started` / `speech_stopped`.

- `WebSocketSession` outlives its WebSocket (T2.8). A disconnect only detaches the socket:
  the `RealtimeSession` (ids, effective session, items, mapper and turn-detector state) and
  the engine pump stay alive, and `talkover.realtime.server.SessionSlot` keeps the session
  reserved for `realtime.trailing_silence_sec`. A reconnection presenting the same
  `session_id` lands in `attach`, which re-announces the session and replays what the
  engine produced meanwhile; `aclose` is what finally ends it.

What this module deliberately does not do: build the `response.*` events itself
(`mapping.py` does) or own the engine lifecycle (`start` / `stop` belong to the application
wiring, T2.9).
"""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import logging
from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

from starlette.websockets import WebSocketDisconnect

from talkover.config import TalkoverConfig
from talkover.engine.asr import Transcript
from talkover.engine.protocol import EngineError, EngineStepEvent
from talkover.realtime import events as ev
from talkover.realtime.audio import InputAudioStream
from talkover.realtime.mapping import ResponseMapper
from talkover.realtime.turn_detection import TurnDetector

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import WebSocket

    from talkover.engine.asr import AsrEvent
    from talkover.engine.protocol import EngineProtocol
else:  # pragma: no cover - runtime duck typing
    EngineProtocol = Any
    WebSocket = Any

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_TURN_DETECTION",
    "DEFAULT_VOICE",
    "DETACHED_EVENT_LIMIT",
    "FATAL_ERROR_CODES",
    "EventSink",
    "RealtimeSession",
    "ReconnectSink",
    "WebSocketSession",
    "WebSocketSink",
]

LOGGER = logging.getLogger(__name__)

#: Echoed as `session.model` when the client sends neither `?model=` nor `session.model`.
#: `talkover.realtime.server.DEFAULT_MODEL` holds the same value for the query parameter;
#: the two are kept equal by a test rather than by an import, so that `server` may import
#: `session` (T2.9) without a cycle.
DEFAULT_MODEL = "talkover"

#: Echoed as `audio.output.voice` until the client sets one. The Talker voice is fixed by
#: the checkpoint, so the field is echo-only (profile section 3).
DEFAULT_VOICE = "gander"

#: Effective `audio.input.turn_detection` until the client sets or clears it. The values
#: are the GA defaults; Talkover's turn detection is approximate (profile section 7).
DEFAULT_TURN_DETECTION: Mapping[str, Any] = {
    "type": "server_vad",
    "threshold": 0.5,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 200,
}

#: Error codes that end the session instead of being reported and survived (profile 8.1).
FATAL_ERROR_CODES = frozenset({"engine_busy", "engine_error"})

#: Nullable session fields: an explicit `null` clears them rather than being ignored.
_NULLABLE_AUDIO_INPUT = ("transcription", "noise_reduction", "turn_detection")

#: Server events `ReconnectSink` holds while no WebSocket is attached (T2.8). One model
#: unit per second produces a handful of events, so the default `trailing_silence_sec`
#: window of 8 s fits many times over; the bound only caps the memory a client that never
#: comes back can cost.
DETACHED_EVENT_LIMIT = 256

#: Errors a `send` on a dying socket raises; they detach the sink instead of ending the
#: session, so the event stays queued for the reconnection.
_SEND_FAILURES = (RuntimeError, OSError, WebSocketDisconnect)


class EventSink(Protocol):
    """Where a session writes the server events it produces."""

    async def send(self, event: Mapping[str, Any]) -> None: ...


class WebSocketSink:
    """An `EventSink` writing one JSON text frame per event."""

    def __init__(self, websocket: WebSocket) -> None:
        self.websocket = websocket

    async def send(self, event: Mapping[str, Any]) -> None:
        await self.websocket.send_json(dict(event))


class ReconnectSink:
    """An `EventSink` that survives the disconnect of its WebSocket (T2.8).

    Every event goes through the same FIFO queue, so ordering holds whether it is written
    straight away or buffered:

    - **attached** — the queue is drained to the socket as events arrive.
    - **detached** — the client is gone and its `realtime.trailing_silence_sec` reconnect
      window is open; the engine keeps producing and the events pile up here. At
      `limit` events the *oldest* is dropped, counted in `dropped` and logged at
      `WARNING`: a resumed call is better served by the end of what it missed than by its
      beginning, and nothing is ever discarded silently.

    `attach` takes the re-announcement in `front` and puts it before the backlog, so a
    resumed client always reads `session.created` first. A failed write detaches the sink
    and leaves the event queued rather than raising: the socket may die between the read
    loop noticing and the engine pump's next event.
    """

    def __init__(
        self, websocket: WebSocket | None = None, *, limit: int = DETACHED_EVENT_LIMIT
    ) -> None:
        self._websocket = websocket
        self._buffer: deque[dict[str, Any]] = deque(maxlen=max(1, limit))
        self._flushing = False
        #: Events lost to the bound of the buffer over the life of this session.
        self.dropped = 0

    @property
    def attached(self) -> bool:
        """Whether a WebSocket is currently receiving this session's events."""
        return self._websocket is not None

    @property
    def buffered(self) -> int:
        """How many events are waiting for a socket to write them to."""
        return len(self._buffer)

    async def send(self, event: Mapping[str, Any]) -> None:
        """Queue one event, then write everything queued in order."""
        self._store(dict(event))
        await self.flush()

    def attach(self, websocket: WebSocket, *, front: Sequence[Mapping[str, Any]] = ()) -> None:
        """Point the sink at `websocket`, putting `front` ahead of the backlog.

        Synchronous on purpose: the socket and the re-announcement are installed without
        an await in between, so a concurrent `send` cannot slip past them.
        """
        for event in reversed(list(front)):
            self._note_drop_if_full()
            self._buffer.appendleft(dict(event))
        self._websocket = websocket

    def detach(self) -> None:
        """Forget the socket; events are buffered until the next `attach`."""
        self._websocket = None

    def clear(self) -> None:
        """Drop the backlog: the session is over and nobody will ever read it."""
        self._buffer.clear()

    async def flush(self) -> None:
        """Write the buffered events to the attached socket, oldest first."""
        if self._flushing:
            return
        self._flushing = True
        try:
            while self._buffer and self._websocket is not None:
                event = self._buffer[0]
                try:
                    await self._websocket.send_json(event)
                except _SEND_FAILURES as exc:
                    LOGGER.debug("Realtime sink detached by a failed send: %s", exc)
                    self.detach()
                    return
                self._buffer.popleft()
        finally:
            self._flushing = False

    def _store(self, event: dict[str, Any]) -> None:
        self._note_drop_if_full()
        self._buffer.append(event)

    def _note_drop_if_full(self) -> None:
        if len(self._buffer) != self._buffer.maxlen:
            return
        self.dropped += 1
        LOGGER.warning(
            "Reconnect buffer full at %d events; dropping the oldest server event "
            "(%d dropped in this session).",
            self._buffer.maxlen,
            self.dropped,
        )


# ---------------------------------------------------------------------------
# the state machine
# ---------------------------------------------------------------------------


class RealtimeSession:
    """One Realtime conversation: session state, input audio path and acknowledgements.

    The session is created for an already-authenticated connection that holds the single
    session slot. `open()` sends `session.created` and `conversation.created`; every
    client frame then goes through `handle_raw`, which answers a rejected frame with the
    documented `error` event and keeps the session open. A fatal code
    (`FATAL_ERROR_CODES`) is re-raised instead, so the transport can close with 1011.
    """

    def __init__(
        self,
        config: TalkoverConfig,
        engine: EngineProtocol,
        *,
        sink: EventSink,
        model: str = DEFAULT_MODEL,
        session_id: str | None = None,
        conversation_id: str | None = None,
        on_engine_event: Callable[[EngineStepEvent], Awaitable[None]] | None = None,
        on_function_call_output: Callable[[ev.FunctionCallOutputItem], Awaitable[None]]
        | None = None,
    ) -> None:
        self.config = config
        self.engine = engine
        self.sink = sink
        self.session_id = session_id or ev.new_session_id()
        self.conversation_id = conversation_id or ev.new_conversation_id()

        #: Engine-event consumer hook (T2.5 / T2.6).
        self.on_engine_event = on_engine_event
        #: Brain seam (T3.7); see the module docstring.
        self.on_function_call_output = on_function_call_output

        #: The effective session object, echoed by `session.created` / `session.updated`.
        self.session: dict[str, Any] = _default_session(self.session_id, model)
        #: Conversation items known to this session, in creation order.
        self.items: dict[str, dict[str, Any]] = {}
        self.item_order: list[str] = []

        #: Set by the mapper while a response is streaming; read here for the rejections.
        self.active_response_id: str | None = None
        #: `response.create.response.metadata` awaiting the response the mapper will open.
        self.pending_response_metadata: Mapping[str, str] | None = None
        #: A `response.cancel` is pending, so the next `interrupted` unit is client-cancelled.
        self.client_cancel_requested = False
        #: The item id of the last committed input-audio item (ASR transcripts target it).
        self.input_item_id: str | None = None
        #: Approximate turn detection (T2.6); fed by `on_asr_event` and the engine pump.
        self.turn_detector = TurnDetector(self.turn_detection)
        #: The `response.*` mapping (T2.5); fed by `handle_engine_event`.
        self.mapper = ResponseMapper(self)

        self._answered_call_ids: set[str] = set()
        self._input = InputAudioStream(self.input_audio_format)
        self._appended_samples = 0
        self._committed_ms = 0.0
        self._event_id: str | None = None

    # -- views -------------------------------------------------------------

    @property
    def model(self) -> str:
        """The echoed model name."""
        return str(self.session["model"])

    @property
    def input_audio_format(self) -> str:
        """Negotiated input format as `talkover.realtime.audio` names it."""
        return ev.audio_format_name(self.session["audio"]["input"]["format"])

    @property
    def output_audio_format(self) -> str:
        """Negotiated output format as `talkover.realtime.audio` names it (T2.5 reads it)."""
        return ev.audio_format_name(self.session["audio"]["output"]["format"])

    @property
    def turn_detection(self) -> Mapping[str, Any] | None:
        """Effective turn detection, or `None` when the client disabled it (T2.6)."""
        value = self.session["audio"]["input"]["turn_detection"]
        return value if isinstance(value, Mapping) else None

    @property
    def buffered_ms(self) -> float:
        """Client audio appended since the last commit or clear, in milliseconds."""
        return 1000.0 * self._appended_samples / self._input.source_rate

    @property
    def committed_ms(self) -> float:
        """Client audio committed since the session started, in milliseconds (T2.6)."""
        return self._committed_ms

    @property
    def last_item_id(self) -> str | None:
        """The most recently created item, used as `previous_item_id`."""
        return self.item_order[-1] if self.item_order else None

    def session_view(self) -> dict[str, Any]:
        """A defensive copy of the effective session object."""
        return _deep_copy(self.session)

    def register_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Record an item in the conversation and return the stored copy.

        T2.5 calls this for assistant output items so that `conversation.item.truncate`
        and `.delete` can reference them.
        """
        stored = _deep_copy(dict(item))
        item_id = str(stored["id"])
        if item_id not in self.items:
            self.item_order.append(item_id)
        self.items[item_id] = stored
        return stored

    # -- lifecycle ---------------------------------------------------------

    async def open(self) -> None:
        """Send the two events that precede any client frame (profile section 1)."""
        await self.emit(ev.SessionCreated(session=self.session_view()))
        await self.emit(ev.ConversationCreated(conversation=self.conversation_view()))

    def conversation_view(self) -> dict[str, Any]:
        """The `conversation` object of `conversation.created`."""
        return {"id": self.conversation_id, "object": "realtime.conversation"}

    def reopen_events(self) -> list[dict[str, Any]]:
        """The events that re-announce this session to a reconnected client (T2.8).

        A resumed connection opens with the same pair as a fresh one — `session.created`
        then `conversation.created` (profile section 1) — but carrying the ids and the
        effective session the dropped connection left behind, which is what tells the
        client the resume worked. `session.updated` follows only when a `session.update`
        had changed something, so a client can see its own configuration survived the gap
        without diffing the created event against the defaults.
        """
        produced: list[ev.ServerEvent] = [
            ev.SessionCreated(session=self.session_view()),
            ev.ConversationCreated(conversation=self.conversation_view()),
        ]
        if self.session != _default_session(self.session_id, self.model):
            produced.append(ev.SessionUpdated(session=self.session_view()))
        return [event.to_wire() for event in produced]

    async def emit(self, event: ev.ServerEvent) -> None:
        """Write one server event to the sink."""
        await self.sink.send(event.to_wire())

    async def pump_engine_events(self) -> None:
        """Hand every `EngineStepEvent` to `handle_engine_event` until the engine stops.

        Returns immediately when the engine exposes no `events()`. A terminal engine
        failure is re-raised as a fatal `engine_error` `ProtocolError`.
        """
        events = getattr(self.engine, "events", None)
        if events is None:
            return
        try:
            async for step in events():
                await self.handle_engine_event(step)
        except EngineError as exc:
            raise ev.ProtocolError("engine_error", f"The inference engine failed: {exc}") from exc

    async def handle_engine_event(self, step: EngineStepEvent) -> None:
        """Turn one model unit into server events.

        Turn detection runs first, so a retroactive `speech_started` precedes the
        `response.done` of the barge-in that revealed it; then the mapper produces the
        `response.*` lifecycle (DESIGN.md 5.3); then the optional `on_engine_event` hook
        sees the raw unit.
        """
        await self.detect_turn(step)
        await self.mapper.on_engine_event(step)
        handler = self.on_engine_event
        if handler is not None:
            await handler(step)

    async def emit_function_call(self, call_id: str, name: str, arguments: str) -> str:
        """Emit the `function_call` pair for one Brain tool call; return its item id.

        The T3.7 seam for `transfer_to_human`: one complete
        `response.function_call_arguments.delta` followed by `.done`, attached to the
        streaming response or, when the model is not speaking, to a response of its own.
        """
        return await self.mapper.emit_function_call(call_id, name, arguments)

    # -- turn detection (T2.6) ---------------------------------------------

    async def on_asr_event(self, event: AsrEvent) -> None:
        """Feed one ASR VAD / transcript signal to turn detection and emit what it makes.

        A finished segment also reaches the mapper, which reports it as
        `conversation.item.input_audio_transcription.completed` on the committed input
        item, after the `speech_stopped` that closed the span.
        """
        for produced in self.turn_detector.on_asr_event(event):
            await self.emit(produced)
        if isinstance(event, Transcript):
            await self.mapper.on_asr_transcript(event)

    async def detect_turn(self, step: EngineStepEvent) -> None:
        """Let a barge-in (`interrupted: true`) open a speech span retroactively."""
        for produced in self.turn_detector.on_engine_event(step):
            await self.emit(produced)

    # -- client frames -----------------------------------------------------

    async def handle_raw(self, raw: str | bytes | Mapping[str, Any]) -> None:
        """Validate and apply one client frame, answering a rejection with `error`."""
        try:
            if isinstance(raw, (bytes, bytearray)):
                raise ev.ProtocolError(
                    "invalid_event",
                    "Binary frames are not accepted; send one JSON object per text frame.",
                )
            await self.handle(ev.parse_client_event(raw))
        except ev.ProtocolError as exc:
            if exc.code in FATAL_ERROR_CODES:
                raise
            await self.sink.send(ev.error_event(exc))

    async def handle(self, event: ev.ClientEvent) -> None:
        """Apply one already-validated client event."""
        self._event_id = event.event_id
        try:
            if isinstance(event, ev.SessionUpdate):
                await self._on_session_update(event)
            elif isinstance(event, ev.InputAudioBufferAppend):
                await self._on_append(event)
            elif isinstance(event, ev.InputAudioBufferCommit):
                await self._on_commit()
            elif isinstance(event, ev.InputAudioBufferClear):
                await self._on_clear()
            elif isinstance(event, ev.ResponseCreate):
                await self._on_response_create(event)
            elif isinstance(event, ev.ResponseCancel):
                await self._on_response_cancel(event)
            elif isinstance(event, ev.ConversationItemCreate):
                await self._on_item_create(event)
            elif isinstance(event, ev.ConversationItemTruncate):
                await self._on_item_truncate(event)
            elif isinstance(event, ev.ConversationItemDelete):
                await self._on_item_delete(event)
            else:  # pragma: no cover - parse_client_event returns nothing else
                raise self.invalid_value(f"Unhandled event type {event.TYPE!r}.", "type")
        finally:
            self._event_id = None

    # -- handlers ----------------------------------------------------------

    async def _on_session_update(self, event: ev.SessionUpdate) -> None:
        raw = event.session.raw
        previous_format = self.input_audio_format
        for name in ("model", "output_modalities", "tools", "tool_choice", "max_output_tokens"):
            if name in raw:
                self.session[name] = _deep_copy(raw[name])
        for name in ("truncation", "tracing", "prompt"):
            if name in raw:
                self.session[name] = raw[name]
        if "audio" in raw:
            _merge_audio(self.session["audio"], raw["audio"])
        if "instructions" in raw:
            self.session["instructions"] = ev.truncate_instructions(
                str(raw["instructions"]), self.config.realtime.memory_slate_max_tokens
            )
            await self._engine_call("set_task_slate", self.session["instructions"])
        if self.input_audio_format != previous_format:
            # A new input format restarts the decoder, the resampler and the framer, so
            # the incomplete unit buffered in the old format is dropped.
            self._reset_input()
        self.turn_detector.configure(self.turn_detection)
        await self.emit(ev.SessionUpdated(session=self.session_view()))

    async def _on_append(self, event: ev.InputAudioBufferAppend) -> None:
        try:
            samples = self._input.decode(event.audio)
            units = self._input.push(samples)
        except (ValueError, binascii.Error) as exc:
            raise self.invalid_value(
                f"Could not decode the base64 audio payload as {self.input_audio_format}: {exc}",
                "audio",
            ) from exc
        self._appended_samples += int(samples.size)
        for unit in units:
            await self._engine_call("feed_pcm16", unit)

    async def _on_commit(self) -> None:
        if self._appended_samples == 0:
            raise ev.ProtocolError(
                "invalid_event",
                "The input audio buffer is empty; there is nothing to commit.",
                None,
                self._event_id,
            )
        await self._flush_input()
        await self._engine_call("flush_pending")
        self._committed_ms += self.buffered_ms
        self._appended_samples = 0
        previous_item_id = self.last_item_id
        item = self.register_item(
            {
                # The speech events of T2.6 name the item this audio ends up in, so the
                # detector's id wins when a speech span preceded the commit.
                "id": self.turn_detector.take_item_id() or ev.new_item_id(),
                "object": "realtime.item",
                "type": "message",
                "role": "user",
                "status": "completed",
                "content": [{"type": "input_audio", "transcript": None}],
            }
        )
        self.input_item_id = str(item["id"])
        await self.emit(
            ev.InputAudioBufferCommitted(
                item_id=self.input_item_id, previous_item_id=previous_item_id
            )
        )
        await self.emit(ev.ConversationItemCreated(item=item, previous_item_id=previous_item_id))

    async def _on_clear(self) -> None:
        self._input.clear()
        self._appended_samples = 0
        await self.emit(ev.InputAudioBufferCleared())

    async def _on_response_create(self, event: ev.ResponseCreate) -> None:
        if self.active_response_id is not None:
            raise ev.ProtocolError(
                "invalid_event",
                "A response is already in progress.",
                None,
                self._event_id,
            )
        self.pending_response_metadata = event.response.metadata if event.response else None
        await self._flush_input()
        await self._engine_call("flush_pending")

    async def _on_response_cancel(self, event: ev.ResponseCancel) -> None:
        active = self.active_response_id
        if active is None:
            raise ev.ProtocolError(
                "invalid_event",
                "There is no response in progress to cancel.",
                None,
                self._event_id,
            )
        if event.response_id is not None and event.response_id != active:
            raise self.invalid_value(
                f"response_id {event.response_id!r} is not the response in progress.",
                "response_id",
            )
        # The engine reports the cancellation as an `interrupted` unit like any barge-in;
        # this flag is what tells the mapper to say `client_cancelled` instead.
        self.client_cancel_requested = True
        await self._engine_call("interrupt_output")

    async def _on_item_create(self, event: ev.ConversationItemCreate) -> None:
        item = event.item
        if item.id is not None and item.id in self.items:
            raise self.invalid_value(
                f"Item id {item.id!r} already exists in this session.", "item.id"
            )
        previous_item_id = event.previous_item_id or self.last_item_id
        if isinstance(item, ev.FunctionCallOutputItem):
            if item.call_id in self._answered_call_ids:
                raise self.invalid_value(
                    f"call_id {item.call_id!r} has already been answered in this session.",
                    "item.call_id",
                )
            self._answered_call_ids.add(item.call_id)
            stored = self.register_item(
                {
                    "id": item.id or ev.new_item_id(),
                    "object": "realtime.item",
                    "type": "function_call_output",
                    "call_id": item.call_id,
                    "output": item.output,
                    "status": "completed",
                }
            )
            await self.emit(
                ev.ConversationItemCreated(item=stored, previous_item_id=previous_item_id)
            )
            if self.on_function_call_output is not None:
                await self.on_function_call_output(item)
            return
        stored = self.register_item(
            {
                "id": item.id or ev.new_item_id(),
                "object": "realtime.item",
                "type": "message",
                "role": item.role,
                "status": "completed",
                "content": [{"type": "input_text", "text": item.text}],
            }
        )
        await self.emit(ev.ConversationItemCreated(item=stored, previous_item_id=previous_item_id))
        await self._engine_call("submit_text_turn", item.text)

    async def _on_item_truncate(self, event: ev.ConversationItemTruncate) -> None:
        self._require_item(event.item_id)
        await self.emit(
            ev.ConversationItemTruncated(
                item_id=event.item_id,
                content_index=event.content_index,
                audio_end_ms=event.audio_end_ms,
            )
        )

    async def _on_item_delete(self, event: ev.ConversationItemDelete) -> None:
        self._require_item(event.item_id)
        self.items.pop(event.item_id, None)
        with contextlib.suppress(ValueError):
            self.item_order.remove(event.item_id)
        if self.input_item_id == event.item_id:
            self.input_item_id = None
        await self.emit(ev.ConversationItemDeleted(item_id=event.item_id))

    # -- helpers -----------------------------------------------------------

    def invalid_value(self, message: str, param: str) -> ev.ProtocolError:
        """Build an `invalid_value` rejection for the client event being handled.

        Public because the Brain bridge (T3.7) raises one from
        `on_function_call_output` for a `call_id` no pending Brain interaction owns, and
        it must carry the same `event_id` as a rejection this module raises itself.
        """
        return ev.ProtocolError("invalid_value", message, param, self._event_id)

    def _require_item(self, item_id: str) -> dict[str, Any]:
        if item_id not in self.items:
            raise self.invalid_value(f"Unknown item id {item_id!r}.", "item_id")
        return self.items[item_id]

    def _reset_input(self) -> None:
        self._input = InputAudioStream(self.input_audio_format)
        self._appended_samples = 0

    async def _flush_input(self) -> None:
        """Send the buffered incomplete unit, zero-padded, to the engine.

        `EngineProtocol.flush_pending` only runs what is already inside the engine, and
        the partial unit lives in this layer's framer, so both `input_audio_buffer.commit`
        and `response.create` push it first.
        """
        tail = self._input.flush(pad=True)
        if tail is not None:
            await self._engine_call("feed_pcm16", tail)

    async def _engine_call(self, name: str, *args: Any) -> None:
        """Call one `EngineProtocol` method, turning a failure into a fatal error."""
        method = getattr(self.engine, name)
        try:
            await method(*args)
        except EngineError as exc:
            raise ev.ProtocolError(
                "engine_error",
                f"The inference engine failed during {name}: {exc}",
                None,
                self._event_id,
            ) from exc


# ---------------------------------------------------------------------------
# effective session object
# ---------------------------------------------------------------------------


def _default_session(session_id: str, model: str) -> dict[str, Any]:
    """The effective session before any `session.update` (profile section 3)."""
    return {
        "id": session_id,
        "object": "realtime.session",
        "type": "realtime",
        "model": model,
        "instructions": "",
        "output_modalities": ["audio"],
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": None,
                "noise_reduction": None,
                "turn_detection": dict(DEFAULT_TURN_DETECTION),
            },
            "output": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "voice": DEFAULT_VOICE,
                "speed": 1.0,
            },
        },
        "tools": [],
        "tool_choice": "auto",
        "max_output_tokens": "inf",
        "truncation": "auto",
        "tracing": None,
        "prompt": None,
    }


def _merge_audio(current: dict[str, Any], update: Mapping[str, Any]) -> None:
    """Merge `session.audio` field-wise; `null` clears the nullable input fields.

    `turn_detection` is the one exception to the field-wise merge: a change of `type`
    replaces the object instead of merging into it. The two modes have disjoint fields
    (`threshold` / `prefix_padding_ms` / `silence_duration_ms` for `server_vad`,
    `eagerness` for `semantic_vad`), and profile section 3 rejects a request carrying the
    fields of the other mode, so merging would echo an effective session no client could
    have sent. An update that keeps the type, or omits it, still merges.
    """
    for side in ("input", "output"):
        block = update.get(side)
        if not isinstance(block, Mapping):
            continue
        target = current[side]
        for key, value in block.items():
            if key in _NULLABLE_AUDIO_INPUT and value is None:
                target[key] = None
            elif _replaces_turn_detection(key, value, target.get(key)):
                target[key] = _deep_copy(dict(value))
            elif isinstance(value, Mapping) and isinstance(target.get(key), Mapping):
                target[key] = {**target[key], **_deep_copy(dict(value))}
            else:
                target[key] = _deep_copy(value)


def _replaces_turn_detection(key: str, value: Any, current: Any) -> bool:
    """Whether this `turn_detection` update switches mode and must replace the object."""
    if key != "turn_detection" or not isinstance(value, Mapping):
        return False
    if not isinstance(current, Mapping):
        return False
    updated_type = value.get("type")
    return updated_type is not None and updated_type != current.get("type")


def _deep_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _deep_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
        return [_deep_copy(item) for item in value]
    return value


# ---------------------------------------------------------------------------
# WebSocket adapter
# ---------------------------------------------------------------------------


class WebSocketSession:
    """Drive one `RealtimeSession` over a WebSocket, across reconnections (T2.8).

    Matches the `session_factory(websocket, config, engine, model)` signature
    `talkover.realtime.server.create_app` calls, so the server stays free of protocol
    logic, and implements the resumable-session seam that `SessionSlot` drives:

    | Method | When the server calls it |
    | --- | --- |
    | `run()` | the first connection; returns when the client disconnects |
    | `attach(websocket)` | a reconnection that presented this `session_id` |
    | `aclose()` | the reconnect window expired, or the session failed |

    A disconnect only detaches the sink: the state machine, the mapper and the engine pump
    keep running, so the units the model produces during the window are buffered rather
    than lost (`ReconnectSink`) and the conversation is intact when the client returns. A
    fatal `ProtocolError` is raised for the server to report and close with 1011; the
    session is then discarded instead of being kept for a resume.
    """

    def __init__(
        self,
        websocket: WebSocket,
        config: TalkoverConfig,
        engine: EngineProtocol,
        model: str = DEFAULT_MODEL,
        *,
        on_engine_event: Callable[[EngineStepEvent], Awaitable[None]] | None = None,
        on_function_call_output: Callable[[ev.FunctionCallOutputItem], Awaitable[None]]
        | None = None,
    ) -> None:
        self.websocket = websocket
        self.sink = ReconnectSink(websocket)
        self.session = RealtimeSession(
            config,
            engine,
            sink=self.sink,
            model=model,
            on_engine_event=on_engine_event,
            on_function_call_output=on_function_call_output,
        )
        self._pump: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def session_id(self) -> str:
        """The `sess_…` id this connection was given (T2.8 reconnects on it)."""
        return self.session.session_id

    @property
    def attached(self) -> bool:
        """Whether a client WebSocket is currently reading this session."""
        return self.sink.attached

    @property
    def resumable(self) -> bool:
        """Whether a reconnection may still resume this session.

        `False` once `aclose` has run, or once the engine's event stream has ended — by a
        terminal failure or by a `stop`. A session whose engine is gone has nothing to
        resume, so the slot releases it at once instead of holding the process for the
        rest of the window.
        """
        if self._closed:
            return False
        pump = self._pump
        return pump is None or not pump.done()

    async def run(self) -> None:
        """Serve the first connection of this session."""
        await self.session.open()
        self._pump = asyncio.create_task(self.session.pump_engine_events())
        await self._serve()

    async def attach(self, websocket: WebSocket) -> None:
        """Resume this session on a reconnected `websocket` (T2.8).

        The re-announcement goes out first, then whatever the engine produced while the
        session was detached, then the new connection is served like any other.
        """
        self.websocket = websocket
        self.sink.attach(websocket, front=self.session.reopen_events())
        await self.sink.flush()
        await self._serve()

    async def aclose(self) -> None:
        """End the session for good: stop the engine pump and drop the backlog."""
        self._closed = True
        self.sink.detach()
        self.sink.clear()
        pump, self._pump = self._pump, None
        if pump is None:
            return
        if not pump.done():
            pump.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pump
        elif not pump.cancelled():
            # Retrieve a failure nobody was attached to see, so asyncio does not log it.
            pump.exception()

    async def _serve(self) -> None:
        """Read client frames until the socket or the engine ends the connection."""
        reader = asyncio.create_task(self._read_loop())
        waiting = {reader} if self._pump is None else {reader, self._pump}
        try:
            done, _ = await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            # The ASGI server cancelled the connection (an aborted socket, or shutdown).
            # Stop reading, but leave the session and its engine pump alive: the slot
            # decides whether this is a reconnectable gap or the end.
            reader.cancel()
            self.sink.detach()
            raise
        if self._pump is not None and self._pump in done:
            # The engine ended the session: stop reading and report what it raised.
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            self.sink.detach()
            self._pump.result()
            return
        # The client is gone; from here the pump's events are buffered for a reconnection.
        self.sink.detach()
        reader.result()

    async def _read_loop(self) -> None:
        while True:
            try:
                message = await self.websocket.receive()
            except WebSocketDisconnect:
                return
            if message["type"] == "websocket.disconnect":
                return
            raw: str | bytes | None = message.get("text")
            if raw is None:
                raw = message.get("bytes")
            if raw is None:
                continue
            await self.session.handle_raw(raw)
