"""The Realtime <-> Brain bridge (T3.7): `transfer_to_human` in both directions.

DESIGN.md 5.3 has three rows this module owns, and nothing else:

- the Brain emits `transfer_to_human` -> `response.function_call_arguments.delta` and
  `.done`, through `RealtimeSession.emit_function_call`;
- `conversation.item.create` with a `function_call_output` -> a `TaskInteractionReply`
  handed to `BusinessRun.respond`, which reaches `BrainSession.resolve`;
- Brain `share` / `question` -> never sent to the client.

**How the bridge obtains the run.** A `BusinessProject` is injected (`app.py` opens one
per process from `BusinessProvider`); the bridge never touches
`ProviderFactoryRegistry`. Talkover runs the Brain in-process with no Gander Gateway, so
there is no `WorkerRequest` to compile and no `WorkerRunChannel` to unwrap:
`BusinessProject.open_run(TaskRequest, autostart=True, intercept=True)` is documented as
exactly this caller's entry point (T3.4), and it is also what applies the T3.5 keyword
pre-interception. Injecting the project is what keeps the bridge testable against
`FakeEngine` + `FakeLLM` with no GPU and no network.

**Where a task comes from.** The Cerebellum asks for one by emitting the native tool call
`task_start` (`gander_runtime/lean_realtime.py`), which reaches this layer as an
`EngineStepEvent` with `is_tool_call` set, so `on_engine_event` is the whole task-start
path. The `TaskRequest` is assembled from two sources, as DESIGN.md 6.1 describes it
("trusted user turn text + task name + bounded context"):

- `instruction` is the latest trusted ASR text (`on_asr_event`), because that is what the
  T3.5 interceptor matches and what the Brain's first user message must be. With no
  transcript yet — ASR is a side channel and may lag or be disabled — the semantic task
  name is used instead, which still carries the customer's intent in the model's words.
- `context` is the last :data:`TRUSTED_CONTEXT_LIMIT` transcripts as `ContextEvent`s, and
  `metadata` keeps the task name and the Realtime session id for logging.

**The way back to the model.** The model expects a bounded tool response to its
`task_start`, and the Brain's `share` / `result` belong to the Cerebellum as worker
deliveries. Both are `EngineProtocol` calls (T1.4, DESIGN.md 4.6) and this module is their
only caller: `EngineProtocol.feed_tool_response` for the first, wrapping upstream
`DuplexLiveSession.feed_tool_response`, and `EngineProtocol.feed_worker_delivery` for the
second, wrapping `feed_runtime_event({"type": "worker_delivery", ...})`. The engine is
optional here only so that a unit test can build a bridge without one; with none, the
payload is logged and dropped, never sent to the client.

**The delivery envelope.** `_deliver` builds exactly what
`gander_runtime.lean_realtime.worker_delivery_response` produces for a Gateway-driven
deployment, because that is the shape the model was trained on: the semantic `task_name`
the model itself chose in its `task_start` (not the internal task id, which the model has
never seen), a `topic` from :data:`~talkover.engine.protocol.WORKER_DELIVERY_TOPICS`, and
the spoken hint in `content`. A Brain `share` is a `milestone`; a Brain `result` is the
terminal `final`, the one topic that carries a `status`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from gander_runtime.contracts import (
    ContextEvent,
    ContextSnapshot,
    ProviderEvent,
    TaskInteractionReply,
    TaskRequest,
    WorkState,
    new_id,
)

from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME
from talkover.engine.asr import Transcript
from talkover.engine.protocol import (
    WORKER_DELIVERY_TYPE,
    EngineError,
    EngineStepEvent,
)
from talkover.realtime import events as ev

if TYPE_CHECKING:  # pragma: no cover - typing only
    from talkover.brain.provider import BusinessProject, BusinessRun
    from talkover.engine.asr import AsrEvent
    from talkover.engine.protocol import EngineProtocol
    from talkover.realtime.session import RealtimeSession

__all__ = [
    "DELIVERY_TOPICS",
    "TASK_START_TOOL",
    "TERMINAL_STATUSES",
    "TRUSTED_CONTEXT_LIMIT",
    "BrainBridge",
    "PendingTransfer",
]

LOGGER = logging.getLogger(__name__)

#: The native front-brain tool that opens a Brain task (`gander_runtime/lean_realtime.py`).
TASK_START_TOOL = "task_start"

#: Native tool calls that are part of the task lifecycle but not of this task's scope.
_TASK_LIFECYCLE_TOOLS = frozenset({"task_send", "task_resolve"})

#: How many trusted transcripts ride along in `TaskRequest.context`.
TRUSTED_CONTEXT_LIMIT = 8


#: Brain event kind -> the `worker_delivery` topic the model was trained on.
DELIVERY_TOPICS = {"share": "milestone", "result": "final"}

#: The `status` values a `final` delivery may carry (`DeliveryRecord.__post_init__`).
TERMINAL_STATUSES = frozenset({"completed", "partial", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class PendingTransfer:
    """One `transfer_to_human` the client is executing, keyed by the Brain's `call_id`."""

    call_id: str
    name: str
    item_id: str
    interaction_id: str
    task_id: str
    session_id: str
    generation: int
    run: BusinessRun


class BrainBridge:
    """Connects one `RealtimeSession` to one `BusinessProject`.

    Lives as long as the Realtime session does, across the T2.8 reconnect window, because
    a Brain task must survive a dropped socket. :meth:`aclose` is what ends it, and
    `app.py` calls it when the session slot is released.
    """

    def __init__(
        self,
        project: BusinessProject,
        *,
        session: RealtimeSession | None = None,
        engine: EngineProtocol | None = None,
    ) -> None:
        self.project = project
        self.engine = engine
        self.session: RealtimeSession | None = session
        self._runs: dict[str, BusinessRun] = {}
        self._task_names: dict[str, str] = {}
        self._pumps: set[asyncio.Task[None]] = set()
        self._pending: dict[str, PendingTransfer] = {}
        self._transcripts: deque[Transcript] = deque(maxlen=TRUSTED_CONTEXT_LIMIT)
        self._seq = 0
        self._warned_about_tools = False
        self._closed = False

    # -- views -------------------------------------------------------------

    @property
    def pending_call_ids(self) -> tuple[str, ...]:
        """`call_id`s emitted to the client and not yet answered."""
        return tuple(self._pending)

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Brain tasks this session has open."""
        return tuple(self._runs)

    @property
    def trusted_text(self) -> str:
        """The latest trusted ASR text, which is what a `task_start` is about."""
        return self._transcripts[-1].text if self._transcripts else ""

    def bind(self, session: RealtimeSession) -> None:
        """Attach the protocol session; `app.py` calls this right after building both."""
        self.session = session

    # -- inbound: the side channels ----------------------------------------

    async def on_asr_event(self, event: AsrEvent) -> None:
        """Record the trusted text of a finished ASR segment (DESIGN.md 4.3).

        Same signature as `RealtimeSession.on_asr_event`, so the application feeds both
        from one ASR stream. Only `Transcript` is kept; the VAD signals belong to turn
        detection.
        """
        if isinstance(event, Transcript):
            self.note_transcript(event)

    def note_transcript(self, transcript: Transcript) -> None:
        """Push one trusted transcript into the bounded context."""
        text = transcript.text.strip()
        if not text:
            return
        self._transcripts.append(transcript)

    async def on_engine_event(self, step: EngineStepEvent) -> None:
        """Open a Brain task for every `task_start` the Cerebellum emits.

        Every other native tool call is logged and left alone: `task_send` /
        `task_resolve` are the rest of the task lifecycle (out of scope for T3.7) and an
        unknown name is a model error, not a protocol one.
        """
        if self._closed or not step.is_tool_call:
            return
        if step.tool_error:
            LOGGER.warning("engine reported a malformed tool call: %s", step.tool_error)
            await self._answer_tool_call(step, status="error", reason=step.tool_error)
            return
        started: list[str] = []
        for call in step.tool_calls:
            name = str(call.get("name") or "")
            if name == TASK_START_TOOL:
                started.append(await self._start_task(call.get("arguments")))
            elif name in _TASK_LIFECYCLE_TOOLS:
                LOGGER.info("ignoring %s: the task lifecycle beyond task_start is not wired", name)
            else:
                LOGGER.warning("ignoring unknown native tool call %r", name)
        await self._answer_tool_call(step, status="ok", task_ids=started)

    # -- inbound: the client ------------------------------------------------

    async def on_function_call_output(self, item: ev.FunctionCallOutputItem) -> None:
        """Route one client `function_call_output` back into the Brain loop.

        An unknown `call_id` is rejected through the session's own error path
        (`invalid_value` on `item.call_id`, profile section 8.4): the client answered a
        call this session never made, so there is no interaction to resolve.
        """
        pending = self._pending.pop(item.call_id, None)
        if pending is None:
            raise self._reject(
                f"call_id {item.call_id!r} does not name a pending Brain tool call.",
                "item.call_id",
            )
        reply = TaskInteractionReply(
            task_id=pending.task_id,
            session_id=pending.session_id,
            generation=pending.generation,
            interaction_id=pending.interaction_id,
            text=item.output,
            metadata={"tool": pending.name, "call_id": pending.call_id},
        )
        try:
            accepted = await pending.run.respond(reply)
        except (ValueError, RuntimeError) as exc:
            # The run moved on (a new generation, or it closed) between the call and the
            # answer. The customer's transfer is already in the client's hands, so this
            # ends the round trip rather than the session.
            LOGGER.warning("Brain refused the output of call %s: %s", pending.call_id, exc)
            return
        if not accepted:
            LOGGER.warning(
                "interaction %s was already resolved when call %s was answered",
                pending.interaction_id,
                pending.call_id,
            )

    # -- outbound: the Brain ------------------------------------------------

    async def _start_task(self, arguments: Any) -> str:
        """Open one `BusinessRun` for a `task_start` call and pump its events."""
        task_name = _task_name(arguments)
        request = self._task_request(task_name)
        run = await self.project.open_run(request, autostart=True, intercept=True)
        self._runs[request.task_id] = run
        self._task_names[request.task_id] = task_name
        pump = asyncio.create_task(
            self._pump(request.task_id, run), name=f"brain-bridge-{task_name}"
        )
        self._pumps.add(pump)
        pump.add_done_callback(self._pumps.discard)
        LOGGER.info(
            "opened Brain task %s (%r) for realtime session %s",
            request.task_id,
            task_name,
            request.session_id,
        )
        return request.task_id

    def _task_request(self, task_name: str) -> TaskRequest:
        session_id = self.session.session_id if self.session is not None else new_id("sess")
        instruction = self.trusted_text or task_name
        return TaskRequest(
            task_id=new_id("task"),
            session_id=session_id,
            generation=1,
            instruction=instruction,
            context=ContextSnapshot(session_id, self._context_events(session_id)),
            work_state=WorkState(),
            metadata={"task_name": task_name, "realtime_session_id": session_id},
        )

    def _context_events(self, session_id: str) -> tuple[ContextEvent, ...]:
        events: list[ContextEvent] = []
        for transcript in self._transcripts:
            self._seq += 1
            events.append(
                ContextEvent(
                    session_id=session_id,
                    seq=self._seq,
                    role="user",
                    kind="audio_transcript",
                    text=transcript.text.strip(),
                    start_ms=int(transcript.start_ms),
                    end_ms=int(transcript.end_ms),
                )
            )
        return tuple(events)

    async def _pump(self, task_id: str, run: BusinessRun) -> None:
        """Translate one run's `ProviderEvent`s until it closes."""
        try:
            async for event in run.events():
                await self._on_provider_event(task_id, run, event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # A provider failure must not take the voice session with it.
            LOGGER.exception("Brain run %s failed", task_id)
        finally:
            self._runs.pop(task_id, None)
            self._task_names.pop(task_id, None)

    async def _on_provider_event(
        self, task_id: str, run: BusinessRun, event: ProviderEvent
    ) -> None:
        if event.kind == "interaction":
            await self._on_interaction(run, event)
        elif event.kind == "share":
            await self._deliver(task_id, event.share.text if event.share else "", "share")
        elif event.kind == "result":
            result = event.result
            await self._deliver(
                task_id,
                result.full_result if result else "",
                "result",
                status=result.status if result else "",
            )
            await run.close()
        elif event.kind == "interaction_resolved":
            self._forget(event.interaction_id)
        elif event.kind == "error":
            # The provider already ended the invocation with a spoken fallback; reporting
            # `engine_error` here would close the WebSocket over a recoverable failure.
            LOGGER.error("Brain run %s reported an error: %s", task_id, event.error)

    async def _on_interaction(self, run: BusinessRun, event: ProviderEvent) -> None:
        """Turn one `TaskInteraction` into the client's `function_call` pair."""
        interaction = event.interaction
        if interaction is None:  # pragma: no cover - the provider always fills it
            return
        metadata = interaction.metadata or {}
        name = str(metadata.get("tool") or "")
        call_id = str(metadata.get("call_id") or "")
        if name != TRANSFER_TO_HUMAN_NAME or not call_id:
            # `query_ticket` / `query_order` execute inside the Brain (DESIGN.md 6.3), so
            # anything else arriving here is a provider bug, not something to hand out.
            LOGGER.error(
                "refusing to forward interaction %s (tool %r)", interaction.interaction_id, name
            )
            return
        session = self.session
        if session is None:
            LOGGER.error("no realtime session is bound; dropping tool call %s", call_id)
            return
        self._warn_if_tool_undeclared(session, name)
        arguments = json.dumps(dict(metadata.get("arguments") or {}), ensure_ascii=False)
        item_id = await session.emit_function_call(call_id, name, arguments)
        self._pending[call_id] = PendingTransfer(
            call_id=call_id,
            name=name,
            item_id=item_id,
            interaction_id=interaction.interaction_id,
            task_id=interaction.task_id,
            session_id=interaction.session_id,
            generation=interaction.generation,
            run=run,
        )

    # -- outbound: the Cerebellum ------------------------------------------

    async def _deliver(self, task_id: str, text: str, kind: str, *, status: str = "") -> None:
        """Hand a Brain `share` / `result` to the Cerebellum, never to the client.

        The envelope is `worker_delivery_response`'s (see the module docstring): the model
        is addressed with the task name it chose itself, and only the terminal `final`
        topic carries a status. A non-terminal `TaskResult.status` (the provider never
        emits one) is reported as `completed`, because upstream refuses the envelope
        otherwise and a dropped final leaves the model waiting.
        """
        text = (text or "").strip()
        if not text:
            return
        topic = DELIVERY_TOPICS[kind]
        payload: dict[str, Any] = {
            "type": WORKER_DELIVERY_TYPE,
            "task_name": self._task_names.get(task_id, task_id),
            "topic": topic,
            "content": text,
        }
        if topic == "final":
            payload["status"] = status if status in TERMINAL_STATUSES else "completed"
        engine = self.engine
        if engine is None:
            LOGGER.debug("no engine channel; dropping the %s of task %s", kind, task_id)
            return
        try:
            await engine.feed_worker_delivery(payload)
        except EngineError as exc:
            # The model refused the envelope (too long, or a tool response is pending).
            # One unspoken update is not worth ending the call over.
            LOGGER.warning("the %s of task %s never reached the model: %s", kind, task_id, exc)

    async def _answer_tool_call(
        self,
        step: EngineStepEvent,
        *,
        status: str,
        task_ids: list[str] | None = None,
        reason: str = "",
    ) -> None:
        """Answer the model's native tool call in the bounded front-brain shape.

        `control_tool_response` (`gander_runtime/lean_realtime.py`) is the shape: a status,
        the task handles, and an optional reason. The model blocks until it arrives, which
        is why a missing channel is logged at WARNING rather than at DEBUG.
        """
        if not step.tool_response_expected:
            return
        payload: dict[str, Any] = {"status": status, "task_ids": list(task_ids or ())}
        if reason:
            payload["reason"] = reason
        engine = self.engine
        if engine is None:
            LOGGER.warning(
                "no engine channel; the model's tool call at unit %d stays unanswered",
                step.unit_index,
            )
            return
        try:
            await engine.feed_tool_response(payload)
        except EngineError as exc:
            LOGGER.error(
                "the model's tool call at unit %d stays unanswered: %s", step.unit_index, exc
            )

    # -- lifecycle ----------------------------------------------------------

    async def aclose(self) -> None:
        """Close every open run and stop pumping their events."""
        if self._closed:
            return
        self._closed = True
        self._pending.clear()
        for run in tuple(self._runs.values()):
            with contextlib.suppress(Exception):
                await run.close()
        self._runs.clear()
        self._task_names.clear()
        pumps = tuple(self._pumps)
        for pump in pumps:
            pump.cancel()
        if pumps:
            await asyncio.gather(*pumps, return_exceptions=True)
        self._pumps.clear()

    # -- helpers ------------------------------------------------------------

    def _forget(self, interaction_id: str | None) -> None:
        if interaction_id is None:
            return
        for call_id, pending in tuple(self._pending.items()):
            if pending.interaction_id == interaction_id:
                self._pending.pop(call_id, None)

    def _reject(self, message: str, param: str) -> ev.ProtocolError:
        session = self.session
        if session is not None:
            return session.invalid_value(message, param)
        return ev.ProtocolError("invalid_value", message, param, None)

    def _warn_if_tool_undeclared(self, session: RealtimeSession, name: str) -> None:
        """Warn once when the client never declared `transfer_to_human` in `session.tools`.

        The call is emitted anyway (profile section 9): the decision to hand the customer
        to a person is the Brain's, and dropping it would strand a caller who asked for a
        human because the client's tool list was incomplete.
        """
        if self._warned_about_tools:
            return
        tools = session.session.get("tools") or []
        declared = any(isinstance(tool, Mapping) and tool.get("name") == name for tool in tools)
        if declared:
            return
        self._warned_about_tools = True
        LOGGER.warning("session.tools does not declare %r; emitting the function call anyway", name)


def _task_name(arguments: Any) -> str:
    """The semantic name of a `task_start` call, or a neutral default."""
    if isinstance(arguments, Mapping):
        name = arguments.get("name")
        if isinstance(name, str) and name.strip():
            return name.strip()
    return "business"
