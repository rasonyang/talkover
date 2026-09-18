"""`WorkerProvider` implementation, registered in `ProviderFactoryRegistry` under `business`.

DESIGN.md 6.2 splits the Cerebellum-facing surface into three levels, mirroring the
upstream Codex provider (`gander_runtime/providers/codex.py`):

==========  ==================  ====================================================
Level       Talkover object     Holds
==========  ==================  ====================================================
Provider    `BusinessProvider`  the `BrainConfig`, the shared `BrainLLM` and
                                `ToolExecutor` (one `httpx` client each), the
                                `BackendCapabilities`
Project     `BusinessProject`   nothing device-bound; it only constructs runs
Run         `BusinessRun`       one `BrainSession`, the pending interaction ids and
                                the `ProviderEvent` queue
==========  ==================  ====================================================

`BrainSession` (T3.3) never imports `gander_runtime`; it yields the neutral events
`Share` / `Question` / `ClientTool` / `Result` and this module translates them:

===========================  ==========================================================
Brain event                  `ProviderEvent`
===========================  ==========================================================
`Share` (main lane)          `kind="share"`, `ShareEvent(kind="important")`
`Question` (main lane)       `kind="share"`, `ShareEvent(kind="need_input")`
`Result` (main lane)         `kind="result"`, `TaskResult(status="completed"|"failed")`
any event on the fork lane   `kind="share"`, `ShareEvent(kind="answer")` -- the kind the
                             Codex provider uses for a side-query answer
                             (`providers/codex.py:1565`)
`ClientTool`                 `kind="interaction"`, `TaskInteraction(kind="user_input")`
===========================  ==========================================================

`transfer_to_human` travels as an interaction because upstream `ProviderEvent.kind`
(`contracts.py:268`) is the closed `Literal`
`share | interaction | interaction_resolved | result | error`: there is no `client_tool`
member and upstream is never forked (DESIGN.md 6.3). The arguments the Realtime layer
(T3.7) needs for `response.function_call_arguments.delta` / `.done` ride in
`TaskInteraction.metadata` as `{"tool", "call_id", "arguments"}`. The reply comes back as
a `TaskInteractionReply` through :meth:`BusinessRun.respond`, which validates the ids and
the generation, emits `ProviderEvent(kind="interaction_resolved", ...)` and feeds the
output to :meth:`BrainSession.resolve`. Upstream `task_resolve` is not that path: its
`TaskResolveAction` (`coordination.py:66`) is an authorization decision.

A `task_start` whose trusted text already asks for a human agent never reaches the LLM:
`BusinessProject.open_run` runs `TransferInterceptor` (T3.5, `intercept.py`,
DESIGN.md 6.4) first and, on a keyword hit, emits the `transfer_to_human` interaction
straight away instead of starting the Brain loop. A miss starts the loop unchanged.

The upstream duck type a run must satisfy is the one `WorkerRunChannel`
(`gateway.py:543`) calls: `events()`, `steer(TaskUpdate)`, `query(TaskQuery)`,
`respond(TaskInteractionReply) -> bool`, `cancel()`, `close()`, plus an optional
`thread_id` attribute. `WorkerProject.start` therefore returns a `WorkerRunChannel`
wrapping a `BusinessRun`, and the Gateway sees ordinary `WorkerEvent`s.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gander_runtime.contracts import (
    ContextEvent,
    ContextSnapshot,
    ProviderEvent,
    ShareEvent,
    ShareKind,
    TaskInteraction,
    TaskInteractionReply,
    TaskQuery,
    TaskRequest,
    TaskResult,
    TaskUpdate,
    WorkState,
    new_id,
)
from gander_runtime.coordination import BackendCapabilities, ProjectRecord, WorkerRequest
from gander_runtime.gateway import WorkerControl, WorkerRunChannel
from gander_runtime.providers import (
    ProviderBuildContext,
    ProviderFactoryRegistry,
    ProviderRegistration,
)

from talkover.brain.intercept import TransferInterceptor, TransferMatch
from talkover.brain.llm import BrainLLM, create_llm
from talkover.brain.session import (
    BrainEvent,
    BrainSession,
    ClientTool,
    Question,
    Result,
    Share,
    ToolRunner,
)
from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME, ToolExecutor
from talkover.config import BrainConfig, load_brain_config

__all__ = [
    "BUSINESS_CAPABILITIES",
    "BUSINESS_PROVIDER_KEY",
    "BUSINESS_PROVIDER_NAME",
    "BUSINESS_PROVIDER_REGISTRATION",
    "TRANSFER_PROMPT",
    "BusinessProject",
    "BusinessProvider",
    "BusinessProviderSettings",
    "BusinessRun",
    "build_business_provider",
    "build_session",
    "provider_registry",
    "task_request_from_worker_request",
]

LOGGER = logging.getLogger(__name__)

#: The public configuration key in `ProviderFactoryRegistry` (DESIGN.md 6.1).
BUSINESS_PROVIDER_KEY = "business"

#: `WorkerProvider.name`, and therefore `ProjectRecord.provider_name`. Upstream's
#: `ProviderRegistration` asserts that the built provider carries exactly this name.
BUSINESS_PROVIDER_NAME = "business"

#: What a client without tool support hears when `transfer_to_human` carries no reason.
TRANSFER_PROMPT = "Let me put you through to a human agent."

#: DESIGN.md 6.2 / 6.3. `steering` is `next_turn` because `BrainSession.update` appends to
#: the conversation and runs a fresh bounded loop; it cannot interrupt an LLM call in
#: flight. `side_queries` is `isolated_fork` because `BrainSession.query` answers on a deep
#: copy that never touches the main conversation. `context_provisioning` stays
#: `push_bounded`: `pull` would require upstream's `context_fetch` worker tool, which the
#: Brain does not expose.
BUSINESS_CAPABILITIES = BackendCapabilities(
    steering="next_turn",
    side_queries="isolated_fork",
    terminal_side_queries="none",
    interactions=True,
    blocking_granularity="run",
    authority_enforcement="none",
    structured_events="native",
    trusted_risk_signals=False,
    session_resume=False,
    modalities=frozenset({"text"}),
    # DESIGN.md 5.1: one Gander session per process, so one task at a time.
    max_parallel_projects=1,
    context_provisioning="push_bounded",
    session="stateful",
    worker_tools=frozenset(),
)


def build_session(config: BrainConfig, llm: BrainLLM, executor: ToolRunner) -> BrainSession:
    """Construct a `BrainSession` from the `brain` config section (DESIGN.md 6.2).

    A `fallback_texts` key left null keeps the built-in English constant in
    `talkover.brain.session`, so only the keys that were set are passed through.
    """
    texts = config.fallback_texts
    overrides: dict[str, Any] = {}
    if texts.round_cap is not None:
        overrides["round_cap_text"] = texts.round_cap
    if texts.failure is not None:
        overrides["failure_text"] = texts.failure
    if texts.no_answer is not None:
        overrides["no_answer_text"] = texts.no_answer
    return BrainSession(
        llm,
        executor,
        max_rounds=config.max_rounds,
        spoken_numbers=config.spoken_numbers,
        **overrides,
    )


def task_request_from_worker_request(request: WorkerRequest) -> TaskRequest:
    """Bridge the Gateway-facing `WorkerRequest` to the provider-facing `TaskRequest`.

    Shaped after `_CodexWorkerProject.start` (`providers/codex.py:553`): `session_id` is
    the provider-neutral `lineage_id`, and the orchestration identifiers the Brain does not
    need are preserved in `metadata` for logging and for T3.7.
    """
    return TaskRequest(
        task_id=request.task_id,
        session_id=request.lineage_id,
        generation=request.generation,
        instruction=request.instruction,
        context=ContextSnapshot(request.lineage_id, ()),
        work_state=WorkState(),
        request_id=request.run_id,
        metadata={
            "owner_id": request.owner_id,
            "project_id": request.project_id,
            "lineage_id": request.lineage_id,
            "orchestration_run_id": request.run_id,
            "worker_request_kind": request.kind,
            "original_turn": request.original_turn,
        },
    )


def _event_text(event: ContextEvent | None) -> str:
    return event.text.strip() if event is not None else ""


@dataclass(frozen=True, slots=True)
class _PendingCall:
    """One `transfer_to_human` the Realtime client is executing.

    `from_session` is False for a call raised by the T3.5 pre-interception, where no LLM
    round is waiting for the output and there is nothing to resolve inside the loop.
    """

    call_id: str
    name: str
    from_session: bool = True


class BusinessRun:
    """One Gander task: one `BrainSession`, one `ProviderEvent` queue.

    Every Brain invocation (`start`, `steer`, `query`, `respond`) is drained by its own
    task, so the caller returns immediately and the events surface through :meth:`events`,
    exactly like `_CodexRun` (`providers/codex.py:836`). Main-lane invocations take
    `_main_lock` and therefore serialise; a fork query runs alongside them because
    `BrainSession.query` works on a deep copy.

    The run never calls `BrainSession.aclose`: the `BrainLLM` is owned by the provider and
    shared by every run.
    """

    def __init__(
        self,
        request: TaskRequest,
        session: BrainSession,
        *,
        control: WorkerControl | None = None,
        on_close: Callable[[BusinessRun], Awaitable[None]] | None = None,
    ) -> None:
        self.request = request
        self.session = session
        self.worker_control = control
        self.generation = request.generation
        # `WorkerRunChannel` reads this as the backend session id; the Brain has no
        # provider-side thread, so the lineage is the closest honest value.
        self.thread_id = request.session_id
        self._queue: asyncio.Queue[ProviderEvent | None] = asyncio.Queue()
        self._main_lock = asyncio.Lock()
        self._tasks: set[asyncio.Task[None]] = set()
        self._pending: dict[str, _PendingCall] = {}
        self._on_close = on_close
        self._closed = False
        #: Set by `BusinessProject.open_run` when the T3.5 rule answered this task
        #: without the LLM; None for a run that entered the Brain loop.
        self.intercepted: TransferMatch | None = None

    # -- introspection ---------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_interactions(self) -> tuple[str, ...]:
        """Interaction ids the Realtime client has not answered yet."""
        return tuple(self._pending)

    # -- lifecycle -------------------------------------------------------------------

    async def start(self) -> None:
        """Run the task named by `TaskRequest.instruction` (`BrainSession.start`)."""
        self._ensure_open()
        self._spawn(self._drain_main(self.session.start(self.request.instruction)), "start")

    def events(self) -> AsyncIterator[ProviderEvent]:
        return self._iterate_events()

    async def _iterate_events(self) -> AsyncIterator[ProviderEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                return
            yield event

    async def steer(self, update: TaskUpdate) -> None:
        """Apply a `TaskUpdate` (upstream `task_send` on the main lane).

        `mode` `replace` is treated like `additive`: a spoken turn cannot retract what the
        customer already said, and `BrainSession` has no way to drop the conversation
        prefix. `cancel` ends the run.
        """
        self._ensure_open()
        if update.mode == "cancel":
            await self.cancel()
            return
        instruction = (update.instruction or "").strip() or _event_text(update.event)
        if not instruction:
            return
        self._spawn(self._drain_main(self.session.update(instruction)), "update")

    async def query(self, query: TaskQuery) -> None:
        """Answer a `TaskQuery` (upstream `task_send` on the fork lane)."""
        self._ensure_open()
        if query.task_id != self.request.task_id or query.session_id != self.request.session_id:
            raise ValueError("side query belongs to another task")
        if query.generation != self.generation:
            raise ValueError("side query targets a stale generation")
        self._spawn(self._drain_fork(self.session.query(query.question), query), "query")

    async def respond(self, reply: TaskInteractionReply) -> bool:
        """Route a client `function_call_output` back into the loop (DESIGN.md 6.3).

        Returns False when the interaction is unknown or already resolved, which is what
        `WorkerRunChannel.send` reports to the Gateway as "a competing client won".
        """
        self._ensure_open()
        if reply.task_id != self.request.task_id or reply.session_id != self.request.session_id:
            raise ValueError("interaction response belongs to another task")
        if reply.generation != self.generation:
            raise ValueError("interaction response targets a stale generation")
        pending = self._pending.pop(reply.interaction_id, None)
        if pending is None:
            return False
        await self._emit(
            ProviderEvent(
                kind="interaction_resolved",
                generation=self.generation,
                interaction_id=reply.interaction_id,
                interaction_resolution="response",
            )
        )
        if not pending.from_session:
            # A pre-intercepted call (T3.5) never entered the loop, so the task is done
            # the moment the client reports the transfer, and there is nothing left to
            # say: the reply carries a `function_call_output`, not spoken text.
            await self._emit(self._result("", ok=True))
            return True
        self._spawn(
            self._drain_main(self.session.resolve(reply.text, call_id=pending.call_id)),
            "resolve",
        )
        return True

    async def emit_client_tool(
        self, name: str, arguments: dict[str, Any], *, call_id: str | None = None
    ) -> str:
        """Raise a client tool call without running the LLM loop; returns the id.

        This is the seam T3.5 (`intercept.py`) uses: on a `transfer_to_human` keyword hit
        the run is opened with `autostart=False` and the interaction is emitted straight
        away, so the customer never waits for a model round.
        """
        self._ensure_open()
        event = self._interaction(
            ClientTool(call_id or new_id("call"), name, dict(arguments)), from_session=False
        )
        await self._emit(event)
        assert event.interaction is not None
        return event.interaction.interaction_id

    async def cancel(self) -> None:
        if self._closed:
            return
        await self._clear_interactions()
        await self._close()

    async def close(self) -> None:
        await self._close()

    async def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
        self._pending.clear()
        await self._queue.put(None)
        if self._on_close is not None:
            await self._on_close(self)

    async def _clear_interactions(self) -> None:
        """Report every unanswered interaction as resolved by the turn ending."""
        pending = tuple(self._pending)
        self._pending.clear()
        for interaction_id in pending:
            await self._emit(
                ProviderEvent(
                    kind="interaction_resolved",
                    generation=self.generation,
                    interaction_id=interaction_id,
                    interaction_resolution="turn",
                )
            )

    # -- draining --------------------------------------------------------------------

    async def _drain_main(self, events: AsyncIterator[BrainEvent]) -> None:
        async with self._main_lock:
            await self._drain(events, self._emit_main)

    async def _drain_fork(self, events: AsyncIterator[BrainEvent], query: TaskQuery) -> None:
        async def emit(event: BrainEvent) -> None:
            await self._emit_fork(event, query)

        await self._drain(events, emit)

    async def _drain(
        self,
        events: AsyncIterator[BrainEvent],
        emit: Callable[[BrainEvent], Awaitable[None]],
    ) -> None:
        try:
            async for event in events:
                await emit(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A provider must not die on a loop failure; it reports one and stays usable.
            LOGGER.exception("brain session failed for task %s", self.request.task_id)
            await self._emit(
                ProviderEvent(
                    kind="error",
                    generation=self.generation,
                    error=f"brain session failed: {exc}",
                )
            )
        finally:
            with contextlib.suppress(Exception):
                await events.aclose()

    # -- translation -----------------------------------------------------------------

    async def _emit_main(self, event: BrainEvent) -> None:
        if isinstance(event, Share):
            await self._emit(self._share(event.text, "important"))
        elif isinstance(event, Question):
            await self._emit(self._share(event.text, "need_input"))
        elif isinstance(event, ClientTool):
            await self._emit(self._interaction(event))
        elif isinstance(event, Result):
            await self._emit(self._result(event.text, ok=event.ok))

    async def _emit_fork(self, event: BrainEvent, query: TaskQuery) -> None:
        """A side query answers with shares only; it must never end the main task.

        Both the intermediate `Share` and the closing `Result` become
        `ShareEvent(kind="answer")`, the way the Codex provider reports a side-query
        answer (`providers/codex.py:1565`). A `Result` mapped to
        `ProviderEvent(kind="result")` here would reach the Gateway as a `DonePayload` and
        terminate the run.
        """
        if isinstance(event, ClientTool):
            # `BrainSession` already refuses `transfer_to_human` on the fork lane.
            LOGGER.warning("ignoring client tool %r raised by a side query", event.name)
            return
        await self._emit(
            self._share(
                event.text,
                "answer",
                state_patch={"_gander": {"lane": "query", "request_id": query.request_id}},
            )
        )

    def _share(
        self, text: str, kind: ShareKind, *, state_patch: dict[str, Any] | None = None
    ) -> ProviderEvent:
        return ProviderEvent(
            kind="share",
            generation=self.generation,
            share=ShareEvent(
                task_id=self.request.task_id,
                session_id=self.request.session_id,
                generation=self.generation,
                kind=kind,
                text=text,
                state_patch=state_patch or {},
            ),
        )

    def _result(self, text: str, *, ok: bool) -> ProviderEvent:
        return ProviderEvent(
            kind="result",
            generation=self.generation,
            result=TaskResult(
                task_id=self.request.task_id,
                session_id=self.request.session_id,
                generation=self.generation,
                status="completed" if ok else "failed",
                full_result=text,
            ),
        )

    def _interaction(self, call: ClientTool, *, from_session: bool = True) -> ProviderEvent:
        interaction_id = new_id("interaction")
        self._pending[interaction_id] = _PendingCall(call.call_id, call.name, from_session)
        arguments = dict(call.arguments)
        reason = arguments.get("reason")
        prompt = reason.strip() if isinstance(reason, str) and reason.strip() else TRANSFER_PROMPT
        return ProviderEvent(
            kind="interaction",
            generation=self.generation,
            interaction=TaskInteraction(
                task_id=self.request.task_id,
                session_id=self.request.session_id,
                generation=self.generation,
                interaction_id=interaction_id,
                # `InteractionKind` is `approval | user_input`; this is an action the
                # client performs and reports back, not an authorization prompt.
                kind="user_input",
                prompt=prompt,
                metadata={
                    "tool": call.name,
                    "call_id": call.call_id,
                    "arguments": arguments,
                },
            ),
        )

    # -- internals -------------------------------------------------------------------

    async def _emit(self, event: ProviderEvent) -> None:
        if self._closed:
            return
        await self._queue.put(event)

    def _spawn(self, coro: Any, name: str) -> asyncio.Task[None]:
        task = asyncio.create_task(coro, name=f"brain-{name}-{self.request.task_id}")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("business Brain run is closed")


class BusinessProject:
    """One `ProjectRecord`. It holds nothing device-bound and only constructs runs."""

    def __init__(self, provider: BusinessProvider, project: ProjectRecord) -> None:
        self.provider = provider
        self.project = project
        self._runs: set[BusinessRun] = set()
        self._closed = False

    async def start(self, request: WorkerRequest, control: WorkerControl) -> WorkerRunChannel:
        """`WorkerProject.start`: the Gateway-facing entry point."""
        if self._closed:
            raise RuntimeError("business Brain project is closed")
        if request.project_id != self.project.project_id:
            raise ValueError("worker request belongs to another project")
        run = await self.open_run(task_request_from_worker_request(request), control=control)
        return WorkerRunChannel(request, control, run, self.provider.capabilities)

    async def open_run(
        self,
        request: TaskRequest,
        *,
        control: WorkerControl | None = None,
        autostart: bool = True,
        intercept: bool = True,
    ) -> BusinessRun:
        """Build a run directly from a `TaskRequest`.

        The Realtime layer (T3.7) uses this rather than `start`, because Talkover drives
        one task in-process and has no Gateway to build a `WorkerRequest`. `autostart`
        False creates the run without entering the Brain loop, which is what the T3.5
        pre-interception needs before it calls `BusinessRun.emit_client_tool`.

        With `autostart`, the task-start text is first offered to the provider's
        `TransferInterceptor` (DESIGN.md 6.4): a keyword hit emits the `transfer_to_human`
        interaction with zero LLM calls and records the hit on `BusinessRun.intercepted`;
        a miss starts the Brain loop unchanged. `intercept` False forces the loop, which
        is how a caller that has already applied the rule avoids applying it twice.
        """
        if self._closed:
            raise RuntimeError("business Brain project is closed")
        run = BusinessRun(
            request,
            self.provider.new_session(),
            control=control,
            on_close=self._forget,
        )
        self._runs.add(run)
        if autostart:
            await self._start_run(run, intercept=intercept)
        return run

    async def _start_run(self, run: BusinessRun, *, intercept: bool) -> None:
        """Pre-intercept the transfer request, else enter the Brain loop."""
        interceptor = self.provider.interceptor
        match = interceptor.match(run.request.instruction) if intercept else None
        if match is None:
            await run.start()
            return
        run.intercepted = match
        LOGGER.info(
            "pre-intercepted transfer for task %s on keyword %r",
            run.request.task_id,
            match.keyword,
        )
        await run.emit_client_tool(TRANSFER_TO_HUMAN_NAME, interceptor.transfer_arguments(match))

    async def _forget(self, run: BusinessRun) -> None:
        self._runs.discard(run)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for run in tuple(self._runs):
            await run.close()
        self._runs.clear()


class BusinessProvider:
    """The `WorkerProvider` Talkover registers under the key `business`.

    The `BrainLLM` and the `ToolExecutor` are built once and shared by every run: both
    wrap an `httpx.AsyncClient` and neither keeps per-task state. Injecting them is how
    `tests/brain` runs the whole chain against `FakeLLM` with no network.
    """

    name = BUSINESS_PROVIDER_NAME
    capabilities = BUSINESS_CAPABILITIES

    def __init__(
        self,
        config: BrainConfig | None = None,
        *,
        llm: BrainLLM | None = None,
        executor: ToolRunner | None = None,
        interceptor: TransferInterceptor | None = None,
    ) -> None:
        self.config = config if config is not None else BrainConfig()
        self._llm = llm
        self._owns_llm = llm is None
        self._executor = executor
        self._owns_executor = executor is None
        self._interceptor = (
            interceptor if interceptor is not None else TransferInterceptor.from_config(self.config)
        )
        self._projects: dict[str, BusinessProject] = {}
        self._closed = False

    @property
    def llm(self) -> BrainLLM:
        if self._llm is None:
            self._llm = create_llm(self.config.llm)
        return self._llm

    @property
    def interceptor(self) -> TransferInterceptor:
        """The T3.5 keyword rule, built once from `brain.transfer_keywords`."""
        return self._interceptor

    @property
    def executor(self) -> ToolRunner:
        if self._executor is None:
            self._executor = ToolExecutor(self.config.business_api)
        return self._executor

    def new_session(self) -> BrainSession:
        return build_session(self.config, self.llm, self.executor)

    async def open_project(self, project: ProjectRecord) -> BusinessProject:
        if self._closed:
            raise RuntimeError("business WorkerProvider is closed")
        if project.provider_name != self.name:
            raise ValueError("project provider does not match the business WorkerProvider")
        existing = self._projects.get(project.project_id)
        if existing is not None:
            return existing
        opened = BusinessProject(self, project)
        self._projects[project.project_id] = opened
        return opened

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for project in tuple(self._projects.values()):
            await project.close()
        self._projects.clear()
        if self._owns_llm and self._llm is not None:
            await self._llm.aclose()
        if self._owns_executor and self._executor is not None:
            aclose = getattr(self._executor, "aclose", None)
            if aclose is not None:
                await aclose()


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BusinessProviderSettings:
    """`worker.settings` for the key `business`; upstream requires a dataclass.

    The Brain is configured by Talkover's own `brain` section (DESIGN.md section 9), so the
    settings only say which file to read and allow the two knobs a deployment is likely to
    tune without a second file.
    """

    config_path: str = ""
    max_rounds: int | None = None
    spoken_numbers: bool | None = None


def _resolve_config(
    context: ProviderBuildContext, settings: BusinessProviderSettings
) -> BrainConfig:
    if settings.config_path:
        path = Path(settings.config_path).expanduser()
        if not path.is_absolute():
            path = Path(context.workspace) / path
        config = load_brain_config(path)
    else:
        config = BrainConfig()
    overrides: dict[str, Any] = {}
    if settings.max_rounds is not None:
        overrides["max_rounds"] = settings.max_rounds
    if settings.spoken_numbers is not None:
        overrides["spoken_numbers"] = settings.spoken_numbers
    return dataclasses.replace(config, **overrides) if overrides else config


def build_business_provider(
    context: ProviderBuildContext, settings: BusinessProviderSettings
) -> BusinessProvider:
    return BusinessProvider(_resolve_config(context, settings))


BUSINESS_PROVIDER_REGISTRATION = ProviderRegistration(
    key=BUSINESS_PROVIDER_KEY,
    provider_name=BusinessProvider.name,
    settings_type=BusinessProviderSettings,
    build=build_business_provider,
)


def provider_registry(base: ProviderFactoryRegistry | None = None) -> ProviderFactoryRegistry:
    """Register the business provider, on `base` when the caller already has a registry."""
    registry = base if base is not None else ProviderFactoryRegistry()
    registry.register(BUSINESS_PROVIDER_REGISTRATION)
    return registry
