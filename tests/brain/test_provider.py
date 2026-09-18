"""Tests for the Brain `WorkerProvider` chain (T3.4). GPU-free, no network, no marker.

Everything runs against `FakeLLM` and a fake tool runner, so the whole provider ->
project -> run chain is exercised without a model, a business API or a Gander Gateway.
The cases cover what DESIGN.md 6.2 / 6.3 fix: the registry key, the `ProviderEvent`
translation per lane, and the `transfer_to_human` interaction round trip.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from gander_runtime.contracts import (
    ContextEvent,
    ContextSnapshot,
    ProviderEvent,
    TaskInteractionReply,
    TaskQuery,
    TaskRequest,
    TaskUpdate,
    WorkState,
    new_id,
)
from gander_runtime.coordination import (
    ContextPlan,
    ProjectRecord,
    WorkerPolicyView,
    WorkerRequest,
)
from gander_runtime.gateway import WorkerControl
from gander_runtime.providers import ProviderBuildContext

from talkover.brain.provider import (
    BUSINESS_CAPABILITIES,
    BUSINESS_PROVIDER_KEY,
    BUSINESS_PROVIDER_REGISTRATION,
    TRANSFER_PROMPT,
    BusinessProvider,
    BusinessRun,
    build_session,
    provider_registry,
)
from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME, ToolResult
from talkover.config import BrainConfig, FallbackTextsConfig

from .fake_llm import FakeLLM, text_response, tool_call_response

TASK_ID = "task_1"
SESSION_ID = "lineage_1"
PROJECT_ID = "project_1"


# --------------------------------------------------------------------------------------
# Doubles and helpers
# --------------------------------------------------------------------------------------


class FakeExecutor:
    """Replays a scripted `ToolResult` per tool name and records every call."""

    def __init__(self, results: dict[str, ToolResult] | None = None) -> None:
        self.results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.closed = False

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        self.calls.append((name, dict(arguments or {})))
        if name in self.results:
            return self.results[name]
        if name == TRANSFER_TO_HUMAN_NAME:
            return ToolResult(
                name=name,
                ok=True,
                data=dict(arguments or {}),
                client_tool=True,
            )
        return ToolResult(name=name, ok=True, data={"stub": name})

    async def aclose(self) -> None:
        self.closed = True


def task_request(instruction: str = "Where is my order?") -> TaskRequest:
    return TaskRequest(
        task_id=TASK_ID,
        session_id=SESSION_ID,
        generation=1,
        instruction=instruction,
        context=ContextSnapshot(SESSION_ID, ()),
        work_state=WorkState(),
    )


def project_record() -> ProjectRecord:
    return ProjectRecord(
        project_id=PROJECT_ID,
        owner_id="owner_1",
        label="customer service",
        provider_name=BusinessProvider.name,
    )


async def open_run(
    script: list[Any],
    *,
    executor: FakeExecutor | None = None,
    config: BrainConfig | None = None,
    instruction: str = "Where is my order?",
    autostart: bool = True,
) -> tuple[BusinessProvider, BusinessRun, FakeLLM, FakeExecutor]:
    llm = FakeLLM(script=script)
    runner = executor if executor is not None else FakeExecutor()
    provider = BusinessProvider(config, llm=llm, executor=runner)
    project = await provider.open_project(project_record())
    run = await project.open_run(task_request(instruction), autostart=autostart)
    return provider, run, llm, runner


async def collect(run: BusinessRun, count: int, *, timeout: float = 2.0) -> list[ProviderEvent]:
    """Take exactly `count` events off the run's stream, or fail the test."""
    events: list[ProviderEvent] = []
    stream = run.events()
    for _ in range(count):
        events.append(await asyncio.wait_for(anext(stream), timeout))
    return events


async def take(stream: Any, count: int, *, timeout: float = 2.0) -> list[ProviderEvent]:
    return [await asyncio.wait_for(anext(stream), timeout) for _ in range(count)]


# --------------------------------------------------------------------------------------
# Registration
# --------------------------------------------------------------------------------------


def test_provider_is_resolvable_by_key() -> None:
    registry = provider_registry()

    assert registry.keys == (BUSINESS_PROVIDER_KEY,)

    factory = registry.configure(BUSINESS_PROVIDER_KEY, {"max_rounds": 3})
    assert factory.provider_name == BusinessProvider.name

    provider = factory.create(
        ProviderBuildContext(workspace=".", runtime_dir=".", runtime_profile="task_scoped")
    )
    assert isinstance(provider, BusinessProvider)
    # The settings override reached the config the sessions are built from.
    assert provider.config.max_rounds == 3
    assert provider.capabilities is BUSINESS_CAPABILITIES


def test_registration_rejects_unknown_settings() -> None:
    registry = provider_registry()
    with pytest.raises(ValueError, match="Unknown setting"):
        registry.configure(BUSINESS_PROVIDER_KEY, {"model": "gpt"})


def test_registration_metadata_matches_the_provider() -> None:
    assert BUSINESS_PROVIDER_REGISTRATION.key == "business"
    assert BUSINESS_PROVIDER_REGISTRATION.provider_name == BusinessProvider.name
    # DESIGN.md 6.3: the transfer travels as an interaction, so the capability must say so.
    assert BUSINESS_CAPABILITIES.interactions is True


def test_build_session_applies_the_brain_config() -> None:
    config = BrainConfig(
        max_rounds=2,
        spoken_numbers=True,
        fallback_texts=FallbackTextsConfig(round_cap="cap", failure="fail"),
    )
    session = build_session(config, FakeLLM(), FakeExecutor())

    assert session._max_rounds == 2
    assert session._spoken_numbers is True
    assert session._round_cap_text == "cap"
    assert session._failure_text == "fail"
    # `no_answer` was left null, so the built-in English constant survives.
    assert session._no_answer_text.startswith("I am sorry, I did not catch that")


# --------------------------------------------------------------------------------------
# Project and provider lifecycle
# --------------------------------------------------------------------------------------


async def test_open_project_is_idempotent_and_checked() -> None:
    provider = BusinessProvider(llm=FakeLLM(), executor=FakeExecutor())
    project = await provider.open_project(project_record())
    assert await provider.open_project(project_record()) is project

    with pytest.raises(ValueError, match="does not match"):
        await provider.open_project(
            ProjectRecord(
                project_id="other",
                owner_id="owner_1",
                label="other",
                provider_name="codex-app-server",
            )
        )
    await provider.close()


async def test_close_leaves_injected_collaborators_alone() -> None:
    executor = FakeExecutor()
    provider, run, llm, _ = await open_run([text_response("Hi.")], executor=executor)
    await collect(run, 1)

    await provider.close()

    assert run.closed is True
    # The provider did not build them, so it must not close them either.
    assert llm.closed is False
    assert executor.closed is False


# --------------------------------------------------------------------------------------
# Main lane: start, share, result
# --------------------------------------------------------------------------------------


async def test_start_maps_share_and_result() -> None:
    executor = FakeExecutor(
        {
            "query_order": ToolResult(
                name="query_order",
                ok=True,
                data={"order_id": "202609170001234", "status": "shipped"},
            )
        }
    )
    provider, run, llm, _ = await open_run(
        [
            tool_call_response(
                "query_order",
                {"order_id": "202609170001234"},
                text="Let me check that order.",
            ),
            text_response("Your order has shipped."),
        ],
        executor=executor,
    )

    share, result = await collect(run, 2)

    assert share.kind == "share"
    assert share.generation == 1
    assert share.share is not None
    assert share.share.kind == "important"
    assert share.share.text == "Let me check that order."
    # The ids live on the TaskRequest; BrainSession never sees them (DESIGN.md 6.2).
    assert (share.share.task_id, share.share.session_id) == (TASK_ID, SESSION_ID)

    assert result.kind == "result"
    assert result.result is not None
    assert result.result.status == "completed"
    assert result.result.full_result == "Your order has shipped."
    assert (result.result.task_id, result.result.session_id) == (TASK_ID, SESSION_ID)
    assert llm.call_count == 2
    await provider.close()


async def test_failed_result_maps_to_status_failed() -> None:
    config = BrainConfig(max_rounds=1)
    provider, run, _, _ = await open_run(
        [tool_call_response("query_order", {"order_id": "1"})],
        config=config,
    )

    (result,) = await collect(run, 1)

    assert result.kind == "result"
    assert result.result is not None
    assert result.result.status == "failed"
    await provider.close()


async def test_update_continues_the_same_conversation() -> None:
    provider, run, llm, _ = await open_run(
        [text_response("Which order do you mean?"), text_response("It ships tomorrow.")],
    )
    stream = run.events()
    (first,) = await take(stream, 1)
    assert first.result is not None
    assert first.result.full_result == "Which order do you mean?"

    await run.steer(
        TaskUpdate(
            mode="additive",
            event=ContextEvent(SESSION_ID, 2, "user", "user_text", "Order 12345."),
            instruction="Order 12345.",
        )
    )
    (second,) = await take(stream, 1)

    assert second.result is not None
    assert second.result.full_result == "It ships tomorrow."
    # The second completion saw the whole conversation, not just the new turn.
    assert len(llm.calls[1].messages) == 3
    await provider.close()


async def test_update_falls_back_to_the_context_event_text() -> None:
    provider, run, llm, _ = await open_run(
        [text_response("One."), text_response("Two.")],
    )
    stream = run.events()
    await take(stream, 1)

    await run.steer(
        TaskUpdate(
            mode="additive",
            event=ContextEvent(SESSION_ID, 2, "user", "audio_transcript", "  spoken turn  "),
        )
    )
    await take(stream, 1)

    assert llm.calls[1].messages[-1].content[0].text == "spoken turn"
    await provider.close()


async def test_cancel_update_closes_the_run() -> None:
    provider, run, _, _ = await open_run([text_response("Hi.")])
    await collect(run, 1)

    await run.steer(
        TaskUpdate(
            mode="cancel",
            event=ContextEvent(SESSION_ID, 2, "user", "user_text", "stop"),
        )
    )

    assert run.closed is True
    await provider.close()


# --------------------------------------------------------------------------------------
# Fork lane: a TaskQuery never ends the task
# --------------------------------------------------------------------------------------


async def test_query_answers_as_a_share_not_a_result() -> None:
    provider, run, _, _ = await open_run(
        [text_response("Your order has shipped."), text_response("It was placed on Monday.")],
    )
    stream = run.events()
    await take(stream, 1)

    query = TaskQuery(
        task_id=TASK_ID,
        session_id=SESSION_ID,
        generation=1,
        question="When was it placed?",
        context=ContextSnapshot(SESSION_ID, ()),
    )
    await run.query(query)
    (answer,) = await take(stream, 1)

    assert answer.kind == "share"
    assert answer.share is not None
    assert answer.share.kind == "answer"
    assert answer.share.text == "It was placed on Monday."
    assert answer.share.state_patch["_gander"]["lane"] == "query"
    assert answer.share.state_patch["_gander"]["request_id"] == query.request_id
    # The fork ran on a copy: the main conversation did not grow by the question.
    assert all("When was it placed?" not in str(message) for message in run.session.messages)
    await provider.close()


async def test_query_rejects_another_task_and_a_stale_generation() -> None:
    provider, run, _, _ = await open_run([text_response("Hi.")])
    await collect(run, 1)

    with pytest.raises(ValueError, match="another task"):
        await run.query(
            TaskQuery(
                task_id="other",
                session_id=SESSION_ID,
                generation=1,
                question="?",
                context=ContextSnapshot(SESSION_ID, ()),
            )
        )
    with pytest.raises(ValueError, match="stale generation"):
        await run.query(
            TaskQuery(
                task_id=TASK_ID,
                session_id=SESSION_ID,
                generation=2,
                question="?",
                context=ContextSnapshot(SESSION_ID, ()),
            )
        )
    await provider.close()


# --------------------------------------------------------------------------------------
# transfer_to_human: interaction -> respond -> resolve
# --------------------------------------------------------------------------------------


async def test_transfer_travels_as_an_interaction_and_resolves() -> None:
    provider, run, llm, executor = await open_run(
        [
            tool_call_response(
                TRANSFER_TO_HUMAN_NAME,
                {"department": "after_sales", "reason": "The customer asked for a person."},
                call_id="call_transfer",
            ),
            text_response("You are being connected now."),
        ],
    )
    stream = run.events()
    (interaction_event,) = await take(stream, 1)

    assert interaction_event.kind == "interaction"
    interaction = interaction_event.interaction
    assert interaction is not None
    # Upstream has no `client_tool` kind, and this is an action, not an authorization.
    assert interaction.kind == "user_input"
    assert interaction.prompt == "The customer asked for a person."
    assert interaction.metadata["tool"] == TRANSFER_TO_HUMAN_NAME
    assert interaction.metadata["call_id"] == "call_transfer"
    assert interaction.metadata["arguments"] == {
        "department": "after_sales",
        "reason": "The customer asked for a person.",
    }
    assert interaction.choices == ()
    assert interaction.questions == ()
    assert run.pending_interactions == (interaction.interaction_id,)
    # The transfer is never executed inside the Brain.
    assert executor.calls == [
        (TRANSFER_TO_HUMAN_NAME, {"department": "after_sales", "reason": interaction.prompt})
    ]

    accepted = await run.respond(
        TaskInteractionReply(
            task_id=TASK_ID,
            session_id=SESSION_ID,
            generation=1,
            interaction_id=interaction.interaction_id,
            text='{"status": "transferred"}',
        )
    )
    assert accepted is True

    resolved, result = await take(stream, 2)
    assert resolved.kind == "interaction_resolved"
    assert resolved.interaction_id == interaction.interaction_id
    assert resolved.interaction_resolution == "response"
    assert result.kind == "result"
    assert result.result is not None
    assert result.result.full_result == "You are being connected now."
    # The client output went back into the loop as the tool result for that call.
    assert '{"status": "transferred"}' in str(llm.calls[1].messages[-1])
    assert run.pending_interactions == ()
    await provider.close()


async def test_transfer_without_a_reason_uses_the_default_prompt() -> None:
    provider, run, _, _ = await open_run(
        [tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"})],
    )
    (event,) = await collect(run, 1)

    assert event.interaction is not None
    assert event.interaction.prompt == TRANSFER_PROMPT
    await provider.close()


async def test_respond_is_validated() -> None:
    provider, run, _, _ = await open_run(
        [tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"})],
    )
    (event,) = await collect(run, 1)
    assert event.interaction is not None
    interaction_id = event.interaction.interaction_id

    def reply(**overrides: Any) -> TaskInteractionReply:
        fields: dict[str, Any] = {
            "task_id": TASK_ID,
            "session_id": SESSION_ID,
            "generation": 1,
            "interaction_id": interaction_id,
            "text": "done",
        }
        fields.update(overrides)
        return TaskInteractionReply(**fields)

    with pytest.raises(ValueError, match="another task"):
        await run.respond(reply(task_id="other"))
    with pytest.raises(ValueError, match="stale generation"):
        await run.respond(reply(generation=9))
    # An unknown interaction is not an error: a competing client already answered it.
    assert await run.respond(reply(interaction_id=new_id("interaction"))) is False
    await provider.close()


async def test_cancel_reports_the_pending_interaction_as_resolved() -> None:
    provider, run, _, _ = await open_run(
        [tool_call_response(TRANSFER_TO_HUMAN_NAME, {"department": "general"})],
    )
    stream = run.events()
    (event,) = await take(stream, 1)
    assert event.interaction is not None

    await run.cancel()
    (resolved,) = await take(stream, 1)

    assert resolved.kind == "interaction_resolved"
    assert resolved.interaction_id == event.interaction.interaction_id
    assert resolved.interaction_resolution == "turn"
    # The sentinel ends the stream after the run closed.
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(anext(stream), 2.0)
    await provider.close()


async def test_intercept_seam_emits_a_transfer_without_the_llm() -> None:
    """The seam T3.5 uses: no model round, and `respond` closes the task straight away."""
    provider, run, llm, _ = await open_run([], autostart=False)
    stream = run.events()

    interaction_id = await run.emit_client_tool(
        TRANSFER_TO_HUMAN_NAME, {"department": "general", "reason": "转人工"}
    )
    (event,) = await take(stream, 1)
    assert event.kind == "interaction"
    assert event.interaction is not None
    assert event.interaction.metadata["arguments"]["department"] == "general"

    assert await run.respond(
        TaskInteractionReply(
            task_id=TASK_ID,
            session_id=SESSION_ID,
            generation=1,
            interaction_id=interaction_id,
            text='{"status": "transferred"}',
        )
    )
    resolved, result = await take(stream, 2)

    assert resolved.kind == "interaction_resolved"
    assert result.kind == "result"
    assert result.result is not None
    assert result.result.status == "completed"
    assert llm.call_count == 0
    await provider.close()


# --------------------------------------------------------------------------------------
# Failure handling
# --------------------------------------------------------------------------------------


async def test_an_unexpected_session_failure_becomes_an_error_event() -> None:
    class Exploding:
        async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
            raise RuntimeError("boom")

    provider, run, _, _ = await open_run(
        [tool_call_response("query_order", {"order_id": "1"})],
        executor=Exploding(),  # type: ignore[arg-type]
    )

    (event,) = await collect(run, 1)

    assert event.kind == "error"
    assert event.error is not None
    assert "boom" in event.error
    await provider.close()


async def test_a_closed_run_refuses_new_work() -> None:
    provider, run, _, _ = await open_run([text_response("Hi.")])
    await collect(run, 1)
    await run.close()

    with pytest.raises(RuntimeError, match="closed"):
        await run.start()
    await provider.close()


# --------------------------------------------------------------------------------------
# The Gateway-facing path: WorkerProject.start -> WorkerRunChannel -> WorkerEvent
# --------------------------------------------------------------------------------------


def worker_request() -> WorkerRequest:
    return WorkerRequest(
        task_id=TASK_ID,
        run_id="run_1",
        project_id=PROJECT_ID,
        owner_id="owner_1",
        generation=1,
        instruction="Where is my order?",
        context_plan=ContextPlan(),
        policy=WorkerPolicyView(1, (), (), (), (), (), ()),
        lineage_id=SESSION_ID,
    )


def worker_control(request: WorkerRequest) -> WorkerControl:
    # `WorkerControl` only touches the gateway in `record_evidence` and the fetch helpers,
    # none of which the Brain uses: its events carry no artifacts.
    return WorkerControl(None, request.task_id, request.run_id, BUSINESS_CAPABILITIES)  # type: ignore[arg-type]


async def test_worker_run_channel_maps_the_run_to_worker_events() -> None:
    """The run satisfies the duck type `WorkerRunChannel` (`gateway.py:543`) calls."""
    llm = FakeLLM(
        script=[
            tool_call_response("query_order", {"order_id": "1"}, text="Checking that order."),
            text_response("Your order has shipped."),
        ]
    )
    provider = BusinessProvider(llm=llm, executor=FakeExecutor())
    project = await provider.open_project(project_record())
    request = worker_request()

    channel = await project.start(request, worker_control(request))
    stream = channel.events()
    update = await asyncio.wait_for(anext(stream), 2.0)
    done = await asyncio.wait_for(anext(stream), 2.0)

    assert update.type == "update"
    assert update.payload.summary == "Checking that order."
    assert done.type == "done"
    assert done.payload.status == "completed"
    assert done.payload.result == "Your order has shipped."
    # The channel reports the run's `thread_id` as the backend session id.
    assert channel.session_id == SESSION_ID
    await channel.close()
    await provider.close()
