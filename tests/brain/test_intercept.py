"""Tests for the transfer pre-interception rule (T3.5, DESIGN.md 6.4). GPU-free, no network.

Two halves: the matcher on its own (a pure function of the trusted text and the configured
keywords) and the provider's task-start path, where a hit must reach the client without a
single LLM call and within a fraction of the two to three seconds the Brain loop costs.

The keywords are runtime data, so the Chinese phrases from `configs/serve.example.yaml` are
what the cases are written against.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from gander_runtime.contracts import (
    ContextSnapshot,
    ProviderEvent,
    TaskInteractionReply,
    TaskRequest,
    WorkState,
)
from gander_runtime.coordination import ProjectRecord

from talkover.brain.intercept import (
    INTERCEPT_DEPARTMENT,
    TransferInterceptor,
    normalize,
)
from talkover.brain.provider import (
    TRANSFER_PROMPT,
    BusinessProject,
    BusinessProvider,
    BusinessRun,
)
from talkover.brain.tools import TRANSFER_TO_HUMAN_NAME
from talkover.config import BrainConfig

from .fake_llm import FakeLLM, text_response

TASK_ID = "task_1"
SESSION_ID = "lineage_1"
PROJECT_ID = "project_1"

#: The configured keywords, exactly as both example configs write them.
KEYWORDS = ("转人工", "人工客服", "找个人")

#: The latency budget for T3.5: the rule must be decided and the interaction emitted well
#: inside a spoken turn. The whole path is in-process, so this is a generous ceiling.
LATENCY_BUDGET_MS = 50.0


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


class FakeExecutor:
    """A tool runner that must never be reached on the intercepted path."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> Any:
        self.calls.append((name, dict(arguments or {})))
        raise AssertionError(f"unexpected tool execution: {name}")

    async def aclose(self) -> None:
        return None


def task_request(instruction: str) -> TaskRequest:
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


async def open_project(
    *,
    keywords: tuple[str, ...] = KEYWORDS,
    script: list[Any] | None = None,
) -> tuple[BusinessProvider, BusinessProject, FakeLLM]:
    llm = FakeLLM(script=script or [])
    provider = BusinessProvider(
        BrainConfig(transfer_keywords=keywords), llm=llm, executor=FakeExecutor()
    )
    project = await provider.open_project(project_record())
    return provider, project, llm


async def next_event(run: BusinessRun, *, timeout: float = 2.0) -> ProviderEvent:
    return await asyncio.wait_for(anext(run.events()), timeout)


# --------------------------------------------------------------------------------------
# The matcher on its own
# --------------------------------------------------------------------------------------


def test_normalize_drops_whitespace_and_punctuation() -> None:
    assert normalize("  转 人工 ！ ") == "转人工"
    assert normalize("Get me an AGENT, now.") == "getmeanagentnow"


def test_normalize_folds_full_width_onto_ascii() -> None:
    assert normalize("ＡＧＥＮＴ") == normalize("agent")
    assert normalize("转人工！") == normalize("转人工!")


@pytest.mark.parametrize(
    "text",
    [
        "转人工",
        "我要转人工",
        "转人工，谢谢",  # full-width comma
        "转人工, 谢谢",  # ASCII comma and a space
        "转 人 工",  # the ASR split the phrase
        "  转人工。  ",  # leading and trailing whitespace
        "喂？转人工！！！",
        "帮我找个人问一下",
        "我想联系人工客服",
    ],
)
def test_match_hits(text: str) -> None:
    match = TransferInterceptor(KEYWORDS).match(text)
    assert match is not None
    assert match.keyword in KEYWORDS
    assert match.text == text
    assert match.department == INTERCEPT_DEPARTMENT


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "，。！",
        "我的订单到哪了",
        "查一下工单状态",  # contains 工 but not 人工
        "这个人真好",  # contains 人 but not 找个人
    ],
)
def test_match_misses(text: str) -> None:
    assert TransferInterceptor(KEYWORDS).match(text) is None


def test_match_is_case_insensitive_for_ascii_keywords() -> None:
    interceptor = TransferInterceptor(("human agent",))
    assert interceptor.matches("Please get me a HUMAN  Agent.")
    assert interceptor.matches("ＨＵＭＡＮ ＡＧＥＮＴ")
    assert not interceptor.matches("please get me a human being")


def test_match_reports_the_first_configured_keyword() -> None:
    interceptor = TransferInterceptor(("人工客服", "转人工"))
    match = interceptor.match("我要转人工，接人工客服")
    assert match is not None
    # Configuration order is priority order, not position in the text.
    assert match.keyword == "人工客服"


def test_empty_and_punctuation_only_keywords_are_dropped() -> None:
    interceptor = TransferInterceptor(("", "   ", "！！", "转人工"))
    assert interceptor.keywords == ("转人工",)
    assert bool(interceptor) is True
    assert interceptor.match("我的订单到哪了") is None


def test_duplicate_keywords_collapse_after_normalisation() -> None:
    assert TransferInterceptor(("转人工", "转 人工", "转人工！")).keywords == ("转人工",)


def test_no_keywords_matches_nothing() -> None:
    interceptor = TransferInterceptor(())
    assert not interceptor
    assert len(interceptor) == 0
    assert interceptor.match("转人工") is None


def test_from_config_reads_the_brain_section() -> None:
    interceptor = TransferInterceptor.from_config(BrainConfig(transfer_keywords=KEYWORDS))
    assert interceptor.keywords == KEYWORDS
    assert list(interceptor) == list(KEYWORDS)


def test_transfer_arguments_carry_only_the_department() -> None:
    interceptor = TransferInterceptor(KEYWORDS)
    match = interceptor.match("转人工")
    assert interceptor.transfer_arguments(match) == {"department": INTERCEPT_DEPARTMENT}


# --------------------------------------------------------------------------------------
# The provider's task-start path
# --------------------------------------------------------------------------------------


async def test_keyword_hit_emits_the_transfer_without_any_llm_call() -> None:
    provider, project, llm = await open_project()
    try:
        run = await project.open_run(task_request("喂，转人工！"))
        event = await next_event(run)

        assert event.kind == "interaction"
        assert event.interaction is not None
        assert event.interaction.kind == "user_input"
        assert event.interaction.prompt == TRANSFER_PROMPT
        assert event.interaction.metadata["tool"] == TRANSFER_TO_HUMAN_NAME
        assert event.interaction.metadata["arguments"] == {"department": INTERCEPT_DEPARTMENT}
        assert event.interaction.task_id == TASK_ID
        assert event.interaction.session_id == SESSION_ID
        assert event.interaction.generation == 1
        assert run.pending_interactions == (event.interaction.interaction_id,)
        assert run.intercepted is not None
        assert run.intercepted.keyword == "转人工"
        assert llm.call_count == 0
    finally:
        await provider.close()


async def test_keyword_miss_runs_the_normal_loop() -> None:
    provider, project, llm = await open_project(script=[text_response("It ships tomorrow.")])
    try:
        run = await project.open_run(task_request("我的订单到哪了"))
        event = await next_event(run)

        assert event.kind == "result"
        assert run.intercepted is None
        assert llm.call_count == 1
    finally:
        await provider.close()


async def test_intercept_false_forces_the_loop() -> None:
    provider, project, llm = await open_project(script=[text_response("Transferring you now.")])
    try:
        run = await project.open_run(task_request("转人工"), intercept=False)
        event = await next_event(run)

        assert event.kind == "result"
        assert run.intercepted is None
        assert llm.call_count == 1
    finally:
        await provider.close()


async def test_no_configured_keywords_never_intercepts() -> None:
    provider, project, llm = await open_project(
        keywords=(), script=[text_response("Transferring you now.")]
    )
    try:
        run = await project.open_run(task_request("转人工"))
        await next_event(run)

        assert run.intercepted is None
        assert llm.call_count == 1
    finally:
        await provider.close()


async def test_autostart_false_still_skips_the_rule() -> None:
    """The T3.5 seam itself: a run opened without autostart emits nothing on its own."""
    provider, project, llm = await open_project()
    try:
        run = await project.open_run(task_request("转人工"), autostart=False)
        with pytest.raises(TimeoutError):
            await next_event(run, timeout=0.05)
        assert run.intercepted is None
        assert llm.call_count == 0
    finally:
        await provider.close()


async def test_client_output_resolves_the_intercepted_transfer() -> None:
    """The reply path is the same as for an LLM-raised call, minus the loop."""
    provider, project, llm = await open_project()
    try:
        run = await project.open_run(task_request("转人工"))
        stream = run.events()
        first = await asyncio.wait_for(anext(stream), 2.0)
        assert first.interaction is not None

        accepted = await run.respond(
            TaskInteractionReply(
                task_id=TASK_ID,
                session_id=SESSION_ID,
                generation=1,
                interaction_id=first.interaction.interaction_id,
                text='{"status": "transferred"}',
            )
        )
        assert accepted is True

        resolved = await asyncio.wait_for(anext(stream), 2.0)
        assert resolved.kind == "interaction_resolved"
        assert resolved.interaction_id == first.interaction.interaction_id
        assert resolved.interaction_resolution == "response"

        result = await asyncio.wait_for(anext(stream), 2.0)
        assert result.kind == "result"
        assert result.result is not None
        assert result.result.status == "completed"
        assert run.pending_interactions == ()
        assert llm.call_count == 0
    finally:
        await provider.close()


async def test_interception_latency_is_under_the_budget() -> None:
    """Measured from `task_start` to the emitted interaction event (T3.5 acceptance)."""
    provider, project, llm = await open_project()
    try:
        # One warm-up run so the measurement excludes first-call imports and the
        # `BrainSession` construction cost that every run pays.
        warmup = await project.open_run(task_request("转人工"))
        await next_event(warmup)
        await warmup.close()

        started = time.perf_counter()
        run = await project.open_run(task_request("我要转人工，谢谢"))
        event = await next_event(run)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        assert event.kind == "interaction"
        assert llm.call_count == 0
        assert elapsed_ms < LATENCY_BUDGET_MS, f"{elapsed_ms:.3f} ms >= {LATENCY_BUDGET_MS} ms"
    finally:
        await provider.close()
