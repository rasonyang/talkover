"""`BrainSession`: the LLM function-calling loop behind one Gander task.

DESIGN.md 6.2 places this module between the Cerebellum-facing `WorkerProvider` (T3.4,
`provider.py`) and the two leaf modules it drives, `llm/` and `tools.py`::

    TaskRequest ──▶ BrainSession.start()   ──▶ Share / ClientTool / Result
    TaskUpdate  ──▶ BrainSession.update()      (main conversation, appended)
    TaskQuery   ──▶ BrainSession.query()       (read-only fork of the conversation)
    function_call_output ──▶ BrainSession.resolve()

This module deliberately does **not** import `gander_runtime`. It yields the neutral
events defined below, and `provider.py` translates them into `ProviderEvent`s. Keeping
the loop upstream-free is what lets `tests/brain` run without a runtime, a model or a
GPU.

Mapping to the upstream contracts (`gander_runtime/contracts.py`), for T3.4:

===================  ===========================================================
Brain event          Upstream translation
===================  ===========================================================
`Share`              `ProviderEvent(kind="share", share=ShareEvent(kind="important", ...))`
`Question`           `ProviderEvent(kind="share", share=ShareEvent(kind="need_input", ...))`
                     for the fork lane, `ShareEvent(kind="answer", ...)` -- that is the
                     kind the Codex provider uses for a side-query answer
                     (`providers/codex.py:1565`)
`ClientTool`         `ProviderEvent(kind="interaction", interaction=TaskInteraction(
                     kind="user_input", ...))`; `transfer_to_human` is executed by the
                     Realtime client, not by the runtime. See the note below.
`Result(ok=True)`    `ProviderEvent(kind="result", result=TaskResult(status="completed"))`
`Result(ok=False)`   `ProviderEvent(kind="result", result=TaskResult(status="failed"))`
===================  ===========================================================

`TaskResult` also needs `task_id` / `session_id` / `generation`, which live on the
`TaskRequest`; this module never sees them, so `provider.py` fills them in.

Note on `ClientTool` (decision, DESIGN.md 6.3): upstream `ProviderEvent.kind`
(`contracts.py:268`) is the closed `Literal`
`share | interaction | interaction_resolved | result | error` — there is no `client_tool`
member and upstream is never forked. The call therefore travels as a `TaskInteraction`
(`contracts.py:211`) with `kind="user_input"`, whose `metadata` carries
`{"tool": "transfer_to_human", "call_id": ..., "arguments": {...}}`, emitted as
`ProviderEvent(kind="interaction")` the way the Codex provider raises a server request
(`providers/codex.py:1362`). The Realtime layer (T3.7) emits the
`response.function_call_arguments.delta` / `.done` pair from that metadata and routes the
client's `function_call_output` back as a `TaskInteractionReply` (`contracts.py:227`)
through the run's `respond()` (`providers/codex.py:983`), which calls
:meth:`BrainSession.resolve`. Upstream `task_resolve` is not that path: its
`TaskResolveAction` (`coordination.py:66`) is an authorization decision
(`cancel | allow_once | allow_session | deny`).

Loop shape (DESIGN.md 6.2, 6.6):

* At most `max_rounds` LLM calls per invocation; exhaustion ends the task with a
  `Result`, never with another round.
* A failed lookup goes back to the model as an error tool result; the system prompt
  tells it to explain the failure and offer a transfer.
* A retryable `LLMError` is retried once; anything else ends the task with a failure
  `Result`.
* `transfer_to_human` is never executed here. The loop emits `ClientTool` and stops
  calling the LLM until :meth:`resolve` supplies the client's output.
"""

from __future__ import annotations

import copy
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from talkover.brain.llm import (
    BrainLLM,
    LLMError,
    LLMResponse,
    Message,
    ToolResultPart,
)
from talkover.brain.spoken import to_spoken
from talkover.brain.tools import (
    BRAIN_TOOLS,
    LOOKUP_TOOLS,
    TRANSFER_TO_HUMAN_NAME,
    ToolResult,
    ToolSpec,
)

__all__ = [
    "BRAIN_SYSTEM_PROMPT",
    "DEFAULT_MAX_ROUNDS",
    "FAILURE_RESULT_TEXT",
    "FORK_SYSTEM_SUFFIX",
    "NO_ANSWER_RESULT_TEXT",
    "ROUND_CAP_RESULT_TEXT",
    "BrainEvent",
    "BrainSession",
    "ClientTool",
    "Question",
    "Result",
    "Share",
    "ToolRunner",
]

#: Six is enough for the two lookups a customer-service turn needs plus a retry, and it
#: bounds the latency budget. `brain.max_rounds` overrides it (DESIGN.md 6.2, section 9).
DEFAULT_MAX_ROUNDS = 6


# --------------------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------------------

# DESIGN.md 5.5 / section 11: `session.instructions` is capped at 256 tokens because it
# can only go into the task slate, so the bulk of the phrasing lives here. This is a
# module constant so that a later config key can override it without touching the loop.
BRAIN_SYSTEM_PROMPT = """\
You are the reasoning brain of a voice customer-service agent. Another model speaks your
words out loud to the customer over a phone line, so everything you write is heard, never
read.

How to answer:
- Answer in the same language the customer is using.
- Use short, plain sentences that are easy to say out loud. No lists, no markdown, no
  emoji, no URLs, no parentheses.
- Say one thing at a time. Two sentences is usually enough.
- Never invent order numbers, ticket numbers, dates, amounts or delivery promises. Every
  fact you state must come from a tool result or from what the customer told you.

Tools:
- query_ticket looks up a support ticket by ticket id, or by the customer's phone number.
- query_order looks up an order by order number, or by the customer's phone number.
- transfer_to_human hands the call to a human agent. It is performed by the telephony
  system, not by you; call it and then stop.

Before calling a lookup tool, ask the customer for the identifier you are missing. If a
tool result reports an error, do not retry it. Tell the customer plainly that the system
cannot be reached right now, and offer to put them through to a human agent. If the
customer asks for a person, call transfer_to_human immediately.

When you have the answer, state it in one or two sentences and stop.\
"""

#: Appended for a fork (`TaskQuery`): a side question must not change the task.
FORK_SYSTEM_SUFFIX = """\

You are answering one side question about the ongoing call. Answer it briefly from what
you already know, or with a read-only lookup. Do not transfer the call, do not promise
any action, and do not treat this question as a new request.\
"""

#: Spoken when the loop runs out of rounds. English by default; `brain.fallback_texts`
#: replaces each of the three (round_cap / failure / no_answer, DESIGN.md 6.2).
ROUND_CAP_RESULT_TEXT = (
    "I am sorry, I could not finish looking this up. Let me put you through to a human agent."
)

#: Spoken when the LLM itself failed (transport error, or a non-retryable error).
FAILURE_RESULT_TEXT = (
    "I am sorry, my system is not responding right now. Let me put you through to a human agent."
)

#: Spoken when the model ended its turn without saying anything.
NO_ANSWER_RESULT_TEXT = "I am sorry, I did not catch that. Could you say it again?"


# --------------------------------------------------------------------------------------
# Neutral events
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Share:
    """An intermediate spoken update, emitted while the loop is still working."""

    text: str


@dataclass(frozen=True, slots=True)
class Question:
    """A question back to the customer; the task stays open until they answer.

    The loop does not emit this yet: on a duplex voice channel the model simply ends its
    turn with the question, which arrives as a `Result`, and the customer's answer comes
    back as a `TaskUpdate`. The type exists so T3.4 can map `ShareEvent(kind="need_input")`
    without a second change here.
    """

    text: str


@dataclass(frozen=True, slots=True)
class ClientTool:
    """A tool the Realtime client executes: today only `transfer_to_human`.

    The loop stops calling the LLM after emitting this and waits for
    :meth:`BrainSession.resolve`.
    """

    call_id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Result:
    """The final answer for this invocation. `ok` False means the task failed."""

    text: str
    ok: bool = True


BrainEvent = Share | Question | ClientTool | Result


# --------------------------------------------------------------------------------------
# Collaborators
# --------------------------------------------------------------------------------------


class ToolRunner(Protocol):
    """The part of `tools.ToolExecutor` the loop uses; tests substitute a fake."""

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult: ...


# --------------------------------------------------------------------------------------
# The session
# --------------------------------------------------------------------------------------


class BrainSession:
    """One Gander task: a conversation, a tool executor and a bounded loop.

    A session is single-use per invocation but long-lived across a task: `start`,
    then any number of `update` / `query` / `resolve` calls. Every one of those is an
    async generator of :data:`BrainEvent`; the caller must drain it.
    """

    def __init__(
        self,
        llm: BrainLLM,
        executor: ToolRunner,
        *,
        system_prompt: str = BRAIN_SYSTEM_PROMPT,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        spoken_numbers: bool = False,
        tools: Sequence[ToolSpec] = BRAIN_TOOLS,
        round_cap_text: str = ROUND_CAP_RESULT_TEXT,
        failure_text: str = FAILURE_RESULT_TEXT,
        no_answer_text: str = NO_ANSWER_RESULT_TEXT,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1")
        self._llm = llm
        self._executor = executor
        self._system_prompt = system_prompt
        self._max_rounds = max_rounds
        self._spoken_numbers = spoken_numbers
        self._tools: tuple[ToolSpec, ...] = tuple(tools)
        # A fork is never offered a tool the client would have to execute.
        self._fork_tools: tuple[ToolSpec, ...] = tuple(
            spec for spec in self._tools if spec in LOOKUP_TOOLS
        )
        self._round_cap_text = round_cap_text
        self._failure_text = failure_text
        self._no_answer_text = no_answer_text
        self._messages: list[Message] = []
        self._pending_call_id: str | None = None
        self._pending_call_name: str | None = None

    # -- introspection ---------------------------------------------------------------

    @property
    def messages(self) -> tuple[Message, ...]:
        """The main conversation. Frozen dataclasses, so this is safe to compare."""
        return tuple(self._messages)

    @property
    def pending_client_tool(self) -> str | None:
        """The call id the loop is waiting on, or None when it is not blocked."""
        return self._pending_call_id

    @property
    def pending_client_tool_name(self) -> str | None:
        """The name of the pending client tool, for the `function_call` item T3.7 emits."""
        return self._pending_call_name

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    # -- entry points ----------------------------------------------------------------

    async def start(self, instruction: str) -> AsyncIterator[BrainEvent]:
        """Run the task named by a `TaskRequest.instruction`."""
        self._messages.append(Message.user(instruction))
        async for event in self._loop(self._messages, fork=False):
            yield event

    async def update(self, instruction: str) -> AsyncIterator[BrainEvent]:
        """Append a `TaskUpdate` (mode `additive`) and continue the same conversation."""
        self._messages.append(Message.user(instruction))
        # A new user turn supersedes whatever the client was asked to execute.
        self._pending_call_id = None
        self._pending_call_name = None
        async for event in self._loop(self._messages, fork=False):
            yield event

    async def query(self, question: str) -> AsyncIterator[BrainEvent]:
        """Answer a `TaskQuery` on a deep copy; the main conversation is untouched."""
        forked = copy.deepcopy(self._messages)
        forked.append(Message.user(question))
        async for event in self._loop(forked, fork=True):
            yield event

    async def resolve(
        self, output: str, *, call_id: str | None = None
    ) -> AsyncIterator[BrainEvent]:
        """Feed a client `function_call_output` back and continue the loop.

        `call_id` is checked against the pending call when given. Resolving when nothing
        is pending raises `RuntimeError`: that is a bug in the caller, not a model error.
        """
        pending = self._pending_call_id
        if pending is None:
            raise RuntimeError("no client tool call is pending")
        if call_id is not None and call_id != pending:
            raise RuntimeError(f"expected output for call {pending!r}, got {call_id!r}")
        self._messages.append(Message.user(ToolResultPart(pending, output)))
        self._pending_call_id = None
        self._pending_call_name = None
        async for event in self._loop(self._messages, fork=False):
            yield event

    async def aclose(self) -> None:
        """Release the LLM client. The executor is owned by the caller."""
        await self._llm.aclose()

    # -- the loop --------------------------------------------------------------------

    async def _loop(self, messages: list[Message], *, fork: bool) -> AsyncIterator[BrainEvent]:
        tools = self._fork_tools if fork else self._tools
        system = self._system_prompt + FORK_SYSTEM_SUFFIX if fork else self._system_prompt
        for _ in range(self._max_rounds):
            try:
                response = await self._complete(messages, tools, system)
            except LLMError:
                yield Result(self._failure_text, ok=False)
                return
            messages.append(response.as_message())
            calls = response.tool_calls
            if not calls:
                text = response.text.strip()
                yield Result(self._say(text) if text else self._no_answer_text, ok=True)
                return
            parts: list[ToolResultPart] = []
            shared = False
            for call in calls:
                result = await self._run_tool(call.name, call.arguments, fork=fork)
                if result.client_tool:
                    # Stop here: the client executes this one and calls resolve().
                    if parts:
                        messages.append(Message("user", tuple(parts)))
                    self._pending_call_id = call.id
                    self._pending_call_name = call.name
                    yield ClientTool(call.id, call.name, dict(result.data))
                    return
                parts.append(ToolResultPart(call.id, result.to_content(), is_error=not result.ok))
                shared = shared or result.ok
            messages.append(Message("user", tuple(parts)))
            # DESIGN.md 6.3: the Brain speaks the lookup back to the customer. The only
            # spoken text available in this round is what the model said alongside its
            # tool calls, so a silent tool round produces no share.
            text = response.text.strip()
            if shared and text:
                yield Share(self._say(text))
        yield Result(self._round_cap_text, ok=False)

    async def _complete(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSpec],
        system: str,
    ) -> LLMResponse:
        """One completion, retried once when the failure is marked retryable."""
        try:
            return await self._llm.complete(messages, tools, system=system)
        except LLMError as first:
            if not first.retryable:
                raise
        return await self._llm.complete(messages, tools, system=system)

    async def _run_tool(self, name: str, arguments: Mapping[str, Any], *, fork: bool) -> ToolResult:
        if fork and name == TRANSFER_TO_HUMAN_NAME:
            # A side question must not transfer the call; report it as a tool error so
            # the model answers the question instead.
            return ToolResult(
                name=name,
                ok=False,
                error_kind="unknown_tool",
                detail="transfer_to_human is not available while answering a side question",
            )
        return await self._executor.execute(name, dict(arguments))

    def _say(self, text: str) -> str:
        """Apply the spoken-number rewrite when the session was built with it on."""
        return to_spoken(text) if self._spoken_numbers else text
