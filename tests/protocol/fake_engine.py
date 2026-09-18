"""A scriptable `EngineProtocol` implementation for the GPU-free protocol tests.

T2.4 uses it to assert which engine call every client event produces; T2.7 extends it
with the scripted step sequences the ported regression cases replay. It imports only
`talkover.engine.protocol`, which pulls in nothing but numpy, so `tests/protocol` never
loads torch or a model.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from talkover.engine.protocol import (
    UNIT_BYTES,
    UNIT_SAMPLES,
    WORKER_DELIVERY_TYPE,
    EngineError,
    EngineStepEvent,
)

__all__ = ["SCRIPTED_CALLS", "FakeEngine", "step_event"]

#: The calls the positional `script` is consumed by, one batch each.
SCRIPTED_CALLS = ("feed_pcm16", "flush_pending")


def step_event(index: int = 1, **overrides: Any) -> EngineStepEvent:
    """Build an `EngineStepEvent`, defaulting to a listening unit."""
    fields: dict[str, Any] = {
        "unit_index": index,
        "is_listen": True,
        "text": "",
        "end_of_turn": False,
        "interrupted": False,
    }
    fields.update(overrides)
    return EngineStepEvent(**fields)


class FakeEngine:
    """Records every `EngineProtocol` call and replays scripted step events.

    - `calls` is the ordered list of `(method, argument)` pairs the session made.
    - `units` keeps the audio handed to `feed_pcm16`, already framed to one unit.
    - `script` is consumed one batch per `feed_pcm16` / `flush_pending`; `push` queues
      more events at any time and `fail_events` makes `events()` raise.
    - `on_call` is the per-method script T2.7's regression cases drive: a batch is queued
      every time the named method is called, and batches for one method are consumed in
      call order. It covers the calls `script` cannot reach — `interrupt_output` (the unit
      that answers a `response.cancel`), `submit_text_turn`, `set_task_slate` — and lets a
      case attach a whole turn to `flush_pending` alone, so the acknowledgements of a
      `commit` never interleave with the units of a response.
    - `tool_responses` / `deliveries` keep the payloads of `feed_tool_response` and
      `feed_worker_delivery`, the two calls `BrainBridge` answers the Cerebellum with
      (T1.4); they are in `calls` as well, so a test may assert on either.
    - `fail_on` names the methods that raise `EngineError` instead of succeeding.
    - `replay_delay` sleeps before each queued batch, for a case that needs the engine to
      answer after the client has sent its next frame.
    """

    def __init__(
        self,
        script: Sequence[Sequence[EngineStepEvent]] = (),
        *,
        on_call: Mapping[str, Sequence[Sequence[EngineStepEvent]] | Sequence[EngineStepEvent]]
        | None = None,
        fail_on: Iterable[str] = (),
        ready: bool = True,
        replay_delay: float = 0.0,
    ) -> None:
        self._script = [list(batch) for batch in script]
        self._on_call = {name: _batches(value) for name, value in (on_call or {}).items()}
        self._events: asyncio.Queue[EngineStepEvent | BaseException | None] = asyncio.Queue()
        self.calls: list[tuple[str, Any]] = []
        self.units: list[np.ndarray] = []
        #: The payloads of `feed_tool_response`, in call order (T1.4's channel back).
        self.tool_responses: list[dict[str, Any]] = []
        #: The payloads of `feed_worker_delivery`, in call order.
        self.deliveries: list[dict[str, Any]] = []
        self.fail_on = set(fail_on)
        self.replay_delay = replay_delay
        self._ready = ready

    # -- EngineProtocol ----------------------------------------------------

    @property
    def ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        self._record("start", None)
        self._ready = True
        await self._replay("start")

    async def feed_pcm16(self, pcm16_16k_1s: bytes | np.ndarray) -> None:
        size = len(pcm16_16k_1s) if isinstance(pcm16_16k_1s, bytes) else pcm16_16k_1s.size
        if size not in (UNIT_BYTES, UNIT_SAMPLES):
            raise ValueError(f"not one 1 s unit: {size}")
        self._record("feed_pcm16", size)
        self.units.append(np.asarray(pcm16_16k_1s))
        await self._replay("feed_pcm16")

    async def flush_pending(self) -> None:
        self._record("flush_pending", None)
        await self._replay("flush_pending")

    async def interrupt_output(self) -> None:
        self._record("interrupt_output", None)
        await self._replay("interrupt_output")

    async def set_task_slate(self, text: str) -> None:
        self._record("set_task_slate", text)
        await self._replay("set_task_slate")

    async def submit_text_turn(self, text: str) -> None:
        self._record("submit_text_turn", text)
        await self._replay("submit_text_turn")

    async def feed_tool_response(self, response: Mapping[str, Any]) -> None:
        payload = dict(response)
        self._record("feed_tool_response", payload)
        self.tool_responses.append(payload)
        await self._replay("feed_tool_response")

    async def feed_worker_delivery(self, delivery: Mapping[str, Any]) -> None:
        payload = dict(delivery)
        payload.setdefault("type", WORKER_DELIVERY_TYPE)
        self._record("feed_worker_delivery", payload)
        self.deliveries.append(payload)
        await self._replay("feed_worker_delivery")

    async def events(self) -> AsyncIterator[EngineStepEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item

    async def stop(self) -> None:
        self._record("stop", None)
        self._ready = False
        await self._events.put(None)

    # -- test helpers ------------------------------------------------------

    @property
    def call_names(self) -> list[str]:
        """Just the method names, in call order."""
        return [name for name, _ in self.calls]

    def argument(self, name: str) -> Any:
        """The argument of the first call to `name`."""
        for call, argument in self.calls:
            if call == name:
                return argument
        raise AssertionError(f"{name} was never called")

    async def push(self, *events: EngineStepEvent) -> None:
        """Queue step events for `events()` without a `feed_pcm16`."""
        for event in events:
            await self._events.put(event)

    async def fail_events(self, exc: BaseException | None = None) -> None:
        """Make the `events()` iterator raise on its next step."""
        await self._events.put(exc or EngineError("the inference thread died"))

    def script_call(self, name: str, *events: EngineStepEvent) -> None:
        """Append one `on_call` batch for `name`, for a case that scripts mid-test."""
        self._on_call.setdefault(name, []).append(list(events))

    def extend_script(self, *batches: Sequence[EngineStepEvent]) -> None:
        """Append batches to the shared `script` consumed by feed / flush."""
        self._script.extend(list(batch) for batch in batches)

    def _record(self, name: str, argument: Any) -> None:
        if name in self.fail_on:
            raise EngineError(f"{name} failed")
        self.calls.append((name, argument))

    async def _replay(self, name: str) -> None:
        """Queue what this call owes: the shared `script`, then `on_call[name]`."""
        batches: list[list[EngineStepEvent]] = []
        if name in SCRIPTED_CALLS:
            batches.append(self._script.pop(0) if self._script else [])
        queued = self._on_call.get(name)
        if queued:
            batches.append(queued.pop(0))
        for batch in batches:
            if self.replay_delay:
                await asyncio.sleep(self.replay_delay)
            for event in batch:
                await self._events.put(event)


def _batches(
    value: Sequence[Sequence[EngineStepEvent]] | Sequence[EngineStepEvent],
) -> list[list[EngineStepEvent]]:
    """Accept either one batch of events or a list of batches, one per call."""
    items = list(value)
    if items and all(isinstance(item, EngineStepEvent) for item in items):
        return [list(items)]  # type: ignore[arg-type]
    return [list(batch) for batch in items]  # type: ignore[arg-type]
