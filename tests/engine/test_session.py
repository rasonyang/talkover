"""Tests for `EngineProtocol`, `EngineStepEvent` and the `EngineSession` threading.

Three layers, all marked:

- `cpu`: a small scriptable fake engine driven through `EngineProtocol`, so the contract
  the realtime layer codes against is exercised without torch. T2.7 writes its own richer
  fake; this one only has to prove the protocol is drivable.
- `cpu`: `EngineSession` itself, with a stub in place of the upstream `DuplexLiveSession`
  (injected through `session_factory`), so the command plumbing, event ordering, error
  propagation and shutdown are covered with no weights loaded.
- `mps`: one real model load from `configs/serve.example.yaml`, fed 5 s of silence.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from talkover.config import TalkoverConfig, load_config, parse_config
from talkover.engine.backend import get_backend
from talkover.engine.protocol import (
    OUTPUT_SAMPLE_RATE,
    UNIT_BYTES,
    UNIT_SAMPLES,
    WORKER_DELIVERY_TOPICS,
    WORKER_DELIVERY_TYPE,
    EngineError,
    EngineProtocol,
    EngineStepEvent,
)
from talkover.engine.session import EngineSession

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_EXAMPLE = REPO_ROOT / "configs" / "serve.example.yaml"

SILENCE_UNIT = b"\x00" * UNIT_BYTES


# --------------------------------------------------------------------------------------
# A scriptable fake engine, driven through EngineProtocol
# --------------------------------------------------------------------------------------


class FakeEngine:
    """A minimal scriptable `EngineProtocol` implementation.

    It replays a scripted list of `EngineStepEvent` values: each `feed_pcm16` /
    `flush_pending` emits the next scripted batch, and every call is recorded in
    :attr:`calls` so a test can assert what the realtime layer asked for.
    """

    def __init__(self, script: Sequence[Sequence[EngineStepEvent]] = ()) -> None:
        self._script = [list(batch) for batch in script]
        self._events: asyncio.Queue[EngineStepEvent | None] = asyncio.Queue()
        self.calls: list[tuple[str, Any]] = []
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        self.calls.append(("start", None))
        self._ready = True

    async def feed_pcm16(self, pcm16_16k_1s: bytes | np.ndarray) -> None:
        size = len(pcm16_16k_1s) if isinstance(pcm16_16k_1s, bytes) else pcm16_16k_1s.size
        if size not in (UNIT_BYTES, UNIT_SAMPLES):
            raise ValueError(f"not one 1 s unit: {size}")
        self.calls.append(("feed_pcm16", size))
        await self._replay()

    async def flush_pending(self) -> None:
        self.calls.append(("flush_pending", None))
        await self._replay()

    async def interrupt_output(self) -> None:
        self.calls.append(("interrupt_output", None))

    async def set_task_slate(self, text: str) -> None:
        self.calls.append(("set_task_slate", text))

    async def submit_text_turn(self, text: str) -> None:
        self.calls.append(("submit_text_turn", text))

    async def feed_tool_response(self, response: Any) -> None:
        self.calls.append(("feed_tool_response", dict(response)))

    async def feed_worker_delivery(self, delivery: Any) -> None:
        self.calls.append(("feed_worker_delivery", dict(delivery)))

    async def events(self) -> AsyncIterator[EngineStepEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item

    async def stop(self) -> None:
        self.calls.append(("stop", None))
        self._ready = False
        await self._events.put(None)

    async def _replay(self) -> None:
        batch = self._script.pop(0) if self._script else []
        for event in batch:
            await self._events.put(event)


def _event(index: int, **overrides: Any) -> EngineStepEvent:
    defaults: dict[str, Any] = {
        "unit_index": index,
        "is_listen": True,
        "text": "",
        "end_of_turn": False,
        "interrupted": False,
    }
    defaults.update(overrides)
    return EngineStepEvent(**defaults)


# --------------------------------------------------------------------------------------
# A stub standing in for the upstream DuplexLiveSession
# --------------------------------------------------------------------------------------


@dataclass
class StubStepEvent:
    """The subset of upstream `DuplexStepEvent` that `from_upstream` reads."""

    index: int
    is_listen: bool = True
    text: str = ""
    end_of_turn: bool = False
    current_time: int | None = None
    audio_waveform: Any | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    generation_id: int = 0
    unit_id: int | None = None
    interrupted: bool = False
    is_tool_call: bool = False
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_error: str | None = None
    tool_response_expected: bool = False


class StubLiveSession:
    """Records upstream calls and returns scripted step events, loading nothing."""

    def __init__(self, *, fail_on: str | None = None, slate_result: bool = True) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.closed = False
        self.fail_on = fail_on
        self.slate_result = slate_result
        self._index = 0
        self.thread_names: list[str] = []

    def _next(self, **overrides: Any) -> StubStepEvent:
        self._index += 1
        return StubStepEvent(index=self._index, **overrides)

    def _record(self, name: str, payload: Any = None) -> None:
        import threading

        self.thread_names.append(threading.current_thread().name)
        self.calls.append((name, payload))
        if self.fail_on == name:
            raise RuntimeError(f"stub failure in {name}")

    def feed_pcm16(self, data: bytes) -> list[StubStepEvent]:
        self._record("feed_pcm16", len(data))
        return [self._next(text=f"unit-{self._index + 1}")]

    def flush_pending(self) -> list[StubStepEvent]:
        self._record("flush_pending")
        return [self._next(end_of_turn=True)]

    def interrupt_output(self) -> None:
        self._record("interrupt_output")

    def set_task_slate(self, slate: str) -> bool:
        self._record("set_task_slate", slate)
        return self.slate_result

    def feed_runtime_event(self, event: Any) -> None:
        self._record("feed_runtime_event", event)

    def feed_tool_response(self, response: Any) -> None:
        self._record("feed_tool_response", response)

    def close(self, **kwargs: Any) -> None:
        self.closed = True
        self.calls.append(("close", kwargs))


def _cpu_config(**engine_overrides: Any) -> TalkoverConfig:
    engine: dict[str, Any] = {"device": "cpu"}
    engine.update(engine_overrides)
    return parse_config({"engine": engine})


async def _session(**kwargs: Any) -> tuple[EngineSession, StubLiveSession]:
    stub = kwargs.pop("stub", None) or StubLiveSession()
    config = kwargs.pop("config", None) or _cpu_config()
    session = EngineSession(
        config,
        backend=get_backend("cpu"),
        session_factory=lambda: stub,
        **kwargs,
    )
    await session.start()
    return session, stub


async def _drain(session: EngineSession, count: int) -> list[EngineStepEvent]:
    """Take exactly `count` events off the session's stream."""
    stream = session.events()
    try:
        return [await anext(stream) for _ in range(count)]
    finally:
        await stream.aclose()


# --------------------------------------------------------------------------------------
# EngineStepEvent and the protocol surface
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_unit_constants_describe_one_second_of_16k_pcm16() -> None:
    assert UNIT_SAMPLES == 16000
    assert UNIT_BYTES == 32000
    assert OUTPUT_SAMPLE_RATE == 24000


@pytest.mark.cpu
def test_step_event_defaults_and_has_audio() -> None:
    silent = _event(1)
    assert silent.audio_waveform is None
    assert silent.has_audio is False
    assert silent.sample_rate == OUTPUT_SAMPLE_RATE
    assert silent.metrics == {}
    assert silent.step_wall_time_sec == 0.0

    spoken = _event(2, is_listen=False, audio_waveform=np.zeros(240, dtype=np.float32))
    assert spoken.has_audio is True


@pytest.mark.cpu
def test_protocol_is_import_light() -> None:
    """`protocol.py` must not drag torch, mcpmft or the session module into the caller.

    The realtime layer and its fake engine import it, and neither may pay for a model
    stack just to name a type. Checked in a subprocess, since this one has torch loaded.
    """
    script = (
        "import sys; import talkover.engine.protocol; "
        "print(sorted(name for name in sys.modules "
        "if name in {'torch', 'mcpmft', 'talkover.engine.session'}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


@pytest.mark.cpu
def test_fake_and_real_engine_satisfy_the_protocol() -> None:
    assert isinstance(FakeEngine(), EngineProtocol)
    assert isinstance(EngineSession(_cpu_config(), backend=get_backend("cpu")), EngineProtocol)


# --------------------------------------------------------------------------------------
# The fake engine, driven through EngineProtocol
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
async def test_fake_engine_drives_the_whole_protocol() -> None:
    speaking = _event(1, is_listen=False, text="hello")
    done = _event(2, is_listen=False, end_of_turn=True)
    engine: EngineProtocol = FakeEngine([[speaking], [done]])

    await engine.start()
    assert engine.ready is True
    await engine.feed_pcm16(SILENCE_UNIT)
    await engine.flush_pending()
    await engine.set_task_slate("be brief")
    await engine.submit_text_turn("order 8642")
    await engine.feed_tool_response({"status": "ok", "task_ids": ["task_1"]})
    await engine.feed_worker_delivery(
        {
            "task_name": "order status",
            "topic": "final",
            "content": "it shipped",
            "status": "completed",
        }
    )
    await engine.interrupt_output()
    await engine.stop()

    assert [event async for event in engine.events()] == [speaking, done]
    assert engine.ready is False
    assert [name for name, _ in engine.calls] == [  # type: ignore[attr-defined]
        "start",
        "feed_pcm16",
        "flush_pending",
        "set_task_slate",
        "submit_text_turn",
        "feed_tool_response",
        "feed_worker_delivery",
        "interrupt_output",
        "stop",
    ]


@pytest.mark.cpu
async def test_fake_engine_rejects_a_wrong_size_unit() -> None:
    engine = FakeEngine()
    await engine.start()
    with pytest.raises(ValueError):
        await engine.feed_pcm16(b"\x00" * 64)


# --------------------------------------------------------------------------------------
# EngineSession: command plumbing against the stub
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
async def test_start_loads_on_the_inference_thread_and_sets_ready() -> None:
    session, stub = await _session()
    try:
        assert session.ready is True
        await session.feed_pcm16(SILENCE_UNIT)
        assert stub.thread_names == ["talkover-engine"]
    finally:
        await session.stop()
    assert session.ready is False
    assert stub.closed is True


@pytest.mark.cpu
async def test_start_is_idempotent() -> None:
    session, _ = await _session()
    try:
        await session.start()
        await session.start()
    finally:
        await session.stop()


@pytest.mark.cpu
async def test_feed_then_events_keeps_submission_order() -> None:
    session, stub = await _session()
    try:
        for _ in range(3):
            await session.feed_pcm16(SILENCE_UNIT)
        events = await _drain(session, 3)
    finally:
        await session.stop()

    assert [event.unit_index for event in events] == [1, 2, 3]
    assert [name for name, _ in stub.calls[:3]] == ["feed_pcm16"] * 3
    assert stub.calls[0][1] == UNIT_BYTES


@pytest.mark.cpu
async def test_feed_accepts_int16_and_float32_arrays() -> None:
    session, stub = await _session()
    try:
        await session.feed_pcm16(np.zeros(UNIT_SAMPLES, dtype=np.int16))
        await session.feed_pcm16(np.zeros(UNIT_SAMPLES, dtype=np.float32))
    finally:
        await session.stop()
    assert [payload for name, payload in stub.calls if name == "feed_pcm16"] == [UNIT_BYTES] * 2


@pytest.mark.cpu
@pytest.mark.parametrize(
    "unit",
    [
        b"\x00" * (UNIT_BYTES - 2),
        b"\x00" * (UNIT_BYTES + 2),
        np.zeros(UNIT_SAMPLES - 1, dtype=np.int16),
        np.zeros((2, UNIT_SAMPLES), dtype=np.int16),
    ],
)
async def test_feed_rejects_a_wrong_size_unit(unit: Any) -> None:
    session, stub = await _session()
    try:
        with pytest.raises(ValueError):
            await session.feed_pcm16(unit)
        assert stub.calls == []
        assert session.ready is True
    finally:
        await session.stop()


@pytest.mark.cpu
async def test_feed_rejects_a_bad_type_and_a_bad_dtype() -> None:
    session, _ = await _session()
    try:
        with pytest.raises(TypeError):
            await session.feed_pcm16([0] * UNIT_SAMPLES)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="dtype"):
            await session.feed_pcm16(np.zeros(UNIT_SAMPLES, dtype=np.int32))
    finally:
        await session.stop()


@pytest.mark.cpu
async def test_flush_interrupt_slate_and_text_turn_reach_upstream() -> None:
    session, stub = await _session()
    try:
        await session.flush_pending()
        await session.interrupt_output()
        await session.set_task_slate("the caller asks about order 8642")
        await session.submit_text_turn("hello")
        events = await _drain(session, 1)
    finally:
        await session.stop()

    assert events[0].end_of_turn is True
    assert stub.calls[:4] == [
        ("flush_pending", None),
        ("interrupt_output", None),
        ("set_task_slate", "the caller asks about order 8642"),
        ("feed_runtime_event", {"type": "user_text", "content": "hello"}),
    ]


@pytest.mark.cpu
async def test_the_cerebellum_channel_reaches_upstream() -> None:
    """T1.4: both channel-back calls are ordinary commands on the inference thread."""
    session, stub = await _session()
    try:
        await session.feed_tool_response({"status": "ok", "task_ids": ["task_1"]})
        await session.feed_worker_delivery(
            {"task_name": "order status", "topic": "milestone", "content": "checking now"}
        )
    finally:
        await session.stop()

    assert stub.calls[:2] == [
        ("feed_tool_response", {"status": "ok", "task_ids": ["task_1"]}),
        (
            "feed_runtime_event",
            {
                "type": WORKER_DELIVERY_TYPE,
                "task_name": "order status",
                "topic": "milestone",
                "content": "checking now",
            },
        ),
    ]
    # Both ran on the inference thread, not on the event loop.
    assert all(name != "MainThread" for name in stub.thread_names)


@pytest.mark.cpu
async def test_a_worker_delivery_keeps_a_type_it_already_carries() -> None:
    session, stub = await _session()
    try:
        await session.feed_worker_delivery(
            {"type": WORKER_DELIVERY_TYPE, "task_name": "t", "topic": "risk", "content": "careful"}
        )
        with pytest.raises(ValueError, match="worker_delivery"):
            await session.feed_worker_delivery({"type": "user_text", "content": "no"})
    finally:
        await session.stop()

    assert [name for name, _ in stub.calls] == ["feed_runtime_event", "close"]


@pytest.mark.cpu
def test_the_delivery_topics_are_the_upstream_set() -> None:
    """The model was trained on `DeliveryRecord.topic`; nothing else may be sent."""
    assert WORKER_DELIVERY_TOPICS == {"milestone", "interaction", "final", "risk", "aggregate"}


@pytest.mark.cpu
async def test_a_refused_channel_payload_does_not_end_the_session() -> None:
    """Upstream refusing a tool response is the bridge's problem, not the call's."""
    session, stub = await _session(stub=StubLiveSession(fail_on="feed_tool_response"))
    try:
        with pytest.raises(EngineError, match="refused"):
            await session.feed_tool_response({"status": "ok", "task_ids": []})
        # The session is untouched: the next unit still runs.
        assert session.ready is True
        await session.feed_pcm16(SILENCE_UNIT)
        events = await _drain(session, 1)
    finally:
        await session.stop()

    assert events[0].unit_index == 1
    assert [name for name, _ in stub.calls[:2]] == ["feed_tool_response", "feed_pcm16"]


@pytest.mark.cpu
async def test_a_refused_task_slate_is_not_an_error() -> None:
    session, _ = await _session(stub=StubLiveSession(slate_result=False))
    try:
        await session.set_task_slate("ignored")
        assert session.ready is True
    finally:
        await session.stop()


@pytest.mark.cpu
async def test_step_wall_time_is_recorded() -> None:
    session, _ = await _session()
    try:
        await session.feed_pcm16(SILENCE_UNIT)
        events = await _drain(session, 1)
    finally:
        await session.stop()
    assert events[0].step_wall_time_sec > 0.0


@pytest.mark.cpu
async def test_stop_is_idempotent_and_closes_upstream_once() -> None:
    session, stub = await _session()
    await session.stop()
    await session.stop()
    assert [name for name, _ in stub.calls].count("close") == 1
    with pytest.raises(EngineError, match="stopped"):
        await session.feed_pcm16(SILENCE_UNIT)


@pytest.mark.cpu
async def test_events_ends_after_stop() -> None:
    session, _ = await _session()
    await session.feed_pcm16(SILENCE_UNIT)
    await session.stop()
    events = [event async for event in session.events()]
    assert len(events) == 1


@pytest.mark.cpu
async def test_a_failed_load_propagates_and_leaves_the_session_unready() -> None:
    def factory() -> StubLiveSession:
        raise RuntimeError("no weights here")

    session = EngineSession(_cpu_config(), backend=get_backend("cpu"), session_factory=factory)
    with pytest.raises(RuntimeError, match="no weights here"):
        await session.start()
    assert session.ready is False
    with pytest.raises(EngineError, match="failed"):
        await session.feed_pcm16(SILENCE_UNIT)
    await session.stop()


@pytest.mark.cpu
async def test_a_step_failure_surfaces_from_the_call_and_from_events() -> None:
    session, stub = await _session(stub=StubLiveSession(fail_on="feed_pcm16"))
    try:
        with pytest.raises(RuntimeError, match="stub failure in feed_pcm16"):
            await session.feed_pcm16(SILENCE_UNIT)
        assert session.ready is False
        with pytest.raises(RuntimeError, match="stub failure in feed_pcm16"):
            async for _ in session.events():
                pass
        with pytest.raises(EngineError, match="failed"):
            await session.flush_pending()
    finally:
        await session.stop()
    assert stub.closed is True


@pytest.mark.cpu
async def test_commands_queued_behind_a_failure_are_failed_not_hung() -> None:
    stub = StubLiveSession(fail_on="flush_pending")
    session, _ = await _session(stub=stub)
    try:
        failing = asyncio.ensure_future(session.flush_pending())
        queued = asyncio.ensure_future(session._submit("interrupt"))
        with pytest.raises(RuntimeError, match="stub failure"):
            await failing
        with pytest.raises(EngineError):
            await asyncio.wait_for(queued, timeout=5.0)
    finally:
        await session.stop()


@pytest.mark.cpu
async def test_calls_before_start_are_refused() -> None:
    session = EngineSession(
        _cpu_config(), backend=get_backend("cpu"), session_factory=StubLiveSession
    )
    with pytest.raises(EngineError, match="not been started"):
        await session.feed_pcm16(SILENCE_UNIT)
    with pytest.raises(EngineError, match="not been started"):
        await anext(session.events())


# --------------------------------------------------------------------------------------
# from_upstream
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_from_upstream_maps_every_field() -> None:
    upstream = StubStepEvent(
        index=7,
        is_listen=False,
        text="hello",
        end_of_turn=True,
        current_time=7000,
        audio_waveform=np.zeros(240, dtype=np.float64),
        metrics={"cost_llm": 0.21},
        generation_id=3,
        unit_id=2,
        interrupted=True,
        is_tool_call=True,
        tool_calls=[{"name": "query_order"}],
        tool_error=None,
        tool_response_expected=True,
    )
    event = EngineSession.from_upstream(upstream, step_wall_time_sec=0.5)

    assert event.unit_index == 7
    assert event.is_listen is False
    assert event.text == "hello"
    assert event.end_of_turn is True
    assert event.interrupted is True
    assert event.current_time == 7000
    assert event.unit_id == 2
    assert event.generation_id == 3
    assert event.is_tool_call is True
    assert event.tool_calls == ({"name": "query_order"},)
    assert event.tool_response_expected is True
    assert event.metrics == {"cost_llm": 0.21}
    assert event.step_wall_time_sec == 0.5
    assert event.audio_waveform is not None
    assert event.audio_waveform.dtype == np.float32
    assert event.sample_rate == OUTPUT_SAMPLE_RATE


@pytest.mark.cpu
def test_from_upstream_normalizes_an_empty_waveform_to_none() -> None:
    event = EngineSession.from_upstream(StubStepEvent(index=1, audio_waveform=np.zeros(0)))
    assert event.audio_waveform is None
    assert event.has_audio is False


@pytest.mark.cpu
def test_from_upstream_accepts_a_torch_style_tensor() -> None:
    import torch

    tensor = torch.zeros(120, dtype=torch.float32)
    event = EngineSession.from_upstream(StubStepEvent(index=1, audio_waveform=tensor))
    assert event.audio_waveform is not None
    assert event.audio_waveform.shape == (120,)


# --------------------------------------------------------------------------------------
# Upstream settings built from the typed config
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_model_arguments_come_from_the_config() -> None:
    from talkover.engine.session import build_model_arguments

    config = _cpu_config(
        base_model="/models/base",
        dtype="bfloat16",
        attn_implementation="sdpa",
        media_mode="voice",
        init_vision=True,
    )
    args = build_model_arguments(config, token2wav_dir="/models/base/assets/token2wav")

    assert args.model_name_or_path == "/models/base"
    assert args.torch_dtype == "bfloat16"
    assert args.attn_implementation == "sdpa"
    # voice mode never loads the vision encoder, whatever the config says
    assert args.init_vision is False
    assert args.init_audio is True
    assert args.init_tts is True
    assert args.device_map is None
    assert args.token2wav_dir == "/models/base/assets/token2wav"


@pytest.mark.cpu
def test_duplex_params_and_live_config_come_from_the_config() -> None:
    from talkover.engine.session import build_duplex_params, build_live_config

    config = parse_config(
        {
            "engine": {
                "device": "cpu",
                "context_max_units": 64,
                "sliding_window_mode": "context_slate",
                "media_mode": "voice",
            },
            "realtime": {"memory_slate_max_tokens": 128, "trailing_silence_sec": 4.0},
        }
    )
    params = build_duplex_params(config)
    assert params.generate_audio is True
    assert params.context_max_units == 64
    assert params.sliding_window_mode == "context_slate"
    assert params.memory_slate_max_tokens == 128

    live_config = build_live_config(config)
    assert live_config.media_mode == "voice"
    assert live_config.trailing_silence_sec == 4.0
    assert live_config.stop_on_turn_end is False


@pytest.mark.cpu
def test_video_media_mode_maps_to_upstream_omni() -> None:
    from talkover.engine.session import build_live_config

    config = _cpu_config(media_mode="video", init_vision=True)
    assert build_live_config(config).media_mode == "omni"


@pytest.mark.cpu
def test_default_token2wav_dir_is_inside_the_base_model(tmp_path: Path) -> None:
    from talkover.engine.session import default_token2wav_dir

    assert default_token2wav_dir(_cpu_config(base_model=str(tmp_path))) is None
    (tmp_path / "assets" / "token2wav").mkdir(parents=True)
    assert default_token2wav_dir(_cpu_config(base_model=str(tmp_path))) == str(
        tmp_path / "assets" / "token2wav"
    )


@pytest.mark.cpu
def test_configured_token2wav_dir_wins_over_the_base_model(tmp_path: Path) -> None:
    from talkover.engine.session import default_token2wav_dir

    base = tmp_path / "base"
    (base / "assets" / "token2wav").mkdir(parents=True)
    elsewhere = tmp_path / "voices" / "token2wav"
    config = _cpu_config(base_model=str(base), token2wav_dir=str(elsewhere))

    # A configured directory that does not exist is "no assets", not the base model one.
    assert default_token2wav_dir(config) is None

    elsewhere.mkdir(parents=True)
    assert default_token2wav_dir(config) == str(elsewhere)


@pytest.mark.cpu
async def test_engine_asset_paths_come_from_the_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkover.engine.session as session_mod

    captured: list[dict[str, Any]] = []

    def fake_build(config: TalkoverConfig, backend: Any, **kwargs: Any) -> StubLiveSession:
        captured.append(kwargs)
        return StubLiveSession()

    monkeypatch.setattr(session_mod, "build_live_session", fake_build)
    config = _cpu_config(
        base_model=str(tmp_path),
        token2wav_dir=str(tmp_path / "token2wav"),
        ref_audio_path="~/voices/agent.wav",
    )

    session = EngineSession(config, backend=get_backend("cpu"))
    await session.start()
    await session.stop()

    assert captured[0]["token2wav_dir"] is None  # build_live_session resolves it
    assert captured[0]["ref_audio_path"] == str(Path("~/voices/agent.wav").expanduser())


@pytest.mark.cpu
async def test_constructor_arguments_override_the_configured_asset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import talkover.engine.session as session_mod

    captured: list[dict[str, Any]] = []

    def fake_build(config: TalkoverConfig, backend: Any, **kwargs: Any) -> StubLiveSession:
        captured.append(kwargs)
        return StubLiveSession()

    monkeypatch.setattr(session_mod, "build_live_session", fake_build)
    config = _cpu_config(
        base_model=str(tmp_path),
        token2wav_dir=str(tmp_path / "configured"),
        ref_audio_path=str(tmp_path / "configured.wav"),
    )

    session = EngineSession(
        config,
        backend=get_backend("cpu"),
        token2wav_dir="/explicit/token2wav",
        ref_audio_path="/explicit/voice.wav",
    )
    await session.start()
    await session.stop()

    assert captured[0]["token2wav_dir"] == "/explicit/token2wav"
    assert captured[0]["ref_audio_path"] == "/explicit/voice.wav"


# --------------------------------------------------------------------------------------
# The M4 / T1.5 seams
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_a_second_talker_device_is_refused_as_the_m4_branch() -> None:
    config = parse_config({"engine": {"device": "cuda:0", "talker_device": "cuda:1"}})
    with pytest.raises(NotImplementedError, match="M4"):
        EngineSession(config, backend=get_backend("cuda:0"))


@pytest.mark.cpu
def test_the_talker_seam_takes_a_factory() -> None:
    """T1.5 filled the seam: the argument is a factory, and it is optional.

    What the factory does once it is given — its own thread, the ``is_audio_chunk``
    events — is covered by ``tests/engine/test_talker.py``.
    """
    session = EngineSession(_cpu_config(), backend=get_backend("cpu"), talker_factory=None)
    assert session.ready is False


# --------------------------------------------------------------------------------------
# The real model on MPS
# --------------------------------------------------------------------------------------


def _incomplete_reason(path: Path) -> str | None:
    """Why this checkpoint cannot be loaded, or ``None`` when it looks complete.

    A directory that is still downloading holds some of its shards, so the test must look
    at the weight files themselves: loading half a checkpoint fails deep inside upstream
    with an error that says nothing useful.
    """
    if not path.is_dir():
        return "directory is missing"
    index = path / "model.safetensors.index.json"
    if index.is_file():
        import json

        weight_map = json.loads(index.read_text()).get("weight_map", {})
        missing = sorted({name for name in weight_map.values() if not (path / name).is_file()})
        if missing:
            return f"{len(missing)} weight shards are missing ({missing[0]}, ...)"
        return None
    if not any(path.glob("*.safetensors")):
        return "no safetensors weight file"
    return None


def _missing_checkpoints(config: TalkoverConfig) -> list[str]:
    paths = {
        "engine.base_model": config.engine.base_model,
        "engine.thinker_checkpoint": config.engine.thinker_checkpoint,
        "engine.talker_checkpoint": config.engine.talker_checkpoint,
    }
    reasons = []
    for key, value in paths.items():
        reason = _incomplete_reason(Path(value))
        if reason is not None:
            reasons.append(f"{key}={value} ({reason})")
    return reasons


@pytest.mark.mps
async def test_real_model_listens_to_five_seconds_of_silence() -> None:
    backend = get_backend("mps")
    if not backend.is_available():
        pytest.skip("MPS is not available on this machine")
    if not SERVE_EXAMPLE.is_file():
        pytest.skip(f"{SERVE_EXAMPLE} is missing")
    config = load_config(SERVE_EXAMPLE)
    missing = _missing_checkpoints(config)
    if missing:
        pytest.skip(
            "Gander checkpoints are not downloaded; missing "
            + ", ".join(missing)
            + " (run scripts/download_models.sh)"
        )

    session = EngineSession(config, backend=backend)
    load_started = time.perf_counter()
    await session.start()
    load_sec = time.perf_counter() - load_started
    try:
        assert session.ready is True
        for _ in range(5):
            await session.feed_pcm16(SILENCE_UNIT)
        events = await _drain(session, 5)
    finally:
        await session.stop()

    assert len(events) == 5
    assert [event.unit_index for event in events] == [1, 2, 3, 4, 5]
    assert all(event.is_listen for event in events), [event.text for event in events]
    assert all(not event.interrupted for event in events)
    steps = [event.step_wall_time_sec for event in events]
    print(
        f"\nmodel load {load_sec:.1f} s; per-unit steps "
        + ", ".join(f"{value:.3f} s" for value in steps)
    )
