"""The composed application (T2.9): lifecycle, the ASR side channel and the release path.

GPU-free and network-free, like the rest of `tests/protocol`: the engine is `FakeEngine`,
the Brain runs on `FakeLLM`, and the ASR stream is a stub that replays scripted events, so
nothing here loads torch, a checkpoint or an ASR model.

What the cases fix:

- `build_app` installs a lifespan that starts the engine before the first request and stops
  it on shutdown (`talkover serve` owns nothing else of the lifecycle);
- a WebSocket round trip through the built application reaches the fake engine, so the
  default session factory really is `WebSocketSession` and not `DefaultSession`;
- releasing the single slot **resets** the engine instead of stopping it, so `GET /health`
  goes back to `idle` (DESIGN.md 5.7, `docs/protocol-profile.md` §10.1);
- one ASR stream feeds both consumers, the protocol session and the Brain bridge;
- `talkover serve --help` and the `--check-only` dry run of the CLI wiring (the dry run
  that touches torch lives in `tests/engine/test_serve_cli.py`).
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pytest
from brain.fake_llm import FakeLLM, text_response
from fastapi.testclient import TestClient

from talkover.app import (
    AsrSideChannel,
    TalkoverApp,
    brain_ready,
    build_app,
    reset_engine,
    startup_summary,
)
from talkover.brain.provider import BusinessProvider
from talkover.brain.tools import ToolResult
from talkover.cli import build_parser, main
from talkover.config import (
    AsrConfig,
    BrainConfig,
    EngineConfig,
    LLMConfig,
    RealtimeConfig,
    ServerConfig,
    TalkoverConfig,
)
from talkover.engine.asr import SpeechStarted, Transcript
from talkover.realtime.audio import encode_base64_pcm16
from talkover.realtime.server import DEFAULT_MODEL, default_model

from .fake_engine import FakeEngine

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

#: One second of client audio: 24 kHz pcm16 in, exactly one 16 kHz model unit out.
ONE_SECOND_24K = encode_base64_pcm16(np.zeros(24000, dtype=np.int16))


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class FakeExecutor:
    """Answers every business lookup with a canned success."""

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        return ToolResult(name=name, ok=True, data={"order_id": "A1"})


class StubAsrStream:
    """Stands in for `AsrStream`: replays one scripted batch of events per fed unit."""

    def __init__(self, script: list[list[Any]] | None = None) -> None:
        self.script = list(script or [])
        self.fed: list[Any] = []
        self.flushes = 0
        self.resets = 0

    async def afeed(self, pcm: Any) -> list[Any]:
        self.fed.append(pcm)
        return self.script.pop(0) if self.script else []

    def flush(self) -> list[Any]:
        self.flushes += 1
        return []

    def reset(self) -> None:
        self.resets += 1


class ResettableEngine(FakeEngine):
    """A `FakeEngine` that offers the `reset()` an engine may implement."""

    async def reset(self) -> None:
        self._record("reset", None)


def make_config(**overrides: Any) -> TalkoverConfig:
    """A config whose only non-default fields are the key and whatever a case overrides."""
    fields: dict[str, Any] = {
        "server": ServerConfig(listen="127.0.0.1:8000", api_key=API_KEY),
        # Zero window: a disconnect releases the slot at once, so a case never waits.
        "realtime": RealtimeConfig(trailing_silence_sec=0.0),
    }
    fields.update(overrides)
    return TalkoverConfig(**fields)


def make_provider(llm: FakeLLM | None = None) -> BusinessProvider:
    return BusinessProvider(
        BrainConfig(),
        llm=llm or FakeLLM(script=[text_response("ok")]),
        executor=FakeExecutor(),
    )


def wait_until(predicate: Any, *, timeout: float = 2.0) -> None:
    """Poll from the test thread while the application's own loop runs elsewhere."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("the application never reached the expected state")
        time.sleep(0.005)


def compose(fake_engine: FakeEngine, *, asr: Any = None, **config_overrides: Any) -> TalkoverApp:
    """Build the application on a throwaway loop, the way `talkover serve` does.

    Keyword arguments are `TalkoverConfig` sections, so a case can give the composed
    application its own `engine=` or `brain=` section.
    """
    return asyncio.run(
        build_app(
            make_config(**config_overrides),
            fake_engine,
            provider=make_provider(),
            asr=asr,
        )
    )


def handshake(websocket: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Consume `session.created` and `conversation.created`."""
    return websocket.receive_json(), websocket.receive_json()


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_lifespan_starts_the_engine_and_stops_it_on_shutdown() -> None:
    engine = FakeEngine()
    composed = compose(engine)
    with TestClient(composed.app):
        assert "start" in engine.call_names
        assert "stop" not in engine.call_names
    assert engine.call_names[-1] == "stop"
    assert engine.ready is False


def test_the_application_can_be_built_without_the_engine_lifecycle() -> None:
    """`manage_engine=False` leaves start and stop to the caller (T3.7's callers do)."""
    engine = FakeEngine()
    composed = asyncio.run(
        build_app(make_config(), engine, provider=make_provider(), manage_engine=False)
    )
    with TestClient(composed.app):
        assert engine.call_names == []
    asyncio.run(composed.aclose())


def test_websocket_round_trip_through_the_built_app_reaches_the_fake_engine() -> None:
    """The built application drives `WebSocketSession`, not the placeholder session."""
    engine = FakeEngine()
    composed = compose(engine)
    with (
        TestClient(composed.app) as client,
        client.websocket_connect("/v1/realtime", headers=AUTH) as websocket,
    ):
        created, conversation = handshake(websocket)
        assert created["type"] == "session.created"
        # `conversation.created` is the T2.4 session machine; `DefaultSession` sends
        # only `session.created`, so receiving the pair proves which factory ran.
        assert conversation["type"] == "conversation.created"

        websocket.send_json({"type": "input_audio_buffer.append", "audio": ONE_SECOND_24K})
        websocket.send_json({"type": "input_audio_buffer.commit"})
        assert websocket.receive_json()["type"] == "input_audio_buffer.committed"

    assert engine.call_names[:1] == ["start"]
    assert "feed_pcm16" in engine.call_names
    assert "flush_pending" in engine.call_names
    assert engine.units[0].size == 16000


# ---------------------------------------------------------------------------
# release: reset, never stop
# ---------------------------------------------------------------------------


def test_release_resets_the_engine_and_health_returns_to_idle() -> None:
    engine = FakeEngine()
    composed = compose(engine)
    with TestClient(composed.app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as websocket:
            handshake(websocket)
            assert client.get("/health").json()["status"] == "busy"
        wait_until(lambda: composed.bridge is None)

        assert "stop" not in engine.call_names
        assert "interrupt_output" in engine.call_names
        body = client.get("/health").json()
        assert body["status"] == "idle"
        assert body["engine"] is True
        assert body["session_active"] is False


def test_release_prefers_the_engines_own_reset_when_it_has_one() -> None:
    engine = ResettableEngine()
    composed = compose(engine)
    with TestClient(composed.app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as websocket:
            handshake(websocket)
        wait_until(lambda: composed.bridge is None)
    assert "reset" in engine.call_names
    assert "interrupt_output" not in engine.call_names


async def test_reset_engine_never_raises_when_the_engine_refuses() -> None:
    """A failing reset is logged, not raised: the release path must always finish."""
    engine = FakeEngine(fail_on=["interrupt_output"])
    await reset_engine(engine)


# ---------------------------------------------------------------------------
# the ASR side channel
# ---------------------------------------------------------------------------


async def test_asr_events_reach_every_bound_consumer() -> None:
    transcript = Transcript(text="hello", start_ms=0, end_ms=1000)
    stream = StubAsrStream([[SpeechStarted(at_ms=0), transcript]])
    channel = AsrSideChannel(lambda: stream)
    first: list[Any] = []
    second: list[Any] = []

    async def record_first(event: Any) -> None:
        first.append(event)

    async def record_second(event: Any) -> None:
        second.append(event)

    await channel.start()
    channel.bind(record_first, record_second)
    assert channel.ready is True
    channel.feed(np.zeros(16000, dtype=np.int16))
    while not first or not second:
        await asyncio.sleep(0)

    assert first == [SpeechStarted(at_ms=0), transcript]
    assert second == first
    channel.unbind()
    await channel.aclose()
    assert channel.ready is False


async def test_the_side_channel_drops_the_oldest_unit_when_it_falls_behind() -> None:
    gate = asyncio.Event()

    class BlockedStream(StubAsrStream):
        async def afeed(self, pcm: Any) -> list[Any]:
            await gate.wait()
            return await super().afeed(pcm)

    stream = BlockedStream()
    channel = AsrSideChannel(lambda: stream, queue_units=1)
    await channel.start()
    for _ in range(4):
        channel.feed(np.zeros(16000, dtype=np.int16))
    assert channel.dropped >= 1
    gate.set()
    await channel.aclose()


async def test_a_failing_consumer_does_not_starve_the_other() -> None:
    stream = StubAsrStream([[SpeechStarted(at_ms=0)]])
    channel = AsrSideChannel(lambda: stream)
    seen: list[Any] = []

    async def boom(event: Any) -> None:
        raise RuntimeError("consumer failed")

    async def record(event: Any) -> None:
        seen.append(event)

    await channel.start()
    channel.bind(boom, record)
    channel.feed(np.zeros(16000, dtype=np.int16))
    while not seen:
        await asyncio.sleep(0)
    await channel.aclose()
    assert seen == [SpeechStarted(at_ms=0)]


def test_units_fed_to_the_engine_are_tapped_into_asr_and_reach_both_consumers() -> None:
    transcript = Transcript(text="my order is late", start_ms=0, end_ms=1000)
    stream = StubAsrStream([[transcript]])
    engine = FakeEngine()
    composed = compose(engine, asr=AsrSideChannel(lambda: stream))

    with TestClient(composed.app) as client:
        assert client.get("/health").json()["asr"] is True
        with client.websocket_connect("/v1/realtime", headers=AUTH) as websocket:
            handshake(websocket)
            runner, bridge = composed.runner, composed.bridge
            assert runner is not None and bridge is not None
            # Both halves of DESIGN.md 4.3 are bound to the one stream.
            assert composed.asr is not None
            assert composed.asr.consumers == (runner.session.on_asr_event, bridge.on_asr_event)

            websocket.send_json({"type": "input_audio_buffer.append", "audio": ONE_SECOND_24K})
            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
            # The tap copied the unit the engine was fed, and the transcript came back to
            # the Brain bridge as trusted text.
            wait_until(lambda: bridge.trusted_text == "my order is late")

        wait_until(lambda: composed.bridge is None)
        # Releasing the slot forgets the consumers and the stream's own VAD state.
        assert composed.asr.consumers == ()
        wait_until(lambda: stream.resets == 1)

    assert len(stream.fed) == 1
    assert stream.fed[0].size == 16000
    assert stream.flushes == 1  # the commit closed the utterance


def test_a_failing_asr_backend_does_not_stop_the_service() -> None:
    """ASR is a side channel: it may be missing, and the call still has to run."""

    def explode() -> Any:
        raise RuntimeError("no mlx-whisper on this machine")

    engine = FakeEngine()
    composed = compose(engine, asr=AsrSideChannel(explode))
    with TestClient(composed.app) as client:
        body = client.get("/health").json()
        assert body["status"] == "idle"
        assert body["asr"] is False
        with client.websocket_connect("/v1/realtime", headers=AUTH) as websocket:
            handshake(websocket)
            websocket.send_json({"type": "input_audio_buffer.append", "audio": ONE_SECOND_24K})
            websocket.send_json({"type": "input_audio_buffer.commit"})
            assert websocket.receive_json()["type"] == "input_audio_buffer.committed"
    assert "feed_pcm16" in engine.call_names


# ---------------------------------------------------------------------------
# /health: the Brain probe
# ---------------------------------------------------------------------------


def test_the_brain_probe_is_the_llm_key_and_a_provider() -> None:
    config = make_config(brain=BrainConfig(llm=LLMConfig(api_key="")))
    assert brain_ready(config, None, needs_key=False) is False
    assert brain_ready(config, BusinessProvider(config.brain), needs_key=True) is False
    with_key = make_config(brain=BrainConfig(llm=LLMConfig(api_key="sk-test")))
    assert brain_ready(with_key, BusinessProvider(with_key.brain), needs_key=True) is True
    # An injected provider carries its own client, so no key is needed.
    assert brain_ready(config, make_provider(), needs_key=False) is True


def test_health_reports_a_brain_without_a_key_as_not_ready() -> None:
    config = make_config(brain=BrainConfig(llm=LLMConfig(api_key="")))
    composed = asyncio.run(build_app(config, FakeEngine()))
    with TestClient(composed.app) as client:
        assert client.get("/health").json()["brain"] is False


def test_health_reports_a_configured_brain_as_ready_and_closes_it() -> None:
    config = make_config(brain=BrainConfig(llm=LLMConfig(api_key="sk-test")))
    composed = asyncio.run(build_app(config, FakeEngine()))
    with TestClient(composed.app) as client:
        assert client.get("/health").json()["brain"] is True
    # The lifespan closed the Brain it owns; the probe reports it.
    assert composed.app.state.brain.ready is False


# ---------------------------------------------------------------------------
# the model echo and the startup report
# ---------------------------------------------------------------------------


def test_the_model_echo_defaults_to_the_configured_checkpoint() -> None:
    config = make_config(engine=EngineConfig(base_model="/models/MiniCPM-o-4_5/"))
    assert default_model(config) == "MiniCPM-o-4_5"
    assert default_model(make_config()) == DEFAULT_MODEL

    composed = compose(FakeEngine(), engine=EngineConfig(base_model="/models/MiniCPM-o-4_5"))
    with (
        TestClient(composed.app) as client,
        client.websocket_connect("/v1/realtime", headers=AUTH) as websocket,
    ):
        created, _ = handshake(websocket)
    assert created["session"]["model"] == "MiniCPM-o-4_5"


def test_an_explicit_model_query_still_wins() -> None:
    composed = compose(FakeEngine(), engine=EngineConfig(base_model="/models/MiniCPM-o-4_5"))
    with (
        TestClient(composed.app) as client,
        client.websocket_connect("/v1/realtime?model=gpt-realtime", headers=AUTH) as ws,
    ):
        created, _ = handshake(ws)
    assert created["session"]["model"] == "gpt-realtime"


def test_startup_summary_names_the_device_the_paths_and_the_memory_estimate() -> None:
    config = make_config(
        engine=EngineConfig(
            device="cpu",
            base_model="/models/MiniCPM-o-4_5",
            thinker_checkpoint="/models/Gander/thinker",
            talker_checkpoint="/models/Gander/talker",
        ),
        asr=AsrConfig(backend="faster_whisper", device="cpu", compute_type="int8"),
        brain=BrainConfig(llm=LLMConfig(api_key="")),
    )
    summary = startup_summary(config)
    assert "127.0.0.1:8000" in summary
    assert "cpu (bfloat16)" in summary
    assert "/models/Gander/thinker" in summary
    assert "/models/MiniCPM-o-4_5/assets/token2wav" in summary
    assert "faster_whisper" in summary
    assert "MISSING" in summary  # the unset DEEPSEEK_API_KEY is visible at startup
    assert "memory estimate for device cpu" in summary


# ---------------------------------------------------------------------------
# the CLI surface (the torch-touching dry run is tests/engine/test_serve_cli.py)
# ---------------------------------------------------------------------------


def test_serve_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["serve", "--help"])
    assert excinfo.value.code == 0


def test_serve_accepts_a_config_and_a_check_only_flag() -> None:
    args = build_parser().parse_args(["serve", "-c", "configs/serve.example.yaml", "--check-only"])
    assert (args.config, args.check_only) == ("configs/serve.example.yaml", True)
    assert build_parser().parse_args(["serve"]).check_only is False


def test_serve_reports_a_bad_config_with_exit_code_2(tmp_path: Any, capsys: Any) -> None:
    bad = tmp_path / "broken.yaml"
    bad.write_text("server:\n  listen: not-a-listen\n", encoding="utf-8")
    assert main(["serve", "-c", str(bad)]) == 2
    assert "talkover serve" in capsys.readouterr().err
