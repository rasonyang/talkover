"""Reconnection inside `realtime.trailing_silence_sec` (T2.8).

Three layers, all GPU-free:

- `SessionSlot` with the timer seam replaced by `FakeTimers`, so the reconnect window
  closes when the case says so instead of after eight real seconds.
- `ReconnectSink`, the buffer that keeps the engine's events while no socket is attached.
- the whole application over `TestClient`, with `WebSocketSession` wired in: disconnect,
  reconnect with the same `session_id`, and the release once the window expires.

The transport cases run inside `with TestClient(app)` on purpose: only then do every
WebSocket and every `/health` request share one event loop, which is what keeps a
detached session, its engine pump and its reconnect timer alive between connections.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from talkover.config import RealtimeConfig, ServerConfig, TalkoverConfig
from talkover.realtime import events as ev
from talkover.realtime.server import (
    RESUME_HEADER,
    SessionSlot,
    create_app,
)
from talkover.realtime.session import (
    RealtimeSession,
    ReconnectSink,
    WebSocketSession,
)

from .fake_engine import FakeEngine, step_event

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

#: The events a speaking unit opens a response with (mapping.py section 1).
RESPONSE_OPENING = "response.created"


def make_config(window_sec: float = 8.0) -> TalkoverConfig:
    return TalkoverConfig(
        server=ServerConfig(listen="127.0.0.1:8000", api_key=API_KEY),
        realtime=RealtimeConfig(trailing_silence_sec=window_sec),
    )


async def settle(times: int = 10) -> None:
    """Let the tasks a timer started run to completion."""
    for _ in range(times):
        await asyncio.sleep(0)


def wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0, what: str = "") -> None:
    """Block the test thread until the application thread reaches a state."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError(f"timed out waiting for {what or 'the condition'}")


# ---------------------------------------------------------------------------
# doubles
# ---------------------------------------------------------------------------


class FakeTimers:
    """The `call_later` seam of `create_app` and `SessionSlot`, under the test's control."""

    def __init__(self) -> None:
        self.pending: list[list[Any]] = []

    def call_later(self, delay: float, callback: Callable[[], None]) -> Any:
        entry = [delay, callback]
        self.pending.append(entry)
        return _FakeHandle(self, entry)

    @property
    def delays(self) -> list[float]:
        return [float(delay) for delay, _ in self.pending]

    def fire(self) -> int:
        """Run every pending timer; returns how many fired."""
        due, self.pending = self.pending, []
        for _, callback in due:
            callback()
        return len(due)


class _FakeHandle:
    def __init__(self, timers: FakeTimers, entry: list[Any]) -> None:
        self._timers = timers
        self._entry = entry

    def cancel(self) -> None:
        if self._entry in self._timers.pending:
            self._timers.pending.remove(self._entry)


class FakeRunner:
    """The smallest object `SessionSlot` accepts as a resumable session."""

    def __init__(self, session_id: str = "sess_fake", *, resumable: bool = True) -> None:
        self.session_id = session_id
        self.resumable = resumable
        self.closed = 0
        self.attachments: list[object] = []

    async def run(self) -> None:  # pragma: no cover - the slot never calls it
        raise AssertionError("the slot drives the runner, the test does not")

    async def attach(self, websocket: object) -> None:
        self.attachments.append(websocket)

    async def aclose(self) -> None:
        self.closed += 1


class PlainRunner:
    """A `SessionRunner` that cannot be resumed, like `DefaultSession`."""

    async def run(self) -> None:  # pragma: no cover - the slot never calls it
        raise AssertionError("the slot drives the runner, the test does not")


class FakeSocket:
    """Records what `ReconnectSink` writes, and can fail like a dying socket.

    `receive` never returns, so a `WebSocketSession` driven by one ends only when its
    engine does.
    """

    def __init__(self, *, failing: bool = False) -> None:
        self.sent: list[dict[str, Any]] = []
        self.failing = failing
        self._silent = asyncio.Event()

    async def send_json(self, event: dict[str, Any]) -> None:
        if self.failing:
            raise RuntimeError("the socket is closed")
        self.sent.append(dict(event))

    async def receive(self) -> dict[str, Any]:
        await self._silent.wait()
        raise AssertionError("unreachable: the fake client never speaks")

    @property
    def types(self) -> list[str]:
        return [str(event["type"]) for event in self.sent]


class ListSink:
    """An `EventSink` collecting wire dicts, for the `RealtimeSession` cases."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    async def send(self, event: Any) -> None:
        self.events.append(dict(event))


def make_slot(
    *, window_sec: float = 8.0, on_release: Callable[[], Any] | None = None
) -> tuple[SessionSlot, FakeTimers]:
    timers = FakeTimers()
    slot = SessionSlot(window_sec=window_sec, call_later=timers.call_later, on_release=on_release)
    return slot, timers


# ---------------------------------------------------------------------------
# SessionSlot: the reconnect window
# ---------------------------------------------------------------------------


async def test_detach_keeps_the_slot_busy_for_the_window() -> None:
    slot, timers = make_slot()
    holder, runner = object(), FakeRunner()
    assert slot.acquire(holder) is True
    slot.bind(holder, runner)

    assert await slot.detach(holder) is True

    assert slot.busy is True
    assert slot.attached is False
    assert slot.detached_session_id == "sess_fake"
    assert slot.window_sec == 8.0
    assert timers.delays == [8.0]
    assert runner.closed == 0


async def test_no_other_connection_may_take_the_slot_during_the_window() -> None:
    slot, _ = make_slot()
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)
    await slot.detach(holder)

    other = object()
    assert slot.acquire(other) is False
    assert slot.resume("sess_other", other) is None
    assert slot.resume(None, other) is None
    assert slot.attached is False


async def test_resume_hands_back_the_same_session_and_cancels_the_window() -> None:
    slot, timers = make_slot()
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)
    await slot.detach(holder)

    again = object()
    assert slot.resume("sess_fake", again) is runner
    assert slot.attached is True
    assert slot.detached_session_id is None
    assert timers.pending == []

    # And the session can go around again.
    assert await slot.detach(again) is True
    assert timers.delays == [8.0]


async def test_a_stale_timer_after_a_resume_releases_nothing() -> None:
    slot, timers = make_slot()
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)
    await slot.detach(holder)
    _, fire = timers.pending[0]

    assert slot.resume("sess_fake", object()) is runner
    fire()  # the timer the resume cancelled, fired anyway
    await settle()

    assert runner.closed == 0
    assert slot.busy is True


async def test_the_window_closes_the_session_and_runs_the_release_hook() -> None:
    released: list[str] = []

    async def on_release() -> None:
        released.append("released")

    slot, timers = make_slot(on_release=on_release)
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)
    await slot.detach(holder)

    assert timers.fire() == 1
    await settle()

    assert runner.closed == 1
    assert released == ["released"]
    assert slot.busy is False
    assert slot.detached_session_id is None
    assert slot.acquire(object()) is True


async def test_a_session_that_cannot_be_resumed_is_released_on_disconnect() -> None:
    slot, timers = make_slot()
    holder, runner = object(), PlainRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)

    assert await slot.detach(holder) is False
    assert slot.busy is False
    assert timers.pending == []


async def test_a_zero_window_releases_on_disconnect() -> None:
    slot, timers = make_slot(window_sec=0.0)
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)

    assert await slot.detach(holder) is False
    assert runner.closed == 1
    assert slot.busy is False
    assert timers.pending == []


async def test_a_runner_that_gave_up_gets_no_window() -> None:
    slot, timers = make_slot()
    holder, runner = object(), FakeRunner(resumable=False)
    slot.acquire(holder)
    slot.bind(holder, runner)

    assert await slot.detach(holder) is False
    assert runner.closed == 1
    assert timers.pending == []


async def test_discard_releases_immediately() -> None:
    slot, timers = make_slot()
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)
    await slot.detach(holder)
    assert slot.resume("sess_fake", holder) is runner

    await slot.discard(holder)

    assert runner.closed == 1
    assert slot.busy is False
    assert timers.pending == []


async def test_a_foreign_detach_or_discard_is_a_no_op() -> None:
    slot, _ = make_slot()
    holder, runner = object(), FakeRunner()
    slot.acquire(holder)
    slot.bind(holder, runner)

    assert await slot.detach(object()) is False
    await slot.discard(object())
    slot.bind(object(), FakeRunner("sess_other"))

    assert slot.attached is True
    assert runner.closed == 0


# ---------------------------------------------------------------------------
# ReconnectSink: what happens to the engine's events while nobody is listening
# ---------------------------------------------------------------------------


async def test_the_sink_writes_through_while_attached() -> None:
    socket = FakeSocket()
    sink = ReconnectSink(socket)

    await sink.send({"type": "one"})
    await sink.send({"type": "two"})

    assert socket.types == ["one", "two"]
    assert sink.buffered == 0
    assert sink.attached is True


async def test_the_sink_buffers_while_detached_and_replays_on_attach() -> None:
    socket = FakeSocket()
    sink = ReconnectSink(socket)
    await sink.send({"type": "before"})

    sink.detach()
    await sink.send({"type": "during"})
    assert sink.attached is False
    assert sink.buffered == 1

    resumed = FakeSocket()
    sink.attach(resumed, front=[{"type": "session.created"}, {"type": "conversation.created"}])
    await sink.flush()

    assert socket.types == ["before"]
    assert resumed.types == ["session.created", "conversation.created", "during"]
    assert sink.buffered == 0
    assert sink.dropped == 0


async def test_the_sink_drops_the_oldest_event_when_the_buffer_is_full() -> None:
    sink = ReconnectSink(limit=2)

    for index in range(4):
        await sink.send({"type": f"e{index}"})

    assert sink.buffered == 2
    assert sink.dropped == 2

    socket = FakeSocket()
    sink.attach(socket)
    await sink.flush()
    assert socket.types == ["e2", "e3"]


async def test_a_failed_send_detaches_the_sink_and_keeps_the_event() -> None:
    sink = ReconnectSink(FakeSocket(failing=True))

    await sink.send({"type": "lost-in-transit"})

    assert sink.attached is False
    assert sink.buffered == 1

    resumed = FakeSocket()
    sink.attach(resumed)
    await sink.flush()
    assert resumed.types == ["lost-in-transit"]


async def test_clear_drops_the_backlog() -> None:
    sink = ReconnectSink()
    await sink.send({"type": "gone"})

    sink.clear()

    assert sink.buffered == 0


# ---------------------------------------------------------------------------
# RealtimeSession: the re-announcement
# ---------------------------------------------------------------------------


def make_session(model: str = "gpt-realtime") -> RealtimeSession:
    return RealtimeSession(make_config(), FakeEngine(), sink=ListSink(), model=model)


async def test_reopen_repeats_the_opening_pair_with_the_same_ids() -> None:
    session = make_session()

    produced = session.reopen_events()

    assert [event["type"] for event in produced] == ["session.created", "conversation.created"]
    assert produced[0]["session"]["id"] == session.session_id
    assert produced[0]["session"]["model"] == "gpt-realtime"
    assert produced[1]["conversation"]["id"] == session.conversation_id


async def test_reopen_reports_a_session_the_client_had_changed() -> None:
    session = make_session()
    await session.handle_raw(
        json.dumps(
            {
                "type": "session.update",
                "session": {"type": "realtime", "instructions": "Be brief."},
            }
        )
    )

    produced = session.reopen_events()

    assert [event["type"] for event in produced] == [
        "session.created",
        "conversation.created",
        "session.updated",
    ]
    assert produced[0]["session"]["instructions"] == "Be brief."
    assert produced[2]["session"]["instructions"] == "Be brief."


# ---------------------------------------------------------------------------
# WebSocketSession: what the slot asks of it
# ---------------------------------------------------------------------------


async def test_a_session_stops_being_resumable_once_its_engine_ends() -> None:
    engine = FakeEngine()
    socket = FakeSocket()
    session = WebSocketSession(socket, make_config(), engine)
    running = asyncio.create_task(session.run())
    await settle()

    assert session.resumable is True
    assert socket.types == ["session.created", "conversation.created"]

    await engine.stop()
    await running

    assert session.resumable is False
    await session.aclose()


async def test_a_session_whose_engine_failed_raises_and_is_not_resumable() -> None:
    engine = FakeEngine()
    session = WebSocketSession(FakeSocket(), make_config(), engine)
    running = asyncio.create_task(session.run())
    await settle()

    await engine.fail_events()
    with pytest.raises(ev.ProtocolError) as excinfo:
        await running

    assert excinfo.value.code == "engine_error"
    assert session.resumable is False
    await session.aclose()


async def test_aclose_cancels_the_engine_pump_and_drops_the_backlog() -> None:
    engine = FakeEngine()
    session = WebSocketSession(FakeSocket(), make_config(), engine)
    running = asyncio.create_task(session.run())
    await settle()

    session.sink.detach()
    await engine.push(step_event(1, is_listen=False, text="unheard"))
    await settle()
    assert session.sink.buffered > 0

    await session.aclose()
    assert session.sink.buffered == 0
    assert session.resumable is False
    running.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await running


# ---------------------------------------------------------------------------
# end to end over the transport
# ---------------------------------------------------------------------------


def make_app(
    engine: FakeEngine,
    *,
    window_sec: float = 8.0,
    timers: FakeTimers | None = None,
    on_release: Callable[[], Any] | None = None,
) -> Any:
    return create_app(
        make_config(window_sec),
        engine,
        session_factory=WebSocketSession,
        call_later=None if timers is None else timers.call_later,
        on_release=on_release,
    )


def handshake(websocket: Any) -> tuple[str, str]:
    """Read `session.created` / `conversation.created`; return both ids."""
    created = websocket.receive_json()
    assert created["type"] == "session.created"
    conversation = websocket.receive_json()
    assert conversation["type"] == "conversation.created"
    return created["session"]["id"], conversation["conversation"]["id"]


async def _fire(timers: FakeTimers) -> None:
    """Close every open reconnect window, from inside the application's event loop."""
    timers.fire()
    await settle()


def test_a_reconnection_resumes_the_session_and_replays_what_it_missed() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?model=gpt-realtime", headers=AUTH) as first:
            session_id, conversation_id = handshake(first)
            first.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Be brief."},
                }
            )
            assert first.receive_json()["type"] == "session.updated"

        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")
        assert client.get("/health").json()["status"] == "busy"

        # The model keeps talking while nobody is connected.
        client.portal.call(engine.push, step_event(1, is_listen=False, text="hello"))

        with client.websocket_connect(
            f"/v1/realtime?session_id={session_id}", headers=AUTH
        ) as second:
            again = second.receive_json()
            assert again["type"] == "session.created"
            assert again["session"]["id"] == session_id
            assert again["session"]["model"] == "gpt-realtime"
            assert again["session"]["instructions"] == "Be brief."
            assert second.receive_json()["conversation"]["id"] == conversation_id
            assert second.receive_json()["type"] == "session.updated"

            replayed = [second.receive_json() for _ in range(5)]
            types = [event["type"] for event in replayed]
            assert types[0] == RESPONSE_OPENING
            assert "response.output_audio_transcript.delta" in types

            # The resumed connection drives the same session state machine.
            second.send_json({"type": "input_audio_buffer.clear"})
            assert second.receive_json()["type"] == "input_audio_buffer.cleared"

    assert engine.argument("set_task_slate") == "Be brief."


def test_the_header_names_the_session_to_resume() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")

        headers = {**AUTH, RESUME_HEADER: session_id}
        with client.websocket_connect("/v1/realtime", headers=headers) as second:
            assert second.receive_json()["session"]["id"] == session_id


def test_a_foreign_session_id_inside_the_window_gets_engine_busy() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")

        with client.websocket_connect("/v1/realtime?session_id=sess_other", headers=AUTH) as other:
            event = other.receive_json()
            assert event["type"] == "error"
            assert event["error"]["code"] == "engine_busy"
            closed = other.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 1011

        # The waiting session is untouched and can still be resumed.
        assert slot.detached_session_id == session_id
        with client.websocket_connect(
            f"/v1/realtime?session_id={session_id}", headers=AUTH
        ) as resumed:
            assert resumed.receive_json()["session"]["id"] == session_id


def test_a_connection_without_a_session_id_inside_the_window_gets_engine_busy() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")

        with client.websocket_connect("/v1/realtime", headers=AUTH) as other:
            assert other.receive_json()["error"]["code"] == "engine_busy"
        assert slot.detached_session_id == session_id


def test_the_session_id_of_a_live_session_cannot_be_hijacked() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/realtime", headers=AUTH) as first,
    ):
        session_id, _ = handshake(first)
        with client.websocket_connect(
            f"/v1/realtime?session_id={session_id}", headers=AUTH
        ) as second:
            assert second.receive_json()["error"]["code"] == "engine_busy"

        # The live session is unaffected.
        first.send_json({"type": "input_audio_buffer.clear"})
        assert first.receive_json()["type"] == "input_audio_buffer.cleared"


def test_the_window_releases_the_session_and_health_reports_idle() -> None:
    engine = FakeEngine()
    timers = FakeTimers()
    app = make_app(engine, timers=timers)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?model=gpt-realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
            first.send_json(
                {
                    "type": "session.update",
                    "session": {"type": "realtime", "instructions": "Be brief."},
                }
            )
            assert first.receive_json()["type"] == "session.updated"
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")
        assert timers.delays == [8.0]

        client.portal.call(_fire, timers)
        wait_until(lambda: not slot.busy, what="the slot to be released")

        assert client.get("/health").status_code == 200
        assert client.get("/health").json()["status"] == "idle"

        # A late reconnection is a new session, not an error.
        with client.websocket_connect(
            f"/v1/realtime?session_id={session_id}&model=gpt-realtime", headers=AUTH
        ) as late:
            created = late.receive_json()
            assert created["session"]["id"] != session_id
            assert created["session"]["instructions"] == ""


def test_the_release_hook_runs_when_the_window_expires() -> None:
    engine = FakeEngine()
    timers = FakeTimers()
    released: list[str] = []

    async def on_release() -> None:
        await engine.stop()
        released.append("released")

    app = make_app(engine, timers=timers, on_release=on_release)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")
        assert released == []

        client.portal.call(_fire, timers)
        wait_until(lambda: not slot.busy, what="the slot to be released")

        assert released == ["released"]
        assert engine.call_names[-1] == "stop"
        # The engine the hook stopped is no longer ready, so `/health` says so.
        assert client.get("/health").json()["status"] == "not_ready"


def test_the_default_timer_closes_the_window_without_a_fake_clock() -> None:
    engine = FakeEngine()
    app = make_app(engine, window_sec=0.05)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
            session_id, _ = handshake(first)
        wait_until(lambda: slot.detached_session_id == session_id, what="the session to detach")

        wait_until(lambda: not slot.busy, what="the reconnect window to expire")
        assert client.get("/health").json()["status"] == "idle"


def test_a_session_id_on_a_free_slot_starts_a_fresh_session() -> None:
    engine = FakeEngine()
    app = make_app(engine)
    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/realtime?session_id=sess_unknown", headers=AUTH) as ws,
    ):
        created = ws.receive_json()
        assert created["type"] == "session.created"
        assert created["session"]["id"] != "sess_unknown"
        assert created["session"]["id"].startswith("sess_")


def test_a_session_factory_that_raises_frees_the_slot() -> None:
    def factory(websocket: Any, config: Any, engine: Any, model: str) -> Any:
        raise RuntimeError("the factory failed")

    app = create_app(make_config(), FakeEngine(), session_factory=factory)
    slot = app.state.session_slot
    with TestClient(app) as client:
        try:
            with client.websocket_connect("/v1/realtime", headers=AUTH):
                pass
        except RuntimeError:
            pass
        assert slot.busy is False
        assert client.get("/health").json()["status"] == "idle"


def test_a_fatal_engine_failure_ends_the_session_without_a_window() -> None:
    engine = FakeEngine(fail_on={"submit_text_turn"})
    timers = FakeTimers()
    app = make_app(engine, timers=timers)
    slot = app.state.session_slot
    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime", headers=AUTH) as ws:
            session_id, _ = handshake(ws)
            ws.send_json(
                {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello"}],
                    },
                }
            )
            assert ws.receive_json()["type"] == "conversation.item.created"
            assert ws.receive_json()["error"]["code"] == "engine_error"
            closed = ws.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 1011

        wait_until(lambda: not slot.busy, what="the failed session to be released")
        assert timers.pending == []
        assert client.get("/health").json()["status"] == "idle"

        # The id of the dead session buys nothing.
        with client.websocket_connect(
            f"/v1/realtime?session_id={session_id}", headers=AUTH
        ) as again:
            assert again.receive_json()["session"]["id"] != session_id
