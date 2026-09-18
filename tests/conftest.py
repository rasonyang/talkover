"""Shared fixtures for the GPU-free suites (`tests/protocol`, `tests/brain`).

Three things live here, all of them what T2.7's data-driven regression cases need and
what the earlier protocol suites had to build for themselves:

- `fake_engine`: the scriptable `tests.protocol.fake_engine.FakeEngine`, so a case never
  loads torch or a model.
- `realtime_config` / `realtime_app` / `realtime_client`: the real ASGI application, wired
  with `create_app(..., session_factory=WebSocketSession)` so that a case exercises the
  transport, the session state machine and the response mapper together.
- `realtime_connect`: a context manager yielding a `RealtimeClient`, which authenticates,
  performs the `session.created` / `conversation.created` handshake and offers the
  reading helpers a regression case needs (`send_all`, `receive_many`, `drain`).

Nothing here imports the engine implementation, only `talkover.engine.protocol`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import pytest
from fastapi.testclient import TestClient
from protocol.fake_engine import FakeEngine

from talkover.config import ServerConfig, TalkoverConfig
from talkover.realtime.server import create_app
from talkover.realtime.session import WebSocketSession

__all__ = ["API_KEY", "AUTH_HEADERS", "RealtimeClient"]

#: The single Bearer key every protocol fixture configures.
API_KEY = "test-key"
#: The header a connecting client must present (profile section 1).
AUTH_HEADERS = {"Authorization": f"Bearer {API_KEY}"}

#: Sent by `RealtimeClient.drain` to mark the end of the events a case produced. It is a
#: `session.update` carrying a model name no case uses: it is always accepted, it makes no
#: engine call, and its `session.updated` echo is recognisable, so a case that emits its
#: own `session.updated` or `input_audio_buffer.cleared` cannot be mistaken for the end.
DRAIN_MODEL = "__drain__"
DRAIN_FRAME = {"type": "session.update", "session": {"type": "realtime", "model": DRAIN_MODEL}}


class RealtimeClient:
    """An authenticated `/v1/realtime` WebSocket with the handshake already consumed.

    `drain()` is the workhorse of the data-driven cases: it sends the sentinel frame and
    reads until its echo comes back, so a case collects exactly the events its own frames
    produced, without guessing a count and without a timeout.
    """

    def __init__(self, websocket: Any, engine: FakeEngine) -> None:
        self.websocket = websocket
        self.engine = engine
        #: `session.created` and `conversation.created`, in that order.
        self.handshake: list[dict[str, Any]] = []

    # -- handshake ---------------------------------------------------------

    def open(self) -> list[dict[str, Any]]:
        """Read the two events that precede any client frame (profile section 1)."""
        self.handshake = [self.receive(), self.receive()]
        return self.handshake

    @property
    def session(self) -> dict[str, Any]:
        """The effective session object `session.created` echoed."""
        return dict(self.handshake[0]["session"])

    # -- traffic -----------------------------------------------------------

    def send(self, frame: Mapping[str, Any]) -> None:
        self.websocket.send_json(dict(frame))

    def send_text(self, raw: str) -> None:
        self.websocket.send_text(raw)

    def send_bytes(self, raw: bytes) -> None:
        self.websocket.send_bytes(raw)

    def send_all(self, frames: Sequence[Any]) -> None:
        """Send a case's frames; `bytes` goes out as a binary frame, `str` as text."""
        for frame in frames:
            if isinstance(frame, (bytes, bytearray)):
                self.send_bytes(bytes(frame))
            elif isinstance(frame, str):
                self.send_text(frame)
            else:
                self.send(frame)

    def receive(self) -> dict[str, Any]:
        return dict(self.websocket.receive_json())

    def receive_many(self, count: int) -> list[dict[str, Any]]:
        return [self.receive() for _ in range(count)]

    def receive_types(self, count: int) -> list[str]:
        return [event["type"] for event in self.receive_many(count)]

    def drain(self) -> list[dict[str, Any]]:
        """Everything emitted since the last drain, without the sentinel itself."""
        self.send(DRAIN_FRAME)
        collected: list[dict[str, Any]] = []
        while True:
            event = self.receive()
            if event["type"] == "session.updated" and event["session"]["model"] == DRAIN_MODEL:
                return collected
            collected.append(event)

    def close_message(self) -> Mapping[str, Any]:
        """The raw `websocket.close` message, for the fatal-error cases."""
        return self.websocket.receive()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return dict(AUTH_HEADERS)


@pytest.fixture
def realtime_config() -> TalkoverConfig:
    """A config whose only non-default field is the single Bearer key."""
    return TalkoverConfig(server=ServerConfig(listen="127.0.0.1:8000", api_key=API_KEY))


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine()


@pytest.fixture
def realtime_app(realtime_config: TalkoverConfig, fake_engine: FakeEngine) -> Any:
    """The real application, with the T2.4 session state machine wired in."""
    return create_app(realtime_config, fake_engine, session_factory=WebSocketSession)


@pytest.fixture
def realtime_client(realtime_app: Any) -> TestClient:
    return TestClient(realtime_app)


@pytest.fixture
def realtime_connect(
    realtime_config: TalkoverConfig, fake_engine: FakeEngine
) -> Callable[..., Any]:
    """Factory yielding a connected, authenticated `RealtimeClient`.

    Every argument is optional: a case that needs its own scripted engine, a different
    `?model=`, extra headers or a different config passes them here.
    """

    @contextmanager
    def connect(
        engine: FakeEngine | None = None,
        *,
        query: str = "",
        headers: Mapping[str, str] | None = None,
        config: TalkoverConfig | None = None,
        handshake: bool = True,
    ) -> Iterator[RealtimeClient]:
        engine = fake_engine if engine is None else engine
        app = create_app(config or realtime_config, engine, session_factory=WebSocketSession)
        client = TestClient(app)
        url = f"/v1/realtime{query}"
        request_headers = {**AUTH_HEADERS, **(headers or {})}
        with client.websocket_connect(url, headers=request_headers) as websocket:
            realtime = RealtimeClient(websocket, engine)
            if handshake:
                realtime.open()
            yield realtime

    return connect
