"""Transport-layer tests: auth, `/health`, and the single-session guard (T2.3).

These cases are GPU-free: the engine is a tiny fake exposing only what the server
reads from `EngineProtocol`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse

from talkover.config import ServerConfig, TalkoverConfig
from talkover.realtime.server import SessionSlot, create_app

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


class FakeEngine:
    """The smallest object the server accepts as an engine."""

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready


class FakeComponent:
    """Stand-in for the ASR backend or the Brain provider."""

    def __init__(self, ready: bool = True) -> None:
        self.ready = ready


def make_config(api_key: str = API_KEY) -> TalkoverConfig:
    return TalkoverConfig(server=ServerConfig(listen="127.0.0.1:8000", api_key=api_key))


def make_client(**kwargs) -> TestClient:
    engine = kwargs.pop("engine", None) or FakeEngine()
    config = kwargs.pop("config", None) or make_config()
    return TestClient(create_app(config, engine, **kwargs))


# ---------------------------------------------------------------------------
# authentication
# ---------------------------------------------------------------------------


def test_missing_key_is_401() -> None:
    client = make_client()
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        client.websocket_connect("/v1/realtime?model=gpt-realtime"),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_bad_key_is_401() -> None:
    client = make_client()
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer wrong-key"}),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_non_bearer_scheme_is_401() -> None:
    client = make_client()
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        client.websocket_connect("/v1/realtime", headers={"Authorization": API_KEY}),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_empty_configured_key_denies_everything() -> None:
    client = make_client(config=make_config(api_key=""))
    with (
        pytest.raises(WebSocketDenialResponse) as excinfo,
        client.websocket_connect("/v1/realtime", headers={"Authorization": "Bearer "}),
    ):
        pass
    assert excinfo.value.status_code == 401


def test_good_key_receives_session_created_with_model_echo() -> None:
    client = make_client()
    with client.websocket_connect("/v1/realtime?model=gpt-realtime", headers=AUTH) as ws:
        event = ws.receive_json()
    assert event["type"] == "session.created"
    assert event["session"]["model"] == "gpt-realtime"
    assert event["session"]["type"] == "realtime"
    assert event["session"]["id"].startswith("sess_")
    assert event["event_id"].startswith("event_")


def test_absent_model_echoes_the_default() -> None:
    client = make_client()
    with client.websocket_connect("/v1/realtime", headers=AUTH) as ws:
        event = ws.receive_json()
    assert event["session"]["model"] == "talkover"


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


def test_health_idle_reports_every_component() -> None:
    client = make_client(asr=FakeComponent(), brain=FakeComponent())
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "idle",
        "engine": True,
        "asr": True,
        "brain": True,
        "session_active": False,
    }


def test_health_reports_unconfigured_components_as_null() -> None:
    body = make_client().get("/health").json()
    assert body["asr"] is None
    assert body["brain"] is None
    assert body["status"] == "idle"


def test_health_is_503_when_the_engine_is_not_ready() -> None:
    client = make_client(engine=FakeEngine(ready=False), asr=FakeComponent(ready=False))
    response = client.get("/health")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["engine"] is False
    assert body["asr"] is False


def test_health_is_busy_during_a_session_and_idle_after_it() -> None:
    client = make_client()
    assert client.get("/health").status_code == 200
    with client.websocket_connect("/v1/realtime", headers=AUTH) as ws:
        ws.receive_json()
        response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["status"] == "busy"
        assert response.json()["session_active"] is True
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "idle"


# ---------------------------------------------------------------------------
# single-session guard
# ---------------------------------------------------------------------------


def test_second_connection_gets_engine_busy_and_is_closed() -> None:
    client = make_client()
    with client.websocket_connect("/v1/realtime", headers=AUTH) as first:
        first.receive_json()
        with client.websocket_connect("/v1/realtime", headers=AUTH) as second:
            event = second.receive_json()
            assert event["type"] == "error"
            assert event["error"] == {
                "type": "server_error",
                "code": "engine_busy",
                "message": "A session is already running on this engine.",
                "param": None,
                "event_id": None,
            }
            closed = second.receive()
            assert closed["type"] == "websocket.close"
            assert closed["code"] == 1011

        # The first session is unaffected by the rejected one.
        first.send_json({"type": "input_audio_buffer.clear"})
        first.send_json({"type": "nope"})
        error = first.receive_json()
        assert error["error"]["code"] == "invalid_value"
        assert error["error"]["param"] == "type"

    assert client.get("/health").json()["status"] == "idle"


def test_slot_is_released_after_a_failing_session() -> None:
    def factory(websocket, config, engine, model):
        class Boom:
            async def run(self) -> None:
                await websocket.accept()  # already accepted: raises RuntimeError
                raise RuntimeError("boom")

        return Boom()

    client = make_client(session_factory=factory)
    with pytest.raises(RuntimeError), client.websocket_connect("/v1/realtime", headers=AUTH):
        pass
    assert client.get("/health").json()["status"] == "idle"


def test_session_slot_tracks_its_holder() -> None:
    slot = SessionSlot()
    first, second = object(), object()
    assert slot.acquire(first) is True
    assert slot.busy is True
    assert slot.acquire(second) is False
    slot.release(second)  # a foreign release is a no-op
    assert slot.busy is True
    slot.release(first)
    assert slot.busy is False
    assert slot.acquire(second) is True


# ---------------------------------------------------------------------------
# frame validation seam
# ---------------------------------------------------------------------------


def test_malformed_json_yields_an_error_and_keeps_the_connection_open() -> None:
    client = make_client()
    with client.websocket_connect("/v1/realtime", headers=AUTH) as ws:
        ws.receive_json()
        ws.send_text("{not json")
        event = ws.receive_json()
        assert event["type"] == "error"
        assert event["error"]["code"] == "invalid_event"
        assert event["error"]["type"] == "invalid_request_error"

        # Still alive: a valid event is accepted silently and a second bad one answered.
        ws.send_text(json.dumps({"type": "input_audio_buffer.commit"}))
        ws.send_text(json.dumps({"type": "session.update", "event_id": "e1"}))
        second = ws.receive_json()
        assert second["error"]["code"] == "invalid_value"
        assert second["error"]["param"] == "session"
        assert second["error"]["event_id"] == "e1"


def test_custom_session_factory_replaces_the_default() -> None:
    seen: dict[str, object] = {}

    def factory(websocket, config, engine, model):
        class Echo:
            async def run(self) -> None:
                seen["model"] = model
                seen["engine"] = engine
                await websocket.send_json({"type": "custom", "model": model})

        return Echo()

    engine = FakeEngine()
    client = make_client(engine=engine, session_factory=factory)
    with client.websocket_connect("/v1/realtime?model=m1", headers=AUTH) as ws:
        assert ws.receive_json() == {"type": "custom", "model": "m1"}
    assert seen == {"model": "m1", "engine": engine}
