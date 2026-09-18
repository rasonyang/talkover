"""FastAPI + WebSocket server, /v1/realtime and /health.

This module owns the transport layer described in `docs/protocol-profile.md` section 1
and section 10:

- `GET /v1/realtime?model=<any>` upgraded to a WebSocket, authenticated with a single
  `Authorization: Bearer <server.api_key>` key; a missing or wrong key is rejected with
  HTTP 401 before the upgrade.
- `GET /health` reporting engine, ASR and Brain readiness; 200 when idle and ready,
  503 while a session is running or while the engine is not ready.
- One session per process: a second concurrent connection is accepted, told
  `code: "engine_busy"` and closed, while the first session keeps running.

- Reconnection (T2.8): a disconnect does not end the session. `SessionSlot` keeps it
  reserved for `realtime.trailing_silence_sec`, and a new connection presenting the same
  `session_id` resumes it; anything else meets the same `engine_busy` a live second
  connection does. When the window expires the session is closed, the release hook runs
  and `/health` reports `idle` again.

The per-connection protocol logic lives in `talkover.realtime.session` (T2.4). This
module only owns the seam: `session_factory(websocket, config, engine, model)` returns
an object with an `async run()` method, and, when it also offers `session_id` / `attach`
/ `aclose` (`ResumableSession`), the slot gives it the reconnect window.
`DefaultSession` below is a placeholder that sends `session.created` and rejects
malformed frames, so the transport can be tested before the state machine exists; it is
not resumable, so its slot is freed the moment the socket closes.

The engine object follows `talkover.engine.protocol.EngineProtocol`; this module only
reads its `ready` property and never imports the module at runtime. Starting, resetting and
stopping the engine is the application's business (`talkover.app`, T2.9), reached through
the `lifespan` and `on_release` hooks of `create_app`.
"""

from __future__ import annotations

import asyncio
import hmac
import logging
from collections.abc import Awaitable, Callable
from pathlib import PurePath
from typing import TYPE_CHECKING, Any, Protocol

from fastapi import FastAPI, WebSocket
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocketDisconnect, WebSocketState

from talkover.config import TalkoverConfig
from talkover.realtime.events import (
    ProtocolError,
    SessionCreated,
    error_event,
    new_session_id,
    parse_client_event,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from talkover.engine.protocol import EngineProtocol
else:  # pragma: no cover - runtime duck typing
    EngineProtocol = Any

__all__ = [
    "DEFAULT_MODEL",
    "RESUME_HEADER",
    "RESUME_QUERY_PARAM",
    "DefaultSession",
    "ResumableSession",
    "SessionSlot",
    "create_app",
    "default_model",
]

LOGGER = logging.getLogger(__name__)

#: `?model=` is echoed, never routed on; this is the echo when the config names nothing.
DEFAULT_MODEL = "talkover"

#: Close code for a fatal server-side error (profile section 8.1).
FATAL_CLOSE_CODE = 1011

#: Query parameter naming the session a reconnection wants to resume (profile section 10).
RESUME_QUERY_PARAM = "session_id"

#: Header carrying the same id, for a client whose URL is fixed by its dialler.
RESUME_HEADER = "x-talkover-session-id"

#: Signature of the timer seam: `call_later(delay, callback)` returning a handle with
#: `cancel()`. `asyncio.AbstractEventLoop.call_later` satisfies it; tests pass a fake so
#: the reconnect window closes on demand instead of after eight real seconds.
CallLater = Callable[[float, Callable[[], None]], Any]


class SessionRunner(Protocol):
    """What `session_factory` must return: an object driving one WebSocket."""

    async def run(self) -> None: ...


class ResumableSession(SessionRunner, Protocol):
    """A `SessionRunner` that outlives its WebSocket (`session.WebSocketSession`).

    `SessionSlot` only opens a reconnect window for a runner offering all of it; the
    `resumable` property lets the runner close the window early, for instance after the
    engine has failed.
    """

    @property
    def session_id(self) -> str: ...

    @property
    def resumable(self) -> bool: ...

    async def attach(self, websocket: WebSocket) -> None: ...

    async def aclose(self) -> None: ...


def _is_resumable(runner: object | None) -> bool:
    """Whether `runner` implements `ResumableSession` and still wants the window."""
    if runner is None:
        return False
    if not all(hasattr(runner, name) for name in ("session_id", "attach", "aclose")):
        return False
    return bool(getattr(runner, "resumable", True))


def _call_later(delay: float, callback: Callable[[], None]) -> Any:
    """Default timer: one shot on the running event loop."""
    return asyncio.get_running_loop().call_later(delay, callback)


# ---------------------------------------------------------------------------
# single-session guard
# ---------------------------------------------------------------------------


class SessionSlot:
    """The one session slot of the process, with its reconnect window (T2.8).

    `acquire` succeeds only when the slot is free and records the holder, so a late
    `release` from an already-replaced connection cannot free someone else's slot.

    A slot holds two things: the *holder*, the WebSocket currently connected, and the
    *runner*, the session object itself. They come apart on a disconnect: `detach` drops
    the holder but keeps the runner, arms a `window_sec` timer and leaves the slot busy,
    so `resume` can hand the same runner to a reconnection presenting its `session_id`
    while every other connection still gets `engine_busy`. When the timer fires — or when
    `discard` reports a fatal session — the runner is closed, `on_release` runs and the
    slot is free again.

    A runner that is not a `ResumableSession`, or a `window_sec` of zero, skips all of
    that: the slot is released as soon as the socket closes.
    """

    def __init__(
        self,
        *,
        window_sec: float = 0.0,
        call_later: CallLater | None = None,
        on_release: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._holder: object | None = None
        self._runner: object | None = None
        self._session_id: str | None = None
        self._window_sec = max(0.0, float(window_sec))
        self._call_later = call_later or _call_later
        self._on_release = on_release
        self._timer: Any = None
        self._releasing = False
        self._expiries: set[asyncio.Task[None]] = set()

    # -- views -------------------------------------------------------------

    @property
    def busy(self) -> bool:
        """Whether the slot is taken, including by a session inside its window."""
        return self._holder is not None or self._runner is not None or self._releasing

    @property
    def attached(self) -> bool:
        """Whether a WebSocket is connected to the session holding the slot."""
        return self._holder is not None

    @property
    def window_sec(self) -> float:
        """How long a disconnected session is kept, in seconds."""
        return self._window_sec

    @property
    def detached_session_id(self) -> str | None:
        """The `sess_…` id a reconnection may resume, or `None`."""
        if self._holder is not None or self._releasing:
            return None
        return self._session_id

    # -- taking and giving back --------------------------------------------

    def acquire(self, holder: object) -> bool:
        """Take the slot for `holder`; return False when it is already taken."""
        if self.busy:
            return False
        self._holder = holder
        return True

    def bind(self, holder: object, runner: object) -> None:
        """Record the session object `holder` is driving."""
        if self._holder is not holder:
            return
        self._runner = runner
        self._session_id = str(runner.session_id) if _is_resumable(runner) else None

    def resume(self, session_id: str | None, holder: object) -> Any | None:
        """Hand back the detached session `session_id` names, or `None`.

        `None` means the caller is an ordinary new connection: either nothing is waiting,
        or what is waiting is not what it asked for, and the slot decides on `acquire`.
        """
        if session_id is None or self._holder is not None or self._releasing:
            return None
        if self._runner is None or session_id != self._session_id:
            return None
        self._cancel_timer()
        self._holder = holder
        return self._runner

    def release(self, holder: object) -> None:
        """Free the slot if `holder` still owns it; a foreign release is a no-op.

        Synchronous and unconditional: it forgets the session without closing it, so it
        is only for a runner that owns nothing. `discard` is the one to use otherwise.
        """
        if self._holder is not holder:
            return
        self._cancel_timer()
        self._holder = None
        self._runner = None
        self._session_id = None

    async def detach(self, holder: object) -> bool:
        """End one connection; return whether the session was kept for a reconnection."""
        if self._holder is not holder:
            return False
        runner = self._runner
        if self._window_sec <= 0.0 or not _is_resumable(runner):
            await self.discard(holder)
            return False
        self._holder = None
        self._arm(runner)
        return True

    async def discard(self, holder: object) -> None:
        """Release the slot now, closing the session; used for a fatal connection."""
        if self._holder is not holder:
            return
        self._cancel_timer()
        self._holder = None
        await self._close(self._runner)

    # -- the window --------------------------------------------------------

    def _arm(self, runner: object) -> None:
        """Start the reconnect window for `runner`."""

        def fire() -> None:
            self._timer = None
            task = asyncio.ensure_future(self._expire(runner))
            self._expiries.add(task)
            task.add_done_callback(self._expiries.discard)

        self._timer = self._call_later(self._window_sec, fire)

    async def _expire(self, runner: object) -> None:
        """Release a session nobody reconnected to."""
        if self._runner is not runner or self._holder is not None:
            return
        try:
            await self._close(runner)
        except Exception:  # pragma: no cover - defensive; a release must not go unnoticed
            LOGGER.exception("Releasing the realtime session after its window failed.")

    async def _close(self, runner: object | None) -> None:
        """Close `runner`, run the release hook, and free the slot either way."""
        self._releasing = True
        try:
            aclose = getattr(runner, "aclose", None)
            if aclose is not None:
                await aclose()
            if self._on_release is not None:
                await self._on_release()
        finally:
            self._releasing = False
            if self._runner is runner:
                self._runner = None
                self._session_id = None

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


# ---------------------------------------------------------------------------
# default (placeholder) session
# ---------------------------------------------------------------------------


class DefaultSession:
    """Minimal stand-in for the T2.4 session state machine.

    It sends `session.created` echoing the requested model, then validates every
    client frame with `parse_client_event`, answering a rejected frame with the
    documented `error` event and keeping the connection open. Accepted events have
    no effect.
    """

    def __init__(
        self,
        websocket: WebSocket,
        config: TalkoverConfig,
        engine: EngineProtocol,
        model: str,
    ) -> None:
        self.websocket = websocket
        self.config = config
        self.engine = engine
        self.model = model
        self.session_id = new_session_id()

    def session_object(self) -> dict[str, Any]:
        """The `session` object echoed by `session.created`."""
        return {
            "id": self.session_id,
            "object": "realtime.session",
            "type": "realtime",
            "model": self.model,
        }

    async def run(self) -> None:
        await self.websocket.send_json(SessionCreated(session=self.session_object()).to_wire())
        while True:
            try:
                message = await self.websocket.receive()
            except WebSocketDisconnect:
                return
            if message["type"] == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                raw = message.get("bytes", b"")
            try:
                parse_client_event(raw)
            except ProtocolError as exc:
                await self.websocket.send_json(error_event(exc))


def _default_session_factory(
    websocket: WebSocket,
    config: TalkoverConfig,
    engine: EngineProtocol,
    model: str,
) -> SessionRunner:
    return DefaultSession(websocket, config, engine, model)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def default_model(config: TalkoverConfig) -> str:
    """The model name echoed when the client sends no `?model=`.

    Talkover serves exactly one model, so `?model=` is never routed on (profile section 1)
    — but a client that asked for nothing should still be told what it reached. The name of
    the base checkpoint directory is that answer (`MiniCPM-o-4_5` for the config DESIGN.md
    section 9 ships); :data:`DEFAULT_MODEL` stands in while `engine.base_model` is unset,
    which is the case in every test that does not care.
    """
    base_model = config.engine.base_model.strip().rstrip("/")
    if not base_model:
        return DEFAULT_MODEL
    return PurePath(base_model).name or DEFAULT_MODEL


def _readiness(component: object | None) -> bool | None:
    """`True` / `False` for a configured component, `None` when not configured."""
    if component is None:
        return None
    return bool(getattr(component, "ready", False))


def _requested_session_id(websocket: WebSocket) -> str | None:
    """The session a connection asks to resume, from the query parameter or the header."""
    value = websocket.query_params.get(RESUME_QUERY_PARAM) or websocket.headers.get(RESUME_HEADER)
    value = (value or "").strip()
    return value or None


def _bearer_token(header: str | None) -> str | None:
    """Extract the token of an `Authorization: Bearer <token>` header."""
    if not header:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        return None
    return token.strip()


def _authorized(websocket: WebSocket, api_key: str) -> bool:
    """Constant-time check of the single Bearer key (profile section 1)."""
    if not api_key:
        # No key configured: nothing can be presented, so every upgrade is denied.
        return False
    token = _bearer_token(websocket.headers.get("authorization"))
    if token is None:
        return False
    return hmac.compare_digest(token, api_key)


async def _deny(websocket: WebSocket, status_code: int, message: str) -> None:
    """Refuse the upgrade with an HTTP status instead of the default 403 close."""
    response = JSONResponse({"error": {"message": message}}, status_code=status_code)
    try:
        await websocket.send_denial_response(response)
    except RuntimeError:  # pragma: no cover - server without the denial extension
        await websocket.close(code=1008, reason=message)


async def _close(websocket: WebSocket, code: int, reason: str) -> None:
    if websocket.client_state is not WebSocketState.DISCONNECTED:
        try:
            await websocket.close(code=code, reason=reason)
        except RuntimeError:  # pragma: no cover - already closed by the peer
            pass


# ---------------------------------------------------------------------------
# application
# ---------------------------------------------------------------------------


def create_app(
    config: TalkoverConfig,
    engine: EngineProtocol,
    *,
    asr: object | None = None,
    brain: object | None = None,
    session_factory: Any = None,
    on_release: Callable[[], Awaitable[None]] | None = None,
    call_later: CallLater | None = None,
    lifespan: Any = None,
) -> FastAPI:
    """Build the Talkover ASGI application.

    `engine` follows `EngineProtocol`; `asr` and `brain` are optional and only need a
    `ready` attribute. `session_factory(websocket, config, engine, model)` returns the
    object that drives one connection; it defaults to the placeholder `DefaultSession`,
    which is what a bare server with no application around it gets. `talkover.app.build_app`
    always passes `session.WebSocketSession` (T2.9).

    `on_release` is awaited every time the single slot becomes free again — after the
    reconnect window of a disconnected session, or straight away for a session that
    cannot be resumed. It is where `talkover.app` resets the engine conversation; this
    module never calls `engine.start` or `engine.stop` itself, and `lifespan` is the seam
    the application hangs that on. `call_later` replaces the reconnect timer, so tests do
    not wait `realtime.trailing_silence_sec` seconds.
    """
    factory = session_factory or _default_session_factory
    app = FastAPI(title="Talkover", version="0.1.0", lifespan=lifespan)
    slot = SessionSlot(
        window_sec=config.realtime.trailing_silence_sec,
        call_later=call_later,
        on_release=on_release,
    )
    app.state.config = config
    app.state.engine = engine
    app.state.asr = asr
    app.state.brain = brain
    app.state.session_slot = slot

    @app.get("/health")
    async def health() -> Response:
        engine_ready = _readiness(engine)
        if not engine_ready:
            status = "not_ready"
        elif slot.busy:
            status = "busy"
        else:
            status = "idle"
        body = {
            "status": status,
            "engine": engine_ready,
            "asr": _readiness(asr),
            "brain": _readiness(brain),
            "session_active": slot.busy,
        }
        return JSONResponse(body, status_code=200 if status == "idle" else 503)

    @app.websocket("/v1/realtime")
    async def realtime(websocket: WebSocket) -> None:
        if not _authorized(websocket, config.server.api_key):
            await _deny(websocket, 401, "Missing or invalid Bearer token.")
            return

        model = websocket.query_params.get("model") or default_model(config)
        requested = _requested_session_id(websocket)
        await websocket.accept()

        runner: Any = slot.resume(requested, websocket)
        resumed = runner is not None
        # A connection that resumed nothing is an ordinary new one, whether it named no
        # session or one that is already gone; both are refused while the slot is taken.
        if not resumed and not slot.acquire(websocket):
            busy = ProtocolError("engine_busy", "A session is already running on this engine.")
            await websocket.send_json(error_event(busy))
            await _close(websocket, FATAL_CLOSE_CODE, "engine_busy")
            return
        try:
            if runner is None:
                runner = factory(websocket, config, engine, model)
                slot.bind(websocket, runner)
            if resumed:
                await runner.attach(websocket)
            else:
                await runner.run()
        except WebSocketDisconnect:
            await slot.detach(websocket)
        except asyncio.CancelledError:
            # The server cancelled the connection rather than the client closing it; the
            # session is no more finished than after a dropped socket, so it keeps its
            # window. `detach` never suspends on this path, so the cancellation stands.
            await slot.detach(websocket)
            raise
        except ProtocolError as exc:
            try:
                await websocket.send_json(error_event(exc))
                await _close(websocket, FATAL_CLOSE_CODE, exc.code)
            finally:
                await slot.discard(websocket)
        except BaseException:
            await slot.discard(websocket)
            raise
        else:
            await slot.detach(websocket)
        finally:
            await _close(websocket, 1000, "")

    return app
