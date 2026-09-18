"""Process composition: engine <-> realtime <-> Brain (DESIGN.md section 3).

This module is the one place where the three packages meet. It owns no protocol logic, no
inference and no business rules; it only builds the objects and hands each one the seams
the others expose:

===============================  =====================================================
Object                           Wired to
===============================  =====================================================
`BusinessProvider` / project      one per process, from `TalkoverConfig.brain` (T3.4)
`BrainBridge`                     one per Realtime session, holding that project and
                                  the engine it answers the Cerebellum through (4.6)
`WebSocketSession`                `on_engine_event` and `on_function_call_output` of
                                  that bridge (T2.4 / T3.7)
`create_app`                      the session factory above, plus `on_release`, which
                                  closes the bridge when the session slot frees (T2.8)
===============================  =====================================================

**Scope (T3.7).** This is the composition skeleton only. It takes an engine that is
already built and says nothing about starting it: `talkover serve`, the uvicorn startup,
model loading, the ASR stream and the engine reset inside `on_release` are T2.9, which
extends `build_app` rather than replacing it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from gander_runtime.contracts import new_id
from gander_runtime.coordination import ProjectRecord

from talkover.brain.provider import BusinessProject, BusinessProvider
from talkover.config import TalkoverConfig
from talkover.realtime.brain_bridge import BrainBridge
from talkover.realtime.server import CallLater, create_app
from talkover.realtime.session import WebSocketSession

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastapi import FastAPI

    from talkover.engine.protocol import EngineProtocol

__all__ = [
    "DEFAULT_PROJECT_LABEL",
    "DEFAULT_PROJECT_OWNER",
    "TalkoverApp",
    "build_app",
]

LOGGER = logging.getLogger(__name__)

#: One process serves one deployment, so the project record is a constant (DESIGN.md 5.7).
DEFAULT_PROJECT_OWNER = "talkover"
DEFAULT_PROJECT_LABEL = "customer service"


class _BrainState:
    """What the session factory writes and `GET /health` reads.

    `create_app` needs both before the application exists, so they live in this small
    mutable object rather than on :class:`TalkoverApp`, which is built afterwards.
    """

    __slots__ = ("bridge", "ready")

    def __init__(self) -> None:
        #: The bridge of the session currently holding the single slot.
        self.bridge: BrainBridge | None = None
        #: `server._readiness` reads this attribute off the `brain` component.
        self.ready = True


@dataclass(slots=True)
class TalkoverApp:
    """The composed application and the objects it owns."""

    app: FastAPI
    config: TalkoverConfig
    engine: EngineProtocol
    provider: BusinessProvider
    project: BusinessProject
    #: Whether this object built the provider and must therefore close it.
    owns_provider: bool = True
    _state: _BrainState = field(default_factory=_BrainState)

    @property
    def bridge(self) -> BrainBridge | None:
        """The Brain bridge of the running session, or `None` when the slot is free."""
        return self._state.bridge

    async def aclose(self) -> None:
        """Release the Brain side; the engine is the caller's to stop (T2.9)."""
        bridge, self._state.bridge = self._state.bridge, None
        if bridge is not None:
            await bridge.aclose()
        self._state.ready = False
        if self.owns_provider:
            await self.provider.close()


async def build_app(
    config: TalkoverConfig,
    engine: EngineProtocol,
    *,
    provider: BusinessProvider | None = None,
    asr: object | None = None,
    call_later: CallLater | None = None,
) -> TalkoverApp:
    """Compose the ASGI application around `engine` and a business Brain provider.

    `provider` is injectable so that a test drives the whole chain against `FakeLLM`;
    when it is omitted one is built from `config.brain` and closed by
    :meth:`TalkoverApp.aclose`. The engine is also the bridge's channel back to the model
    (`feed_tool_response` / `feed_worker_delivery`, DESIGN.md 4.6), so no separate
    Cerebellum object is passed around.

    The engine is neither started nor stopped here.
    """
    owns_provider = provider is None
    brain_provider = provider if provider is not None else BusinessProvider(config.brain)
    project = await brain_provider.open_project(
        ProjectRecord(
            project_id=new_id("project"),
            owner_id=DEFAULT_PROJECT_OWNER,
            label=DEFAULT_PROJECT_LABEL,
            provider_name=brain_provider.name,
        )
    )
    state = _BrainState()

    def session_factory(
        websocket: Any, session_config: TalkoverConfig, session_engine: Any, model: str
    ) -> WebSocketSession:
        """Build one Realtime session with its own Brain bridge."""
        bridge = BrainBridge(project, engine=session_engine)
        runner = WebSocketSession(
            websocket,
            session_config,
            session_engine,
            model,
            on_engine_event=bridge.on_engine_event,
            on_function_call_output=bridge.on_function_call_output,
        )
        bridge.bind(runner.session)
        state.bridge = bridge
        return runner

    async def on_release() -> None:
        """Close the Brain side of the session whose slot has just been released.

        T2.9 adds the engine reset here; this module never calls `engine.start` or
        `engine.stop`, exactly as `create_app` does not (T2.8).
        """
        bridge, state.bridge = state.bridge, None
        if bridge is not None:
            await bridge.aclose()

    app = create_app(
        config,
        engine,
        asr=asr,
        brain=state,
        session_factory=session_factory,
        on_release=on_release,
        call_later=call_later,
    )
    return TalkoverApp(
        app=app,
        config=config,
        engine=engine,
        provider=brain_provider,
        project=project,
        owns_provider=owns_provider,
        _state=state,
    )
