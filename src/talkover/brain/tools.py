"""Schemas and execution for query_ticket / query_order / transfer_to_human.

This module is provider-neutral: it never imports from `talkover.brain.llm`. The LLM
clients consume :data:`BRAIN_TOOLS` (a list of :class:`ToolSpec`) and render it with
:func:`anthropic_tools` or :func:`openai_tools`.

Execution model (DESIGN.md 6.3):

* `query_ticket` and `query_order` are executed here, as HTTP GET requests against
  `brain.business_api.base_url`.
* `transfer_to_human` is never executed locally. Calling :meth:`ToolExecutor.execute`
  with it returns a successful :class:`ToolResult` carrying ``client_tool=True`` and the
  validated arguments; the caller (T3.3 / T3.4) turns that into a
  ``ProviderEvent(kind="interaction")`` carrying a ``TaskInteraction``. No exception is
  raised for this case.

:meth:`ToolExecutor.execute` never raises: every failure becomes a
``ToolResult(ok=False, error_kind=..., detail=...)`` that the Brain loop renders as
"explain the situation and suggest a transfer to a human agent" (DESIGN.md 6.6).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from talkover.config import BusinessApiConfig

__all__ = [
    "BRAIN_TOOLS",
    "CLIENT_TOOLS",
    "ERROR_KINDS",
    "LOOKUP_TOOLS",
    "ORDERS_PATH",
    "QUERY_ORDER",
    "QUERY_TICKET",
    "TICKETS_PATH",
    "TRANSFER_TO_HUMAN",
    "TRANSFER_TO_HUMAN_NAME",
    "ErrorKind",
    "ToolExecutor",
    "ToolResult",
    "ToolSpec",
    "anthropic_tools",
    "openai_tools",
    "tool_by_name",
]

# --------------------------------------------------------------------------------------
# Endpoint paths
# --------------------------------------------------------------------------------------

# DESIGN.md 6.6 fixes the transport (JSON over HTTP, 3 s timeout) but not the paths.
# These constants are the contract the mock business API (T3.6) must serve.
TICKETS_PATH = "/tickets"
ORDERS_PATH = "/orders"

TRANSFER_TO_HUMAN_NAME = "transfer_to_human"

ErrorKind = Literal[
    "timeout",  # the business API did not answer within business_api.timeout_sec
    "http_error",  # the business API answered with a non-2xx status
    "unavailable",  # connection failure, or a body that is not JSON
    "invalid_arguments",  # the model supplied arguments the tool cannot use
    "unknown_tool",  # the model called a name this executor does not implement
]

ERROR_KINDS: tuple[str, ...] = (
    "timeout",
    "http_error",
    "unavailable",
    "invalid_arguments",
    "unknown_tool",
)


# --------------------------------------------------------------------------------------
# Provider-neutral tool description
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """One tool, described once and rendered per LLM provider."""

    name: str
    description: str
    parameters: dict[str, Any]

    def to_anthropic(self) -> dict[str, Any]:
        """Render in the Anthropic Messages API shape (`input_schema`)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    def to_openai(self) -> dict[str, Any]:
        """Render in the OpenAI chat-completions shape (`function.parameters`)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _identifier_schema(
    primary: str,
    primary_description: str,
    *,
    object_description: str,
) -> dict[str, Any]:
    """A schema with two optional identifiers of which at least one must be present."""
    return {
        "type": "object",
        "description": object_description,
        "properties": {
            primary: {"type": "string", "description": primary_description},
            "phone": {
                "type": "string",
                "description": (
                    "The phone number the customer registered with, digits only. "
                    f"Use this when the customer does not know the {primary}."
                ),
            },
        },
        "anyOf": [{"required": [primary]}, {"required": ["phone"]}],
        "additionalProperties": False,
    }


QUERY_TICKET = ToolSpec(
    name="query_ticket",
    description=(
        "Look up a customer-service ticket by its ticket id, or by the customer's phone "
        "number when the ticket id is unknown. Supply at least one of the two."
    ),
    parameters=_identifier_schema(
        "ticket_id",
        "The ticket id the customer read out, for example 'TK20260917001'.",
        object_description="Identifiers for the ticket to look up.",
    ),
)

QUERY_ORDER = ToolSpec(
    name="query_order",
    description=(
        "Look up an order by its order number, or by the customer's phone number when "
        "the order number is unknown. Supply at least one of the two."
    ),
    parameters=_identifier_schema(
        "order_id",
        "The order number the customer read out, for example '202609170001234'.",
        object_description="Identifiers for the order to look up.",
    ),
)

TRANSFER_TO_HUMAN = ToolSpec(
    name=TRANSFER_TO_HUMAN_NAME,
    description=(
        "Hand the call over to a human agent. Call this when the customer asks for a "
        "person, or when a lookup failed and the question cannot be answered."
    ),
    parameters={
        "type": "object",
        "description": "Where to transfer the call and why.",
        "properties": {
            "department": {
                "type": "string",
                "description": (
                    "The department to transfer to, for example 'after_sales', "
                    "'billing' or 'general'."
                ),
            },
            "reason": {
                "type": "string",
                "description": "A short reason for the transfer, for the agent's screen.",
            },
        },
        "required": ["department"],
        "additionalProperties": False,
    },
)

#: Tools this executor performs itself.
LOOKUP_TOOLS: tuple[ToolSpec, ...] = (QUERY_TICKET, QUERY_ORDER)

#: Tools the Realtime client executes; only these are declared in `session.tools`.
CLIENT_TOOLS: tuple[ToolSpec, ...] = (TRANSFER_TO_HUMAN,)

#: Everything the Brain LLM is offered.
BRAIN_TOOLS: tuple[ToolSpec, ...] = LOOKUP_TOOLS + CLIENT_TOOLS

_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in BRAIN_TOOLS}

# name -> (endpoint path, primary identifier)
_LOOKUP_ROUTES: dict[str, tuple[str, str]] = {
    QUERY_TICKET.name: (TICKETS_PATH, "ticket_id"),
    QUERY_ORDER.name: (ORDERS_PATH, "order_id"),
}


def tool_by_name(name: str) -> ToolSpec | None:
    """Return the spec registered under `name`, or None."""
    return _BY_NAME.get(name)


def anthropic_tools(specs: tuple[ToolSpec, ...] = BRAIN_TOOLS) -> list[dict[str, Any]]:
    """Render `specs` for the Anthropic Messages API."""
    return [spec.to_anthropic() for spec in specs]


def openai_tools(specs: tuple[ToolSpec, ...] = BRAIN_TOOLS) -> list[dict[str, Any]]:
    """Render `specs` for OpenAI-compatible chat completions."""
    return [spec.to_openai() for spec in specs]


# --------------------------------------------------------------------------------------
# Execution result
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of one tool call.

    `ok` is True for a successful lookup and for `transfer_to_human`, which is marked
    with `client_tool=True` and carries the validated arguments in `data`.
    """

    name: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error_kind: ErrorKind | None = None
    detail: str | None = None
    client_tool: bool = False

    def to_content(self) -> str:
        """A JSON string suitable as the tool-result content sent back to the LLM."""
        if self.ok:
            return json.dumps({"ok": True, **self.data}, ensure_ascii=False)
        return json.dumps(
            {"ok": False, "error": self.error_kind, "detail": self.detail},
            ensure_ascii=False,
        )


def _failure(name: str, kind: ErrorKind, detail: str) -> ToolResult:
    return ToolResult(name=name, ok=False, error_kind=kind, detail=detail)


def _clean_string(arguments: dict[str, Any], key: str) -> tuple[str | None, str | None]:
    """Return `(value, error)` for `key`.

    `value` is the stripped string, or None when the key is absent or empty. `error` is a
    message when the value is present but not a string; no exception is raised.
    """
    if key not in arguments or arguments[key] is None:
        return None, None
    value = arguments[key]
    if not isinstance(value, str):
        return None, f"'{key}' must be a string, got {type(value).__name__}"
    value = value.strip()
    return (value or None), None


# --------------------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------------------


class ToolExecutor:
    """Executes Brain tool calls against the business HTTP API.

    Pass `client` to reuse an externally owned `httpx.AsyncClient` (it is then not closed
    by :meth:`aclose`), or `transport` to inject an `httpx.MockTransport` in tests while
    still exercising the configured base URL and timeout.
    """

    def __init__(
        self,
        config: BusinessApiConfig,
        *,
        client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._config = config
        self._owns_client = client is None
        if client is None:
            client = httpx.AsyncClient(
                base_url=config.base_url.rstrip("/"),
                timeout=httpx.Timeout(config.timeout_sec),
                transport=transport,
            )
        elif transport is not None:
            raise ValueError("pass either 'client' or 'transport', not both")
        self._client = client

    @property
    def config(self) -> BusinessApiConfig:
        return self._config

    @property
    def client(self) -> httpx.AsyncClient:
        """The underlying client; tests assert the configured timeout on it."""
        return self._client

    async def execute(self, name: str, arguments: dict[str, Any] | None) -> ToolResult:
        """Run one tool call. Never raises."""
        arguments = arguments or {}
        if not isinstance(arguments, dict):
            return _failure(name, "invalid_arguments", "arguments must be a JSON object")
        if name == TRANSFER_TO_HUMAN_NAME:
            return self._transfer(arguments)
        route = _LOOKUP_ROUTES.get(name)
        if route is None:
            return _failure(name, "unknown_tool", f"no tool named '{name}'")
        path, primary = route
        identifier, id_error = _clean_string(arguments, primary)
        phone, phone_error = _clean_string(arguments, "phone")
        if id_error or phone_error:
            return _failure(name, "invalid_arguments", id_error or phone_error or "")
        if identifier is None and phone is None:
            return _failure(
                name,
                "invalid_arguments",
                f"at least one of '{primary}' or 'phone' is required",
            )
        params: dict[str, str] = {}
        if identifier is not None:
            params[primary] = identifier
        if phone is not None:
            params["phone"] = phone
        return await self._get_json(name, path, params)

    def _transfer(self, arguments: dict[str, Any]) -> ToolResult:
        """`transfer_to_human` is handed to the client, never executed here."""
        department, dept_error = _clean_string(arguments, "department")
        reason, reason_error = _clean_string(arguments, "reason")
        if dept_error or reason_error:
            return _failure(
                TRANSFER_TO_HUMAN_NAME,
                "invalid_arguments",
                dept_error or reason_error or "",
            )
        if department is None:
            return _failure(TRANSFER_TO_HUMAN_NAME, "invalid_arguments", "'department' is required")
        data: dict[str, Any] = {"department": department}
        if reason is not None:
            data["reason"] = reason
        return ToolResult(
            name=TRANSFER_TO_HUMAN_NAME,
            ok=True,
            data=data,
            client_tool=True,
        )

    async def _get_json(self, name: str, path: str, params: dict[str, str]) -> ToolResult:
        try:
            response = await self._client.get(path, params=params)
        except httpx.TimeoutException as exc:
            return _failure(
                name,
                "timeout",
                f"business API did not respond within {self._config.timeout_sec}s: {exc}",
            )
        except httpx.HTTPError as exc:
            return _failure(name, "unavailable", f"business API is unreachable: {exc}")
        if response.status_code < 200 or response.status_code >= 300:
            return _failure(
                name,
                "http_error",
                f"business API returned HTTP {response.status_code}",
            )
        try:
            payload = response.json()
        except ValueError as exc:
            return _failure(name, "unavailable", f"business API returned a non-JSON body: {exc}")
        data = payload if isinstance(payload, dict) else {"result": payload}
        return ToolResult(name=name, ok=True, data=data)

    async def aclose(self) -> None:
        """Close the client, unless it was injected by the caller."""
        if self._owns_client:
            await self._client.aclose()
