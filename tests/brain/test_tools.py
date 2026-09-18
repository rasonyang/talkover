"""Tests for the Brain tool schemas and their HTTP execution (T3.1). GPU-free."""

from __future__ import annotations

import httpx
import pytest

from talkover.brain.tools import (
    BRAIN_TOOLS,
    CLIENT_TOOLS,
    ORDERS_PATH,
    QUERY_ORDER,
    QUERY_TICKET,
    TICKETS_PATH,
    TRANSFER_TO_HUMAN,
    ToolExecutor,
    ToolSpec,
    anthropic_tools,
    openai_tools,
    tool_by_name,
)
from talkover.config import BusinessApiConfig

BASE_URL = "http://business.test"


def make_executor(handler) -> ToolExecutor:
    config = BusinessApiConfig(base_url=BASE_URL, timeout_sec=3.0)
    return ToolExecutor(config, transport=httpx.MockTransport(handler))


# --------------------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------------------


def test_three_tools_are_declared() -> None:
    assert [spec.name for spec in BRAIN_TOOLS] == [
        "query_ticket",
        "query_order",
        "transfer_to_human",
    ]
    assert all(isinstance(spec, ToolSpec) for spec in BRAIN_TOOLS)
    assert CLIENT_TOOLS == (TRANSFER_TO_HUMAN,)
    assert tool_by_name("query_order") is QUERY_ORDER
    assert tool_by_name("nope") is None


@pytest.mark.parametrize(
    ("spec", "primary"),
    [(QUERY_TICKET, "ticket_id"), (QUERY_ORDER, "order_id")],
)
def test_lookup_schema_requires_at_least_one_identifier(spec: ToolSpec, primary: str) -> None:
    params = spec.parameters
    assert set(params["properties"]) == {primary, "phone"}
    assert params["anyOf"] == [{"required": [primary]}, {"required": ["phone"]}]
    assert params.get("required") is None


def test_transfer_schema() -> None:
    params = TRANSFER_TO_HUMAN.parameters
    assert set(params["properties"]) == {"department", "reason"}
    assert params["required"] == ["department"]


def test_provider_rendering() -> None:
    anthropic = anthropic_tools()
    assert anthropic[0]["name"] == "query_ticket"
    assert anthropic[0]["input_schema"] is QUERY_TICKET.parameters
    assert "parameters" not in anthropic[0]

    openai = openai_tools()
    assert openai[0]["type"] == "function"
    assert openai[0]["function"]["name"] == "query_ticket"
    assert openai[0]["function"]["parameters"] is QUERY_TICKET.parameters


# --------------------------------------------------------------------------------------
# Successful lookups
# --------------------------------------------------------------------------------------


async def test_query_ticket_success() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"ticket_id": "TK001", "status": "open"})

    executor = make_executor(handler)
    result = await executor.execute("query_ticket", {"ticket_id": " TK001 "})
    await executor.aclose()

    assert result.ok is True
    assert result.client_tool is False
    assert result.data == {"ticket_id": "TK001", "status": "open"}
    assert seen[0].url.path == TICKETS_PATH
    # Whitespace is stripped before the value reaches the query string.
    assert dict(seen[0].url.params) == {"ticket_id": "TK001"}


async def test_query_order_success_by_phone() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"order_id": "202609170001234", "amount": "128.00"})

    executor = make_executor(handler)
    result = await executor.execute("query_order", {"phone": "13800000000"})
    await executor.aclose()

    assert result.ok is True
    assert result.data["order_id"] == "202609170001234"
    assert seen[0].url.path == ORDERS_PATH
    assert dict(seen[0].url.params) == {"phone": "13800000000"}
    assert '"ok": true' in result.to_content()


async def test_non_dict_json_body_is_wrapped() -> None:
    executor = make_executor(lambda request: httpx.Response(200, json=[{"order_id": "A"}]))
    result = await executor.execute("query_order", {"order_id": "A"})
    await executor.aclose()

    assert result.ok is True
    assert result.data == {"result": [{"order_id": "A"}]}


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


async def test_timeout_returns_structured_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("timed out", request=request)

    executor = make_executor(handler)
    result = await executor.execute("query_order", {"order_id": "A"})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "timeout"
    assert "3.0" in (result.detail or "")


async def test_configured_timeout_is_applied_to_the_client() -> None:
    config = BusinessApiConfig(base_url=BASE_URL, timeout_sec=3.0)
    executor = ToolExecutor(config, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    timeout = executor.client.timeout
    await executor.aclose()

    assert timeout.connect == 3.0
    assert timeout.read == 3.0
    assert timeout.write == 3.0
    assert timeout.pool == 3.0


async def test_http_500_returns_http_error() -> None:
    executor = make_executor(lambda request: httpx.Response(500, text="boom"))
    result = await executor.execute("query_ticket", {"ticket_id": "TK001"})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "http_error"
    assert "500" in (result.detail or "")
    assert '"ok": false' in result.to_content()


async def test_connection_error_returns_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    executor = make_executor(handler)
    result = await executor.execute("query_ticket", {"phone": "13800000000"})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "unavailable"


async def test_non_json_body_returns_unavailable() -> None:
    executor = make_executor(lambda request: httpx.Response(200, text="<html>nope</html>"))
    result = await executor.execute("query_ticket", {"ticket_id": "TK001"})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "unavailable"


@pytest.mark.parametrize(
    "arguments",
    [{}, {"ticket_id": "  "}, {"ticket_id": None, "phone": None}, {"ticket_id": 123}],
)
async def test_invalid_arguments_make_no_http_call(arguments: dict) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    executor = make_executor(handler)
    result = await executor.execute("query_ticket", arguments)
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "invalid_arguments"
    assert calls == []


async def test_unknown_tool() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    executor = make_executor(handler)
    result = await executor.execute("delete_everything", {"x": 1})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "unknown_tool"
    assert calls == []


# --------------------------------------------------------------------------------------
# transfer_to_human
# --------------------------------------------------------------------------------------


async def test_transfer_to_human_performs_no_http_call() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={})

    executor = make_executor(handler)
    result = await executor.execute(
        "transfer_to_human", {"department": "after_sales", "reason": "customer asked"}
    )
    await executor.aclose()

    assert calls == []
    assert result.ok is True
    assert result.client_tool is True
    assert result.data == {"department": "after_sales", "reason": "customer asked"}


async def test_transfer_to_human_without_reason() -> None:
    executor = make_executor(lambda request: httpx.Response(200, json={}))
    result = await executor.execute("transfer_to_human", {"department": "general"})
    await executor.aclose()

    assert result.ok is True
    assert result.data == {"department": "general"}


async def test_transfer_to_human_requires_department() -> None:
    executor = make_executor(lambda request: httpx.Response(200, json={}))
    result = await executor.execute("transfer_to_human", {"reason": "no idea"})
    await executor.aclose()

    assert result.ok is False
    assert result.error_kind == "invalid_arguments"


# --------------------------------------------------------------------------------------
# Client ownership
# --------------------------------------------------------------------------------------


async def test_injected_client_is_not_closed() -> None:
    client = httpx.AsyncClient(
        base_url=BASE_URL,
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"ok": 1})),
    )
    executor = ToolExecutor(BusinessApiConfig(base_url=BASE_URL), client=client)
    result = await executor.execute("query_order", {"order_id": "A"})
    await executor.aclose()

    assert result.ok is True
    assert client.is_closed is False
    await client.aclose()


def test_client_and_transport_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        ToolExecutor(
            BusinessApiConfig(base_url=BASE_URL),
            client=httpx.AsyncClient(),
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        )


# --------------------------------------------------------------------------------------
# Against the mock business API, in-process (T3.6)
# --------------------------------------------------------------------------------------


def make_mock_executor() -> ToolExecutor:
    from talkover.brain.mock_api import create_mock_app

    config = BusinessApiConfig(base_url=BASE_URL, timeout_sec=3.0)
    client = httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.ASGITransport(app=create_mock_app())
    )
    return ToolExecutor(config, client=client)


async def test_query_ticket_against_the_mock_api() -> None:
    executor = make_mock_executor()
    result = await executor.execute("query_ticket", {"ticket_id": "TK20260917001"})
    await executor.client.aclose()

    assert result.ok is True
    assert result.data["ticket_id"] == "TK20260917001"


async def test_query_order_against_the_mock_api() -> None:
    executor = make_mock_executor()
    result = await executor.execute("query_order", {"order_id": "202609150001234567"})
    await executor.client.aclose()

    assert result.ok is True
    assert result.data["amount"] == "1299.00"
