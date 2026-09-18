"""Tests for the mock business API (T3.6), driven through ToolExecutor. GPU-free."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from talkover.brain.mock_api import DEFAULT_DATASET, create_mock_app
from talkover.brain.tools import ToolExecutor
from talkover.config import BusinessApiConfig

BASE_URL = "http://mock.business.test"

KNOWN_TICKET_ID = "TK20260917001"
KNOWN_ORDER_ID = "202609150001234567"
KNOWN_PHONE = "13800138000"


class DeadlineTransport(httpx.AsyncBaseTransport):
    """ASGI transport that enforces a deadline.

    `httpx.ASGITransport` calls the app in-process, so the client's own read timeout is
    never armed; this wrapper turns a slow app into the `httpx.ReadTimeout` a real socket
    would raise, which is what maps to the executor's `timeout` error kind.
    """

    def __init__(self, app, timeout_sec: float) -> None:
        self._inner = httpx.ASGITransport(app=app)
        self._timeout_sec = timeout_sec

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        try:
            return await asyncio.wait_for(
                self._inner.handle_async_request(request), self._timeout_sec
            )
        except TimeoutError as exc:
            raise httpx.ReadTimeout("mock business API timed out", request=request) from exc


def make_executor(*, delay_sec: float = 0.0, timeout_sec: float = 3.0) -> ToolExecutor:
    app = create_mock_app(delay_sec=delay_sec)
    config = BusinessApiConfig(base_url=BASE_URL, timeout_sec=timeout_sec)
    client = httpx.AsyncClient(
        base_url=BASE_URL,
        timeout=httpx.Timeout(timeout_sec),
        transport=DeadlineTransport(app, timeout_sec),
    )
    return ToolExecutor(config, client=client)


@pytest.fixture
async def executor():
    executor = make_executor()
    yield executor
    await executor.client.aclose()


# --------------------------------------------------------------------------------------
# Successful lookups
# --------------------------------------------------------------------------------------


async def test_query_ticket_by_id(executor: ToolExecutor) -> None:
    result = await executor.execute("query_ticket", {"ticket_id": KNOWN_TICKET_ID})

    assert result.ok is True
    assert result.data["ticket_id"] == KNOWN_TICKET_ID
    assert result.data["phone"] == KNOWN_PHONE
    assert result.data["status"] == "processing"


async def test_query_order_by_id(executor: ToolExecutor) -> None:
    result = await executor.execute("query_order", {"order_id": KNOWN_ORDER_ID})

    assert result.ok is True
    assert result.data["order_id"] == KNOWN_ORDER_ID
    assert result.data["amount"] == "1299.00"
    assert result.data["currency"] == "CNY"
    assert result.data["logistics"]["tracking_number"] == "SF1234567890123"


async def test_query_ticket_by_phone_returns_all_matches(executor: ToolExecutor) -> None:
    result = await executor.execute("query_ticket", {"phone": KNOWN_PHONE})

    assert result.ok is True
    assert result.data["count"] == 2
    ids = [ticket["ticket_id"] for ticket in result.data["tickets"]]
    assert KNOWN_TICKET_ID in ids


async def test_query_order_by_phone_returns_all_matches(executor: ToolExecutor) -> None:
    result = await executor.execute("query_order", {"phone": KNOWN_PHONE})

    assert result.ok is True
    assert result.data["count"] == 2
    assert all(order["phone"] == KNOWN_PHONE for order in result.data["orders"])


async def test_unknown_phone_is_an_empty_but_successful_result(executor: ToolExecutor) -> None:
    result = await executor.execute("query_order", {"phone": "13000000000"})

    assert result.ok is True
    assert result.data == {"phone": "13000000000", "count": 0, "orders": []}


async def test_id_and_phone_must_both_match(executor: ToolExecutor) -> None:
    result = await executor.execute(
        "query_order", {"order_id": KNOWN_ORDER_ID, "phone": "13000000000"}
    )

    assert result.ok is False
    assert result.error_kind == "http_error"


# --------------------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("query_ticket", {"ticket_id": "TK00000000000"}),
        ("query_order", {"order_id": "999999999999999999"}),
    ],
)
async def test_unknown_id_maps_to_http_error(
    executor: ToolExecutor, tool: str, arguments: dict
) -> None:
    result = await executor.execute(tool, arguments)

    assert result.ok is False
    assert result.error_kind == "http_error"
    assert "404" in (result.detail or "")


async def test_delay_beyond_the_timeout_maps_to_timeout() -> None:
    executor = make_executor(delay_sec=0.2, timeout_sec=0.05)
    result = await executor.execute("query_order", {"order_id": KNOWN_ORDER_ID})
    await executor.client.aclose()

    assert result.ok is False
    assert result.error_kind == "timeout"


async def test_per_request_delay_header_overrides_the_default() -> None:
    app = create_mock_app()
    async with httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.get(
            "/orders", params={"order_id": KNOWN_ORDER_ID}, headers={"X-Mock-Delay": "0.05"}
        )
        query = await client.get("/orders", params={"order_id": KNOWN_ORDER_ID, "_delay": "0.05"})

    assert response.status_code == 200
    assert query.status_code == 200


# --------------------------------------------------------------------------------------
# Direct HTTP contract
# --------------------------------------------------------------------------------------


async def test_missing_parameters_and_health() -> None:
    app = create_mock_app()
    async with httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.ASGITransport(app=app)
    ) as client:
        tickets = await client.get("/tickets")
        orders = await client.get("/orders")
        health = await client.get("/healthz")
        missing = await client.get("/tickets", params={"ticket_id": "nope"})

    assert tickets.status_code == 422
    assert tickets.json()["error"] == "invalid_request"
    assert orders.status_code == 422
    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert missing.status_code == 404
    assert missing.json()["error"] == "not_found"


async def test_dataset_is_copied_not_shared() -> None:
    app = create_mock_app()
    async with httpx.AsyncClient(
        base_url=BASE_URL, transport=httpx.ASGITransport(app=app)
    ) as client:
        response = await client.get("/orders", params={"order_id": KNOWN_ORDER_ID})

    body = response.json()
    body["amount"] = "0.00"
    assert DEFAULT_DATASET["orders"][0]["amount"] == "1299.00"
