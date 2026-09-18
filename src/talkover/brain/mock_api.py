"""Mock business API server for demos and development (T3.6).

Serves the endpoint contract that :mod:`talkover.brain.tools` calls (DESIGN.md 6.6):

* ``GET /tickets`` with ``ticket_id`` and/or ``phone``
* ``GET /orders`` with ``order_id`` and/or ``phone``
* ``GET /healthz``

A lookup by id returns the single matching record as a JSON object; a lookup by phone
returns an object carrying the list of matching records (``tickets`` / ``orders``) so the
body stays a JSON object, which the LLM receives unchanged. An unknown id is ``404`` with
a JSON error body; a request with neither parameter is ``422``.

The fixture records are runtime data: their text fields are Chinese on purpose, matching
what a mainland-China customer-service backend would return.

An artificial delay exercises the 3 s timeout path of ``ToolExecutor``. It comes from
``delay_sec`` at app construction, and can be overridden per request with the
``X-Mock-Delay`` header or the ``_delay`` query parameter.

Run it with ``uv run talkover mock-api`` (see :func:`run_mock_api`).
"""

from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from typing import Any
from urllib.parse import urlparse

from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from talkover.brain.tools import ORDERS_PATH, TICKETS_PATH

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "Dataset",
    "add_cli_arguments",
    "create_mock_app",
    "main",
    "run_mock_api",
]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9100

#: One fixture dataset: ``{"tickets": [...], "orders": [...]}``.
Dataset = dict[str, list[dict[str, Any]]]

_TICKETS: list[dict[str, Any]] = [
    {
        "ticket_id": "TK20260917001",
        "phone": "13800138000",
        "status": "processing",
        "status_text": "处理中",
        "category": "售后维修",
        "subject": "耳机右声道无声",
        "created_at": "2026-09-15T10:24:31+08:00",
        "updated_at": "2026-09-17T09:02:10+08:00",
        "assignee": "客服 A032",
        "related_order_id": "202609150001234567",
        "latest_reply": "工程师已确认为硬件故障，预计 2 个工作日内寄出换新件。",
    },
    {
        "ticket_id": "TK20260916042",
        "phone": "13912345678",
        "status": "resolved",
        "status_text": "已解决",
        "category": "退款咨询",
        "subject": "退款未到账",
        "created_at": "2026-09-16T14:08:02+08:00",
        "updated_at": "2026-09-16T18:41:55+08:00",
        "assignee": "客服 A017",
        "related_order_id": "SO2026091600004821",
        "latest_reply": "退款 268.50 元已于 16 日退回原支付渠道，银行入账需 1-3 个工作日。",
    },
    {
        "ticket_id": "TK20260914118",
        "phone": "13800138000",
        "status": "pending_customer",
        "status_text": "待用户确认",
        "category": "物流异常",
        "subject": "包裹显示签收但未收到",
        "created_at": "2026-09-14T08:55:12+08:00",
        "updated_at": "2026-09-15T11:30:00+08:00",
        "assignee": "客服 A045",
        "related_order_id": "202609140009876543",
        "latest_reply": "快递公司已发起调查，请确认是否由门卫代收。",
    },
]

_ORDERS: list[dict[str, Any]] = [
    {
        "order_id": "202609150001234567",
        "phone": "13800138000",
        "status": "shipped",
        "status_text": "已发货",
        "amount": "1299.00",
        "currency": "CNY",
        "paid_at": "2026-09-15T10:02:47+08:00",
        "items": [
            {"name": "无线降噪耳机 Pro", "sku": "AUD-NC-PRO-BLK", "quantity": 1, "price": "1299.00"}
        ],
        "logistics": {
            "carrier": "顺丰速运",
            "carrier_code": "SF",
            "tracking_number": "SF1234567890123",
            "shipped_at": "2026-09-15T19:31:00+08:00",
            "status_text": "运输中，预计 9 月 18 日送达",
        },
    },
    {
        "order_id": "SO2026091600004821",
        "phone": "13912345678",
        "status": "refunded",
        "status_text": "已退款",
        "amount": "268.50",
        "currency": "CNY",
        "paid_at": "2026-09-16T09:12:05+08:00",
        "items": [
            {"name": "智能体脂秤", "sku": "HOM-SCL-002", "quantity": 1, "price": "199.00"},
            {"name": "运动手环表带", "sku": "ACC-BND-017", "quantity": 1, "price": "69.50"},
        ],
        "logistics": {
            "carrier": "中通快递",
            "carrier_code": "ZTO",
            "tracking_number": "ZT78945612303",
            "shipped_at": "2026-09-16T12:40:00+08:00",
            "status_text": "已退回仓库",
        },
        "refund": {
            "amount": "268.50",
            "currency": "CNY",
            "refunded_at": "2026-09-16T18:40:11+08:00",
        },
    },
    {
        "order_id": "202609140009876543",
        "phone": "13800138000",
        "status": "delivered",
        "status_text": "已签收",
        "amount": "89.90",
        "currency": "CNY",
        "paid_at": "2026-09-14T08:31:20+08:00",
        "items": [
            {"name": "USB-C 快充线 2m", "sku": "ACC-CBL-2M", "quantity": 2, "price": "44.95"}
        ],
        "logistics": {
            "carrier": "京东物流",
            "carrier_code": "JD",
            "tracking_number": "JDVA15987654321",
            "shipped_at": "2026-09-14T15:02:00+08:00",
            "delivered_at": "2026-09-15T10:12:44+08:00",
            "status_text": "已由本人签收",
        },
    },
    {
        "order_id": "20260917000112233445",
        "phone": "15901234567",
        "status": "pending_payment",
        "status_text": "待支付",
        "amount": "3599.00",
        "currency": "CNY",
        "paid_at": None,
        "items": [
            {"name": "折叠屏手机 X5", "sku": "PHN-X5-256-GRY", "quantity": 1, "price": "3599.00"}
        ],
        "logistics": None,
    },
]

#: The dataset served when the caller passes no ``dataset``.
DEFAULT_DATASET: Dataset = {"tickets": _TICKETS, "orders": _ORDERS}


def _error(status_code: int, error: str, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"error": error, "detail": detail})


def _request_delay(request: Request, default: float) -> float:
    """The delay for this request: ``X-Mock-Delay`` header, ``_delay`` query, or default."""
    for raw in (request.headers.get("x-mock-delay"), request.query_params.get("_delay")):
        if raw is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if value >= 0:
            return value
    return default


def create_mock_app(*, delay_sec: float = 0.0, dataset: Dataset | None = None) -> FastAPI:
    """Build the mock business API.

    `delay_sec` is slept before every lookup response, so a value above
    `brain.business_api.timeout_sec` exercises the executor's `timeout` path.
    `dataset` replaces :data:`DEFAULT_DATASET`; it is deep-copied, so the app never hands
    out references into the caller's fixtures.
    """
    data: Dataset = deepcopy(dataset if dataset is not None else DEFAULT_DATASET)
    tickets = data.get("tickets", [])
    orders = data.get("orders", [])

    app = FastAPI(
        title="Talkover mock business API",
        description="Fixture ticket and order lookups for Brain development and demos.",
        version="1",
    )

    async def _lookup(
        request: Request,
        records: list[dict[str, Any]],
        primary: str,
        identifier: str | None,
        phone: str | None,
        list_key: str,
    ) -> JSONResponse:
        identifier = identifier.strip() if identifier else None
        phone = phone.strip() if phone else None
        if identifier is None and phone is None:
            return _error(
                422,
                "invalid_request",
                f"at least one of '{primary}' or 'phone' is required",
            )
        delay = _request_delay(request, delay_sec)
        if delay > 0:
            await asyncio.sleep(delay)
        if identifier is not None:
            for record in records:
                if record.get(primary) != identifier:
                    continue
                if phone is not None and record.get("phone") != phone:
                    continue
                return JSONResponse(content=record)
            return _error(404, "not_found", f"no record for {primary}='{identifier}'")
        matches = [record for record in records if record.get("phone") == phone]
        return JSONResponse(content={"phone": phone, "count": len(matches), list_key: matches})

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return {"status": "ok", "tickets": len(tickets), "orders": len(orders)}

    @app.get(TICKETS_PATH)
    async def get_tickets(
        request: Request,
        ticket_id: str | None = Query(default=None),
        phone: str | None = Query(default=None),
    ) -> JSONResponse:
        return await _lookup(request, tickets, "ticket_id", ticket_id, phone, "tickets")

    @app.get(ORDERS_PATH)
    async def get_orders(
        request: Request,
        order_id: str | None = Query(default=None),
        phone: str | None = Query(default=None),
    ) -> JSONResponse:
        return await _lookup(request, orders, "order_id", order_id, phone, "orders")

    return app


def run_mock_api(host: str, port: int, delay_sec: float = 0.0) -> None:
    """Serve the mock API with uvicorn until interrupted."""
    import uvicorn

    uvicorn.run(create_mock_app(delay_sec=delay_sec), host=host, port=port, log_level="info")


# --------------------------------------------------------------------------------------
# CLI helpers (wired into `talkover mock-api` by cli.py)
# --------------------------------------------------------------------------------------


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Declare the `talkover mock-api` options on `parser`."""
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="config file whose brain.business_api.base_url supplies host and port",
    )
    parser.add_argument("--host", default=None, help=f"bind address (default {DEFAULT_HOST})")
    parser.add_argument(
        "--port", type=int, default=None, help=f"bind port (default {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--delay-sec",
        type=float,
        default=0.0,
        help="artificial delay before every lookup response, to exercise the timeout path",
    )
    return parser


def _address_from_config(path: str) -> tuple[str, int]:
    """Read `brain.business_api.base_url` from `path` and split it into host and port."""
    from talkover.config import ConfigError, load_brain_config, load_config

    try:
        # A standalone brain config; a full serve config is rejected and retried below.
        brain = load_brain_config(path)
    except ConfigError:
        brain = load_config(path).brain
    parsed = urlparse(brain.business_api.base_url)
    return (parsed.hostname or DEFAULT_HOST, parsed.port or DEFAULT_PORT)


def main(args: argparse.Namespace) -> int:
    """Entry point for the `mock-api` subcommand."""
    host, port = DEFAULT_HOST, DEFAULT_PORT
    config = getattr(args, "config", None)
    if config:
        host, port = _address_from_config(config)
    if getattr(args, "host", None):
        host = args.host
    if getattr(args, "port", None):
        port = args.port
    run_mock_api(host, port, getattr(args, "delay_sec", 0.0) or 0.0)
    return 0
