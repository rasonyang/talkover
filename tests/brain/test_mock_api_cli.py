"""`talkover mock-api` argument parsing and dispatch (no server is ever started)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from talkover.brain import mock_api
from talkover.cli import build_parser, main

CONFIG = str(Path(__file__).resolve().parents[2] / "configs" / "brain.business.example.yaml")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, float]]:
    """Replace `run_mock_api` so `mock_api.main` records its address instead of serving."""
    recorded: list[tuple[str, int, float]] = []

    def fake_run(host: str, port: int, delay_sec: float = 0.0) -> None:
        recorded.append((host, port, delay_sec))

    monkeypatch.setattr(mock_api, "run_mock_api", fake_run)
    return recorded


def test_parser_registers_mock_api() -> None:
    args = build_parser().parse_args(
        ["mock-api", "--host", "127.0.0.1", "--port", "9123", "--delay-sec", "0.1"]
    )
    assert args.command == "mock-api"
    assert args.func is mock_api.main
    assert (args.host, args.port, args.delay_sec) == ("127.0.0.1", 9123, 0.1)
    assert args.config is None


def test_parser_defaults_to_none() -> None:
    args = build_parser().parse_args(["mock-api"])
    assert (args.config, args.host, args.port, args.delay_sec) == (None, None, None, 0.0)


def test_config_short_option_is_accepted() -> None:
    """`-c` belongs to the subparser only; the top-level parser declares no conflicting flag."""
    args = build_parser().parse_args(["mock-api", "-c", CONFIG])
    assert args.config == CONFIG


def test_main_resolves_host_and_port_from_config(calls: list[tuple[str, int, float]]) -> None:
    assert main(["mock-api", "-c", CONFIG]) == 0
    assert calls == [("127.0.0.1", 9100, 0.0)]


def test_cli_flags_override_the_config(calls: list[tuple[str, int, float]]) -> None:
    assert main(["mock-api", "-c", CONFIG, "--port", "9123", "--delay-sec", "0.1"]) == 0
    assert calls == [("127.0.0.1", 9123, 0.1)]


def test_main_without_config_uses_module_defaults(calls: list[tuple[str, int, float]]) -> None:
    assert main(["mock-api"]) == 0
    assert calls == [(mock_api.DEFAULT_HOST, mock_api.DEFAULT_PORT, 0.0)]


def test_return_value_is_the_exit_code(calls: list[tuple[str, int, float]]) -> None:
    """`main` forwards the subcommand's int return, which the console script exits with."""
    parser = build_parser()
    args = parser.parse_args(["mock-api", "--port", "9123"])
    assert isinstance(args.func(args), int)
    assert isinstance(args, argparse.Namespace)
