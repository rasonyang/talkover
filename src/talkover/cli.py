"""Command line entry point: talkover serve / check / bench / mock-api."""

from __future__ import annotations

import argparse
import sys


def _not_implemented(args: argparse.Namespace) -> int:
    print(f"talkover {args.command}: not implemented", file=sys.stderr)
    return 2


def _run_check(args: argparse.Namespace) -> int:
    """Run `talkover check`: 0 when nothing failed, 1 on failures, 2 on a bad config."""
    from talkover.check import format_report, run_checks
    from talkover.config import ConfigError, load_config

    if not args.config:
        print(
            "talkover check: a config file is required, e.g. -c configs/serve.example.yaml",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"talkover check: {exc}", file=sys.stderr)
        return 2
    results = run_checks(config)
    print(format_report(results))
    return 1 if any(r.failed for r in results) else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talkover", description="Talkover inference gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="start the Realtime compatible service")
    p_serve.add_argument("-c", "--config", default="configs/serve.example.yaml")

    p_check = sub.add_parser("check", help="check environment, dependencies and GPU memory budget")
    p_check.add_argument("-c", "--config", default=None)

    p_bench = sub.add_parser("bench", help="measure per-unit processing time")
    p_bench.add_argument("-c", "--config", default=None)

    for p in (p_serve, p_bench):
        p.set_defaults(func=_not_implemented)
    p_check.set_defaults(func=_run_check)

    # Imported here rather than at module level: mock_api pulls in fastapi, which no other
    # subcommand needs, and this keeps `import talkover.cli` cheap.
    from talkover.brain import mock_api

    p_mock = sub.add_parser("mock-api", help="serve the mock business API for Brain demos")
    mock_api.add_cli_arguments(p_mock)
    p_mock.set_defaults(func=mock_api.main)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
