"""Command line entry point: talkover serve / check / bench / mock-api.

`serve` is the only subcommand that builds the whole process (T2.9). It stays thin on
purpose — the composition itself lives in `talkover.app.build_app`, and this module only
does what a command line has to do:

1. load and validate the YAML config (`talkover.config`);
2. build the device backend and **apply the upstream patches before anything imports
   `mcpmft.infer`** (`talkover.engine.patches`, which raises `PatchOrderError` otherwise);
3. build the engine, the ASR side channel and, through `build_app`, the Brain and the ASGI
   application;
4. log the startup summary and hand the application to uvicorn on `server.listen`.

Loading weights is not among them: `EngineSession.start` does that from the application's
lifespan, and `serve --check-only` returns before the lifespan runs, so the command can be
used to validate the wiring on a machine whose checkpoints are still downloading.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from talkover.config import TalkoverConfig

#: Log format of a serving process; uvicorn keeps its own for the access log.
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"

#: The bench harness `talkover bench` delegates to (T1.8), relative to this source tree.
BENCH_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_rtf.py"

LOGGER = logging.getLogger("talkover.serve")


def _load_bench_module() -> ModuleType:
    """Import `scripts/bench_rtf.py`, which lives outside the installed package (T1.8).

    The bench harness is a script, not part of `talkover`: it is long, it is only ever run
    by hand or from this subcommand, and tasks.md keeps it in `scripts/`. This subcommand
    is a thin front for it, so it loads the file by path, relative to this source tree.

    Raises:
        FileNotFoundError: when talkover was installed as a wheel without the repository
            next to it, in which case the script has to be run directly.
    """
    import importlib.util

    path = BENCH_SCRIPT
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: `talkover bench` runs the repository's bench script, so it "
            "needs a source checkout (an editable `uv sync`), not a wheel"
        )
    spec = importlib.util.spec_from_file_location("talkover_bench_rtf", path)
    if spec is None or spec.loader is None:  # pragma: no cover - only on a broken loader
        raise FileNotFoundError(f"{path} cannot be imported as a module")
    module = importlib.util.module_from_spec(spec)
    # Registered before it is executed: `dataclasses` resolves a class's own module out of
    # `sys.modules` while it processes the decorator, and the script defines dataclasses.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def _bench_argv(args: argparse.Namespace) -> list[str]:
    """Turn the parsed subcommand into the script's own argv."""
    argv = ["-c", args.config, "--units", str(args.units), "--engine", args.engine]
    argv += ["--seed", str(args.seed)]
    if args.out:
        argv += ["--out", args.out]
    if args.clip:
        argv += ["--clip", args.clip]
    if args.warmup:
        argv += ["--warmup", str(args.warmup)]
    if args.asr:
        argv.append("--asr")
    if args.talker_thread:
        argv.append("--talker-thread")
    return argv


def _run_bench(args: argparse.Namespace) -> int:
    """Run `talkover bench`: 0 on success, 2 on a bad config or an unusable device."""
    if not args.config:
        print(
            "talkover bench: a config file is required, e.g. -c configs/serve.example.yaml",
            file=sys.stderr,
        )
        return 2
    try:
        bench = _load_bench_module()
    except FileNotFoundError as exc:
        print(f"talkover bench: {exc}", file=sys.stderr)
        return 2
    return int(bench.main(_bench_argv(args)))


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


def _run_serve(args: argparse.Namespace) -> int:
    """Run `talkover serve`: 0 on a clean shutdown, 2 on a bad config or a bad startup."""
    import asyncio

    from talkover.config import ConfigError, load_config

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"talkover serve: {exc}", file=sys.stderr)
        return 2

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    try:
        return asyncio.run(_serve(config, check_only=bool(args.check_only)))
    except KeyboardInterrupt:  # pragma: no cover - interactive only
        return 0


async def _serve(config: TalkoverConfig, *, check_only: bool = False) -> int:
    """Build the process and serve it; with `check_only`, build it and return.

    Imports are local because they are expensive: `talkover.engine.session` pulls in torch
    and the upstream packages, which `talkover check`, `talkover mock-api` and
    `talkover serve --help` must not pay for.
    """
    from talkover.app import AsrSideChannel, build_app, startup_summary
    from talkover.engine.asr import AsrStream, create_asr
    from talkover.engine.backend import get_backend
    from talkover.engine.patches import PatchOrderError, apply_patches

    try:
        backend = get_backend(config.engine.device)
    except (TypeError, ValueError) as exc:
        print(f"talkover serve: {exc}", file=sys.stderr)
        return 2

    try:
        # Before `talkover.engine.session` — and therefore before anything imports
        # `mcpmft.infer` — so that no module captures an unpatched reference. Importing
        # that module applies the patches again against the device auto-detected at import
        # time, so the configured backend is re-applied right after it.
        apply_patches(backend)
        from talkover.engine.session import EngineSession

        apply_patches(backend)
    except PatchOrderError as exc:
        print(f"talkover serve: {exc}", file=sys.stderr)
        return 2

    LOGGER.info("%s", startup_summary(config))

    engine = EngineSession(config, backend=backend)
    asr = AsrSideChannel(lambda: AsrStream(create_asr(config.asr, backend)))
    composed = await build_app(config, engine, asr=asr)
    try:
        if check_only:
            LOGGER.info("talkover serve --check-only: the application was built; not serving")
            return 0
        import uvicorn

        server = uvicorn.Server(
            uvicorn.Config(
                composed.app,
                host=config.server.host,
                port=config.server.port,
                log_level="info",
                lifespan="on",
            )
        )
        await server.serve()
        return 0
    finally:
        # Idempotent: on the serving path the lifespan has already run both.
        await composed.aclose()
        await asr.aclose()
        await engine.stop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="talkover", description="Talkover inference gateway")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="start the Realtime compatible service")
    p_serve.add_argument("-c", "--config", default="configs/serve.example.yaml")
    p_serve.add_argument(
        "--check-only",
        action="store_true",
        help="build the application and exit, without binding a port or loading weights",
    )

    p_check = sub.add_parser("check", help="check environment, dependencies and GPU memory budget")
    p_check.add_argument("-c", "--config", default=None)

    # The flags are declared here rather than by importing the script, because importing
    # it pulls in torch and the upstream packages, and `talkover --help` must stay cheap.
    # `scripts/bench_rtf.py --help` is the full set; these are the ones worth a subcommand.
    p_bench = sub.add_parser(
        "bench",
        help="measure per-unit processing time (T1.8)",
        description=(
            "Feed a fixed clip through the engine one 1 s unit at a time, print the "
            "per-stage table and write the JSON dump. Run scripts/bench_rtf.py directly "
            "for the remaining options."
        ),
    )
    p_bench.add_argument("-c", "--config", default=None)
    p_bench.add_argument(
        "--out", default=None, help="JSON dump path; the default is bench_rtf.<backend>.json"
    )
    p_bench.add_argument("--units", type=int, default=5, help="how many 1 s units to feed")
    p_bench.add_argument(
        "--warmup", type=int, default=0, help="units fed before measurement starts"
    )
    p_bench.add_argument(
        "--engine",
        choices=("real", "fake"),
        default="real",
        help="'fake' runs the harness with no weights and no device",
    )
    p_bench.add_argument("--clip", default=None, help="a 16 kHz mono pcm16 wav, or 'synthetic'")
    p_bench.add_argument("--seed", type=int, default=20260918)
    p_bench.add_argument(
        "--asr", action="store_true", help="also time the ASR side channel on every unit"
    )
    p_bench.add_argument(
        "--talker-thread",
        action="store_true",
        help="run the Talker on its own thread (T1.5); needs engine.ref_audio_path",
    )

    p_bench.set_defaults(func=_run_bench)
    p_serve.set_defaults(func=_run_serve)
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
