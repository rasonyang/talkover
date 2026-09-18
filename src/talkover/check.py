"""`talkover check`: report whether this machine can run the service, without loading it.

Every check is static. Nothing here loads model weights, builds a model, opens a socket or
calls an API: the command has to be usable on a machine where the weights are still
downloading and no credentials are exported. Importability is probed with
:func:`importlib.util.find_spec`, which resolves a module without executing it, so neither
the upstream packages nor an ASR library is actually imported.

Checks, in report order (DESIGN.md sections 8 and 9):

* `python` — the interpreter must be 3.12 (DESIGN.md 8).
* `torch` — version inside `>=2.6,<2.7` (upstream `mcpmft` constraint).
* `device` — `engine.device` reached through
  :func:`talkover.engine.backend.get_backend` and its ``is_available()``.
* `upstream.mcpmft` / `upstream.gander_runtime` — importable path dependencies.
* `upstream.commit` — the upstream checkout's `HEAD` against :data:`UPSTREAM_PINNED_COMMIT`.
* `engine.base_model` / `engine.thinker_checkpoint` / `engine.talker_checkpoint` — the
  configured paths exist.
* `engine.token2wav_dir` — the token2wav assets exist, either at the configured path or at
  `<engine.base_model>/assets/token2wav`. A missing directory is a warning, not a failure:
  the Thinker still runs, only the Talker stays silent. The line also echoes
  `engine.token2wav_timesteps`, the vocoder's flow-matching step count (DESIGN.md 4.4).
* `asr.<backend>` — the resolved ASR backend's library is installed.
* `brain.llm.api_key` — non-empty after `${ENV_VAR}` expansion.
* `memory` — the static estimate from :mod:`talkover.engine.memory`.

A `borderline` memory verdict is a warning, `over_budget` is a failure. The command exits
0 when no check failed (warnings do not change the exit code) and 1 otherwise; a config
file that cannot be loaded exits 2.
"""

from __future__ import annotations

import importlib.machinery
import importlib.metadata
import importlib.util
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from talkover.config import TalkoverConfig
from talkover.engine.memory import Verdict, estimate_memory, format_estimate

__all__ = [
    "ASR_LIBRARIES",
    "TOKEN2WAV_SUBDIR",
    "TORCH_MAX_EXCLUSIVE",
    "TORCH_MIN",
    "UPSTREAM_PINNED_COMMIT",
    "CheckResult",
    "Status",
    "format_report",
    "run_checks",
]

UPSTREAM_PINNED_COMMIT = "cf43838"
"""Pinned upstream `Omni-Interaction-Agent` commit.

This mirrors the pin recorded in `docs/mps-porting-notes.md`; upgrading upstream means
changing both (and re-running `uv sync`). `tests/protocol/test_check.py` asserts the two
stay in sync.
"""

PYTHON_REQUIRED = (3, 12)
TORCH_MIN = (2, 6)
TORCH_MAX_EXCLUSIVE = (2, 7)

ASR_LIBRARIES = {
    "mlx_whisper": "mlx_whisper",
    "faster_whisper": "faster_whisper",
}
"""Resolved `asr.backend` name -> the library module name that must be importable."""

TOKEN2WAV_SUBDIR = ("assets", "token2wav")
"""Where the token2wav assets sit inside the base model when `engine.token2wav_dir` is null.

It mirrors `talkover.engine.session._TOKEN2WAV_SUBDIR`, which is duplicated rather than
imported because importing that module applies the upstream monkeypatches.
"""

_GIB = 1024**3


class Status(StrEnum):
    """Outcome of a single check."""

    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One line of the report: what was checked, how it went and why."""

    name: str
    status: Status
    detail: str

    @property
    def failed(self) -> bool:
        return self.status is Status.FAIL


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Status.OK, detail)


def _warn(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Status.WARN, detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name, Status.FAIL, detail)


# --------------------------------------------------------------------------------------
# Individual checks
# --------------------------------------------------------------------------------------


def _check_python() -> CheckResult:
    found = sys.version_info[:2]
    want = ".".join(str(p) for p in PYTHON_REQUIRED)
    detail = f"{sys.version.split()[0]} at {sys.executable}"
    if found != PYTHON_REQUIRED:
        return _fail("python", f"{detail}; Talkover requires Python {want} (DESIGN.md 8)")
    return _ok("python", detail)


def _parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in text.split("+")[0].split("."):
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _check_torch() -> CheckResult:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return _fail("torch", "not installed; run `uv sync --extra dev --extra mps`")
    parsed = _parse_version(version)[:2]
    want = ">=2.6,<2.7"
    if not parsed:
        return _warn("torch", f"{version}; cannot parse the version, expected {want}")
    if not (TORCH_MIN <= parsed < TORCH_MAX_EXCLUSIVE):
        return _fail("torch", f"{version}; upstream requires {want}")
    return _ok("torch", version)


def _check_device(config: TalkoverConfig) -> CheckResult:
    device = config.engine.device
    try:
        from talkover.engine.backend import get_backend

        backend = get_backend(device)
    except Exception as exc:  # noqa: BLE001 - any import/parse failure is a check failure
        return _fail("device", f"{device}: cannot build a backend: {exc}")
    try:
        available = backend.is_available()
    except Exception as exc:  # noqa: BLE001 - a broken driver must not abort the report
        return _fail("device", f"{device}: availability probe raised {type(exc).__name__}: {exc}")
    if not available:
        return _fail("device", f"{device}: backend {backend.name!r} reports it is unavailable")
    return _ok("device", f"{device}: backend {backend.name!r} is available")


def _find_spec(module: str) -> importlib.machinery.ModuleSpec | None:
    """`find_spec` that never raises; a missing or broken parent means "not importable"."""
    try:
        return importlib.util.find_spec(module)
    except (ImportError, ValueError):
        return None


def _upstream_origin(module: str) -> Path | None:
    spec = _find_spec(module)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _check_upstream_package(module: str) -> CheckResult:
    origin = _upstream_origin(module)
    if origin is None:
        return _fail(
            f"upstream.{module}",
            "not importable; the path dependency needs ../Omni-Interaction-Agent next to "
            "this repository, then `uv sync`",
        )
    return _ok(f"upstream.{module}", str(origin))


def _git_head(directory: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def _check_upstream_commit() -> CheckResult:
    name = "upstream.commit"
    origin = _upstream_origin("mcpmft") or _upstream_origin("gander_runtime")
    if origin is None:
        return _fail(name, "upstream packages are not importable; cannot read their commit")
    head = _git_head(origin)
    if head is None:
        return _warn(
            name,
            f"{origin} is not a readable git checkout; cannot verify the pin "
            f"{UPSTREAM_PINNED_COMMIT}",
        )
    if not (head.startswith(UPSTREAM_PINNED_COMMIT) or UPSTREAM_PINNED_COMMIT.startswith(head)):
        return _fail(
            name,
            f"upstream HEAD is {head}, expected the pinned {UPSTREAM_PINNED_COMMIT} "
            "(docs/mps-porting-notes.md)",
        )
    return _ok(name, f"{head} matches the pin in docs/mps-porting-notes.md")


def _check_path(name: str, value: str) -> CheckResult:
    if not value:
        return _fail(name, "not configured (empty path)")
    path = Path(value).expanduser()
    if not path.exists():
        return _fail(name, f"{path} does not exist")
    if not path.is_dir():
        return _warn(name, f"{path} exists but is not a directory")
    return _ok(name, str(path))


def _check_token2wav(config: TalkoverConfig) -> CheckResult:
    """The token2wav assets the Talker needs, without importing the engine.

    `talkover.engine.session.default_token2wav_dir` resolves the same two cases, but
    importing that module applies the upstream monkeypatches, which `check` must not do.
    """
    name = "engine.token2wav_dir"
    engine = config.engine
    if engine.token2wav_dir:
        path = Path(engine.token2wav_dir).expanduser()
        source = "engine.token2wav_dir"
    elif engine.base_model:
        path = Path(engine.base_model).expanduser().joinpath(*TOKEN2WAV_SUBDIR)
        source = "<engine.base_model>/" + "/".join(TOKEN2WAV_SUBDIR)
    else:
        return _warn(name, "engine.base_model is empty; cannot resolve the token2wav assets")
    steps = engine.token2wav_timesteps
    if not path.is_dir():
        return _warn(
            name,
            f"{path} ({source}) does not exist; the Thinker still runs but the Talker "
            "produces no audio",
        )
    return _ok(name, f"{path} ({source}), {steps} flow-matching steps")


def _check_asr(config: TalkoverConfig) -> CheckResult:
    backend = config.asr.backend
    name = f"asr.{backend}"
    module = ASR_LIBRARIES.get(backend)
    if module is None:
        return _warn(name, f"no importable library is defined for asr.backend {backend!r}")
    if _find_spec(module) is None:
        extra = "mps" if backend == "mlx_whisper" else "cuda"
        return _fail(
            name,
            f"library {module!r} is not installed; run `uv sync --extra dev --extra {extra}`",
        )
    return _ok(
        name,
        f"library {module!r} is installed (device {config.asr.device}, "
        f"compute_type {config.asr.compute_type})",
    )


def _check_llm_key(config: TalkoverConfig) -> CheckResult:
    name = "brain.llm.api_key"
    llm = config.brain.llm
    if llm.api_key:
        return _ok(name, f"set for kind {llm.kind!r}, model {llm.model!r}")
    detail = (
        f"empty after ${{ENV_VAR}} expansion for kind {llm.kind!r} "
        f"at {llm.base_url}; export the key the config refers to"
    )
    if llm.kind == "openai_compat":
        return _warn(name, f"{detail} (a local OpenAI-compatible server may not need one)")
    return _fail(name, detail)


def _fmt_bytes(num_bytes: int) -> str:
    return f"{num_bytes / _GIB:.2f} GiB"


def _check_memory(config: TalkoverConfig) -> CheckResult:
    estimate = estimate_memory(config)
    summary = (
        f"{_fmt_bytes(estimate.total_bytes)} of {_fmt_bytes(estimate.budget_bytes)} "
        f"({estimate.utilization:.0%} used), verdict {estimate.verdict.value}"
    )
    detail = f"{summary}\n{format_estimate(estimate)}"
    if estimate.verdict is Verdict.OVER_BUDGET:
        return _fail("memory", detail)
    if estimate.verdict is Verdict.BORDERLINE:
        return _warn("memory", detail)
    return _ok("memory", detail)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def run_checks(config: TalkoverConfig) -> list[CheckResult]:
    """Run every environment check for `config` and return one result per check.

    The function never raises for a failing environment: a broken dependency becomes a
    `fail` result. Only a programming error propagates.
    """
    results = [_check_python(), _check_torch(), _check_device(config)]
    results.append(_check_upstream_package("mcpmft"))
    results.append(_check_upstream_package("gander_runtime"))
    results.append(_check_upstream_commit())
    results.append(_check_path("engine.base_model", config.engine.base_model))
    results.append(_check_path("engine.thinker_checkpoint", config.engine.thinker_checkpoint))
    results.append(_check_path("engine.talker_checkpoint", config.engine.talker_checkpoint))
    results.append(_check_token2wav(config))
    results.append(_check_asr(config))
    results.append(_check_llm_key(config))
    results.append(_check_memory(config))
    return results


def format_report(results: list[CheckResult]) -> str:
    """Render results as plain text: one line per check, then the problems, if any.

    A multi-line `detail` keeps its first line on the check's own line; the rest is
    indented underneath it.
    """
    if not results:
        return "no checks were run"
    width = max(len(r.name) for r in results)
    lines: list[str] = []
    for result in results:
        head, _, rest = result.detail.partition("\n")
        lines.append(f"[{result.status.value.upper():<4}] {result.name.ljust(width)}  {head}")
        for extra in rest.splitlines():
            lines.append(f"{' ' * (width + 9)}{extra}")

    failures = [r for r in results if r.status is Status.FAIL]
    warnings = [r for r in results if r.status is Status.WARN]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} check(s) failed:")
        for result in failures:
            lines.append(f"  - {result.name}: {result.detail.partition(chr(10))[0]}")
    if warnings:
        lines.append(f"{len(warnings)} warning(s):")
        for result in warnings:
            lines.append(f"  - {result.name}: {result.detail.partition(chr(10))[0]}")
    if not failures:
        lines.append("all checks passed" if not warnings else "no failures; warnings only")
    return "\n".join(lines)
