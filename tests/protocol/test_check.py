"""Tests for `talkover check` (T0.3). GPU-free: no weights, no network, no credentials.

Every environment probe is monkeypatched, so the suite result never depends on what is
installed on the machine running it.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from talkover import check as check_mod
from talkover.check import (
    UPSTREAM_PINNED_COMMIT,
    CheckResult,
    Status,
    format_report,
    run_checks,
)
from talkover.cli import main
from talkover.config import load_config
from talkover.engine import memory as memory_mod

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIG_TEMPLATE = """
server:
  listen: 127.0.0.1:8000
  api_key: test-key

engine:
  device: {device}
  dtype: bfloat16
  init_vision: false
  base_model: {base_model}
  thinker_checkpoint: {thinker}
  talker_checkpoint: {talker}
  context_max_units: 128
  media_mode: voice
  token2wav_dir: {token2wav_dir}

asr:
  backend: auto
  model: large-v3-turbo
  device: {asr_device}
  compute_type: auto

brain:
  provider: business
  llm:
    kind: {llm_kind}
    base_url: https://api.deepseek.com/anthropic
    api_key: ${{TALKOVER_TEST_KEY}}
    model: deepseek-flash
"""


class _FakeBackend:
    def __init__(self, name: str, available: bool = True) -> None:
        self.name = name
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _write_config(
    tmp_path: Path,
    *,
    device: str = "mps",
    asr_device: str = "auto",
    llm_kind: str = "anthropic",
    missing_talker: bool = False,
    missing_token2wav: bool = False,
    token2wav_dir: str = "null",
) -> Path:
    for name in ("base", "thinker", "talker"):
        (tmp_path / name).mkdir(exist_ok=True)
    if not missing_token2wav:
        (tmp_path / "base" / "assets" / "token2wav").mkdir(parents=True, exist_ok=True)
    talker = tmp_path / ("talker_missing" if missing_talker else "talker")
    path = tmp_path / "serve.yaml"
    path.write_text(
        CONFIG_TEMPLATE.format(
            device=device,
            base_model=tmp_path / "base",
            thinker=tmp_path / "thinker",
            talker=talker,
            asr_device=asr_device,
            llm_kind=llm_kind,
            token2wav_dir=token2wav_dir,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def healthy_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Make every external probe report a correctly prepared machine."""
    upstream = tmp_path / "upstream" / "minicpm_ft" / "mcpmft"
    upstream.mkdir(parents=True)

    monkeypatch.setenv("TALKOVER_TEST_KEY", "sk-test")
    monkeypatch.setattr(check_mod.importlib.metadata, "version", lambda _name: "2.6.0")
    monkeypatch.setattr(check_mod, "_upstream_origin", lambda _module: upstream)
    monkeypatch.setattr(check_mod, "_git_head", lambda _dir: UPSTREAM_PINNED_COMMIT)
    monkeypatch.setattr(check_mod, "_find_spec", lambda _module: object())
    monkeypatch.setattr("talkover.engine.backend.get_backend", lambda device: _FakeBackend(device))
    # The MPS budget is the host's physical memory, so without this the result would
    # depend on the RAM of the machine running the suite (a 16 GB CI runner reports
    # over_budget where the 128 GB dev machine does not). Pin it to the dev machine.
    monkeypatch.setattr(memory_mod, "_host_memory_bytes", lambda: 128 * 1024**3)
    return upstream


def _by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def test_pin_matches_porting_notes() -> None:
    notes = (REPO_ROOT / "docs" / "mps-porting-notes.md").read_text(encoding="utf-8")
    assert f"`{UPSTREAM_PINNED_COMMIT}`" in notes


def test_all_ok(healthy_env, tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path))
    results = run_checks(config)
    by_name = _by_name(results)

    assert by_name["torch"].status is Status.OK
    assert by_name["device"].status is Status.OK
    assert by_name["upstream.mcpmft"].status is Status.OK
    assert by_name["upstream.commit"].status is Status.OK
    assert by_name["engine.talker_checkpoint"].status is Status.OK
    assert by_name["engine.token2wav_dir"].status is Status.OK
    assert by_name["asr.mlx_whisper"].status is Status.OK
    assert by_name["brain.llm.api_key"].status is Status.OK
    assert by_name["memory"].status is not Status.FAIL
    assert not [r for r in results if r.status is Status.FAIL]
    assert "all checks passed" in format_report(results) or "warnings only" in format_report(
        results
    )


def test_missing_checkpoint_fails(healthy_env, tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, missing_talker=True))
    results = run_checks(config)
    result = _by_name(results)["engine.talker_checkpoint"]
    assert result.status is Status.FAIL
    assert "does not exist" in result.detail
    assert "engine.talker_checkpoint" in format_report(results)


def test_missing_token2wav_assets_warn(healthy_env, tmp_path: Path) -> None:
    # The Thinker still runs without them, so this must not fail the report.
    config = load_config(_write_config(tmp_path, missing_token2wav=True))
    results = run_checks(config)
    result = _by_name(results)["engine.token2wav_dir"]

    assert result.status is Status.WARN
    assert "assets/token2wav" in result.detail
    assert not [r for r in results if r.status is Status.FAIL]


def test_configured_token2wav_dir_is_checked(healthy_env, tmp_path: Path) -> None:
    elsewhere = tmp_path / "voices" / "token2wav"
    config = load_config(_write_config(tmp_path, token2wav_dir=str(elsewhere)))
    assert _by_name(run_checks(config))["engine.token2wav_dir"].status is Status.WARN

    elsewhere.mkdir(parents=True)
    result = _by_name(run_checks(config))["engine.token2wav_dir"]

    assert result.status is Status.OK
    assert str(elsewhere) in result.detail


def test_unset_api_key_fails(healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("TALKOVER_TEST_KEY", raising=False)
    config = load_config(_write_config(tmp_path))
    result = _by_name(run_checks(config))["brain.llm.api_key"]
    assert result.status is Status.FAIL
    assert "empty" in result.detail


def test_unset_api_key_is_warning_for_openai_compat(
    healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("TALKOVER_TEST_KEY", raising=False)
    config = load_config(_write_config(tmp_path, llm_kind="openai_compat"))
    results = run_checks(config)
    assert _by_name(results)["brain.llm.api_key"].status is Status.WARN
    assert not [r for r in results if r.status is Status.FAIL]


def test_over_budget_memory_fails(healthy_env, tmp_path: Path) -> None:
    # CUDA with ASR on the same card exceeds the 24 GiB A10 budget (DESIGN.md 4.5).
    config = load_config(_write_config(tmp_path, device="cuda:0"))
    results = run_checks(config)
    memory = _by_name(results)["memory"]
    assert memory.status is Status.FAIL
    assert "over_budget" in memory.detail
    assert "memory estimate for device cuda:0" in memory.detail


def test_borderline_memory_warns(healthy_env, tmp_path: Path) -> None:
    config = load_config(_write_config(tmp_path, device="cuda:0", asr_device="cpu"))
    memory = _by_name(run_checks(config))["memory"]
    assert memory.status is Status.WARN
    assert "borderline" in memory.detail


def test_unavailable_device_fails(
    healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "talkover.engine.backend.get_backend",
        lambda device: _FakeBackend(device, available=False),
    )
    config = load_config(_write_config(tmp_path))
    result = _by_name(run_checks(config))["device"]
    assert result.status is Status.FAIL
    assert "unavailable" in result.detail


def test_missing_asr_library_fails(
    healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(check_mod, "_find_spec", lambda module: None)
    config = load_config(_write_config(tmp_path))
    results = _by_name(run_checks(config))
    assert results["asr.mlx_whisper"].status is Status.FAIL
    assert results["upstream.mcpmft"].status is Status.OK  # origin is patched separately


def test_upstream_not_a_git_checkout_warns(
    healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(check_mod, "_git_head", lambda _dir: None)
    config = load_config(_write_config(tmp_path))
    result = _by_name(run_checks(config))["upstream.commit"]
    assert result.status is Status.WARN
    assert "not a readable git checkout" in result.detail


def test_upstream_commit_mismatch_fails(
    healthy_env, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(check_mod, "_git_head", lambda _dir: "deadbee")
    config = load_config(_write_config(tmp_path))
    result = _by_name(run_checks(config))["upstream.commit"]
    assert result.status is Status.FAIL
    assert UPSTREAM_PINNED_COMMIT in result.detail


def test_git_head_returns_none_without_git(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", boom)
    assert check_mod._git_head(tmp_path) is None


def test_python_check_reports_current_interpreter() -> None:
    result = check_mod._check_python()
    expected = Status.OK if sys.version_info[:2] == (3, 12) else Status.FAIL
    assert result.status is expected


def test_torch_version_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(check_mod.importlib.metadata, "version", lambda _name: "2.8.0")
    assert check_mod._check_torch().status is Status.FAIL


def test_torch_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_name: str) -> str:
        raise check_mod.importlib.metadata.PackageNotFoundError("torch")

    monkeypatch.setattr(check_mod.importlib.metadata, "version", missing)
    assert check_mod._check_torch().status is Status.FAIL


def test_format_report_indents_multiline_detail() -> None:
    results = [CheckResult("memory", Status.WARN, "summary line\nsecond line")]
    report = format_report(results)
    assert "summary line" in report
    assert "  second line" in report
    assert "1 warning(s):" in report


def test_cli_exit_code_zero(healthy_env, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    path = _write_config(tmp_path)
    assert main(["check", "-c", str(path)]) == 0
    assert "[OK  ]" in capsys.readouterr().out


def test_cli_exit_code_one_on_failure(
    healthy_env, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    path = _write_config(tmp_path, missing_talker=True)
    assert main(["check", "-c", str(path)]) == 1
    assert "check(s) failed" in capsys.readouterr().out


def test_cli_exit_code_two_on_bad_config(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("engine:\n  device: tpu\n", encoding="utf-8")
    assert main(["check", "-c", str(bad)]) == 2
    assert "engine.device" in capsys.readouterr().err


def test_cli_exit_code_two_without_config(capsys: pytest.CaptureFixture) -> None:
    assert main(["check"]) == 2
    assert "config file is required" in capsys.readouterr().err
