"""`talkover serve` wiring (T2.9), without a port and without weights.

Marked `cpu` because it is the one `serve` case that reaches the device layer: it builds
the backend, applies the upstream patches and constructs `EngineSession`. It stops there —
`--check-only` returns before the application's lifespan runs, so no checkpoint, no ASR
model and no socket is touched. The rest of the composition is covered GPU-free in
`tests/protocol/test_app.py`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from talkover.cli import main

pytestmark = pytest.mark.cpu

CONFIG = """
server:
  listen: 127.0.0.1:18123
  api_key: dry-run-key

engine:
  device: cpu
  dtype: float32
  base_model: {base_model}
  thinker_checkpoint: {base_model}/thinker
  talker_checkpoint: {base_model}/talker

asr:
  backend: faster_whisper
  device: cpu
  compute_type: int8

brain:
  llm:
    api_key: sk-dry-run
"""


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / "serve.yaml"
    path.write_text(CONFIG.format(base_model=tmp_path / "models"), encoding="utf-8")
    return path


def test_check_only_builds_the_application_without_binding_or_loading(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="talkover.serve"):
        assert main(["serve", "-c", str(write_config(tmp_path)), "--check-only"]) == 0
    summary = "\n".join(record.getMessage() for record in caplog.records)
    # The startup report names the device, the checkpoints and the memory estimate.
    assert "cpu (float32)" in summary
    assert "thinker" in summary
    assert "memory estimate for device cpu" in summary
    assert "not serving" in summary


def test_an_unsupported_device_exits_two(tmp_path: Path) -> None:
    path = tmp_path / "serve.yaml"
    path.write_text("engine:\n  device: tpu\n", encoding="utf-8")
    assert main(["serve", "-c", str(path)]) == 2


def test_a_missing_config_file_exits_two(tmp_path: Path) -> None:
    assert main(["serve", "-c", str(tmp_path / "absent.yaml")]) == 2
