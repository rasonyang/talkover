"""Tests for `scripts/bench_rtf.py`, the `talkover bench` subcommand and the timing hook.

Everything here is `cpu` except the last case: the harness is built so that the whole
pipeline — clip slicing, the `on_unit_timing` hook, the statistics, the table and the JSON
dump — runs against a fake engine with no weights and no device, which is what makes T1.8
testable while the checkpoints are still downloading.

Four layers:

- the statistics helpers (`percentile`, `summarize`, `budget_violations`);
- the clip loader, including the synthetic fallback and looping;
- `StageTimer` against `FakeBenchEngine` through `run_bench`, and against the real
  `EngineSession` with a stub upstream session, which is what proves the hook reports the
  upstream `cost_*` metrics as stages;
- `main` and `talkover bench` end to end, including the shape of the JSON dump.

The `mps` case runs the real command and skips while the weights are missing; it is the
one that produces the numbers T1.8 has to record in `docs/mps-porting-notes.md`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from engine.test_session import _missing_checkpoints  # the T1.4 weight-completeness probe

# The bench harness lives in `scripts/`, outside the package; this is the loader
# `talkover bench` itself uses, so importing it here covers that path too.
from talkover.cli import _load_bench_module
from talkover.cli import main as cli_main
from talkover.config import parse_config
from talkover.engine.backend import get_backend
from talkover.engine.protocol import UNIT_BYTES
from talkover.engine.session import STAGE_METRIC_KEYS, EngineSession, UnitTiming, stage_times

bench = _load_bench_module()

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_EXAMPLE = REPO_ROOT / "configs" / "serve.example.yaml"
FIXTURE_CLIP = REPO_ROOT / "tests" / "fixtures" / "zh_order_16k.wav"


def _config_file(tmp_path: Path, device: str = "cpu") -> Path:
    """A minimal serve config on disk, which is what the CLI takes."""
    path = tmp_path / "bench.yaml"
    path.write_text(f"engine:\n  device: {device}\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_percentile_uses_nearest_rank() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert bench.percentile(values, 50) == 0.3
    assert bench.percentile(values, 95) == 0.5
    assert bench.percentile(values, 0) == 0.1
    assert bench.percentile(values, 100) == 0.5
    # Unsorted input is sorted first, and a single sample is every percentile.
    assert bench.percentile([0.4, 0.1, 0.9], 50) == 0.4
    assert bench.percentile([0.7], 95) == 0.7


@pytest.mark.cpu
def test_percentile_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty"):
        bench.percentile([], 50)


@pytest.mark.cpu
def test_summarize_reports_count_mean_and_percentiles() -> None:
    stats = bench.summarize([0.1, 0.2, 0.3, 0.4])
    assert stats["count"] == 4
    assert stats["mean"] == pytest.approx(0.25)
    assert stats["p50"] == pytest.approx(0.2)
    assert stats["p95"] == pytest.approx(0.4)
    assert stats["min"] == pytest.approx(0.1)
    assert stats["max"] == pytest.approx(0.4)
    assert bench.summarize([]) == {"count": 0}


@pytest.mark.cpu
def test_budget_violations_reports_only_stages_over_their_budget() -> None:
    stats = {
        "unit": {"count": 3, "p95": 0.8},
        "thinker": {"count": 3, "p95": 0.61},
        "talker": {"count": 0},
    }
    violations = bench.budget_violations(stats)
    assert [item["stage"] for item in violations] == ["thinker"]
    assert violations[0]["budget_sec"] == pytest.approx(0.45)
    assert "budget" in bench.format_verdict({"violations": violations})
    assert "within" in bench.format_verdict({"violations": []})


# --------------------------------------------------------------------------------------
# The clip
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_load_clip_slices_the_bundled_fixture_into_whole_units() -> None:
    clip = bench.load_clip(str(FIXTURE_CLIP), 3, seed=1)
    assert clip.source == "file"
    assert clip.looped is False
    assert len(clip.units) == 3
    assert {len(unit) for unit in clip.units} == {UNIT_BYTES}
    described = clip.describe()
    assert described["sample_rate"] == 16000
    assert described["units"] == 3
    assert len(described["sha256"]) == 64


@pytest.mark.cpu
def test_load_clip_loops_a_short_clip_and_records_it() -> None:
    clip = bench.load_clip(bench.SYNTHETIC, 2, seed=7)
    longer = bench.load_clip(str(FIXTURE_CLIP), 12, seed=7)
    assert clip.source == bench.SYNTHETIC
    assert clip.path is None
    assert longer.looped is True
    assert len(longer.units) == 12
    # The fixture holds 5 whole units, so unit 6 is unit 1 again.
    assert longer.units[5] == longer.units[0]


@pytest.mark.cpu
def test_synthetic_clip_is_reproducible_from_the_seed() -> None:
    first = bench.load_clip(bench.SYNTHETIC, 2, seed=11)
    same = bench.load_clip(bench.SYNTHETIC, 2, seed=11)
    other = bench.load_clip(bench.SYNTHETIC, 2, seed=12)
    assert first.units == same.units
    assert first.units != other.units


@pytest.mark.cpu
def test_load_clip_rejects_a_wav_the_engine_cannot_take(tmp_path: Path) -> None:
    import wave

    path = tmp_path / "stereo.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00" * 4 * 16000)
    with pytest.raises(ValueError, match="mono pcm16"):
        bench.load_clip(str(path), 1, seed=1)

    with pytest.raises(ValueError, match="must be positive"):
        bench.load_clip(bench.SYNTHETIC, 0, seed=1)


# --------------------------------------------------------------------------------------
# The timing hook
# --------------------------------------------------------------------------------------


@dataclass
class _StubStepEvent:
    """The subset of the upstream `DuplexStepEvent` the timing hook reads."""

    index: int
    is_listen: bool = False
    text: str = "hi"
    end_of_turn: bool = False
    unit_id: int | None = 1
    generation_id: int = 0
    generated_token_ids: list[int] = field(default_factory=lambda: [11, 22])
    audio_waveform: Any = None
    metrics: dict[str, Any] = field(default_factory=dict)


class _StubLiveSession:
    """Returns one scripted step event per call, with upstream-shaped metrics."""

    def __init__(self) -> None:
        self.index = 0

    def _event(self) -> _StubStepEvent:
        self.index += 1
        return _StubStepEvent(
            index=self.index,
            metrics={
                "cost_llm": 0.30,
                "cost_tts_prep": 0.01,
                "cost_tts": 0.20,
                "cost_token2wav": 0.15,
                "cost_all": 0.66,
                "n_tts_tokens": 25,
                "talker": {"active": False},
            },
        )

    def feed_pcm16(self, data: bytes) -> list[_StubStepEvent]:
        return [self._event()]

    def flush_pending(self) -> list[_StubStepEvent]:
        return [self._event()]

    def close(self, **kwargs: Any) -> None:
        return None


@pytest.mark.cpu
def test_stage_times_maps_the_upstream_metric_keys() -> None:
    assert STAGE_METRIC_KEYS["thinker"] == ("cost_llm",)
    stages = stage_times(
        {"cost_llm": 0.3, "cost_tts": 0.2, "cost_token2wav": 0.1, "n_tokens": 7, "fake": True}
    )
    assert stages == {"thinker": 0.3, "talker": 0.2, "token2wav": 0.1}
    # Absent, non-numeric and boolean values never become a stage.
    assert stage_times({}) == {}
    assert stage_times(None) == {}
    assert stage_times({"cost_llm": "slow"}) == {}


@pytest.mark.cpu
async def test_engine_session_reports_one_timing_per_unit() -> None:
    """The hook is what the bench harness drives the real engine through (T1.8)."""
    timer = bench.StageTimer()
    session = EngineSession(
        parse_config({"engine": {"device": "cpu"}}),
        backend=get_backend("cpu"),
        session_factory=_StubLiveSession,
        on_unit_timing=timer.record,
    )
    await session.start()
    try:
        await session.feed_pcm16(b"\x00" * UNIT_BYTES)
        await session.feed_pcm16(b"\x00" * UNIT_BYTES)
        await session.flush_pending()
    finally:
        await session.stop()

    assert [timing.unit_index for timing in timer.timings] == [1, 2, 3]
    assert all(timing.source == "step" for timing in timer.timings)
    first = timer.timings[0]
    assert isinstance(first, UnitTiming)
    assert first.stages == {
        "thinker": 0.30,
        "talker_prep": 0.01,
        "talker": 0.20,
        "token2wav": 0.15,
    }
    assert first.text_token_ids == (11, 22)
    assert first.speak_token_count == 25
    assert first.wall_sec > 0.0  # measured around the call, not taken from the metrics

    stats = timer.stats()
    assert stats["thinker"]["count"] == 3
    assert stats["thinker"]["p95"] == pytest.approx(0.30)
    assert stats["step"]["count"] == 3
    assert "asr" not in stats  # the engine never measures the side channel


@pytest.mark.cpu
async def test_a_raising_callback_does_not_break_the_session() -> None:
    def explode(timing: UnitTiming) -> None:
        raise RuntimeError("bench callback is broken")

    session = EngineSession(
        parse_config({"engine": {"device": "cpu"}}),
        backend=get_backend("cpu"),
        session_factory=_StubLiveSession,
        on_unit_timing=explode,
    )
    await session.start()
    try:
        await session.feed_pcm16(b"\x00" * UNIT_BYTES)
        assert session.ready is True
    finally:
        await session.stop()


# --------------------------------------------------------------------------------------
# The harness, end to end with the fake engine
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
async def test_run_bench_collects_units_stages_and_events() -> None:
    timer = bench.StageTimer()
    engine = bench.FakeBenchEngine(seed=3, on_unit_timing=timer.record)
    clip = bench.load_clip(bench.SYNTHETIC, 3, seed=3)

    events = await bench.run_bench(clip=clip, engine=engine, timer=timer)

    # Three fed units plus the one the flush produced.
    assert [event.unit_index for event in events] == [1, 2, 3, 4]
    assert events[-1].end_of_turn is True
    assert [timing.unit_index for timing in timer.timings] == [1, 2, 3, 4]
    samples = timer.samples()
    assert len(samples["unit"]) == 3
    assert len(samples["flush"]) == 1
    assert len(samples["thinker"]) == 4
    assert engine.ready is False


@pytest.mark.cpu
async def test_warmup_units_are_dropped_from_the_measurement() -> None:
    timer = bench.StageTimer()
    engine = bench.FakeBenchEngine(seed=3, on_unit_timing=timer.record)
    clip = bench.load_clip(bench.SYNTHETIC, 3, seed=3)

    events = await bench.run_bench(clip=clip, engine=engine, timer=timer, warmup=1)

    assert min(event.unit_index for event in events) == 2
    assert min(timing.unit_index for timing in timer.timings) == 2
    assert len(timer.samples()["unit"]) == 2


@pytest.mark.cpu
def test_main_writes_a_dump_whose_schema_is_pinned(tmp_path: Path, capsys: Any) -> None:
    out = tmp_path / "bench.json"
    config = _config_file(tmp_path)

    code = bench.main(
        ["-c", str(config), "--engine", "fake", "--units", "3", "--out", str(out), "--seed", "5"]
    )

    assert code == 0
    printed = capsys.readouterr().out
    assert "engine=fake" in printed
    assert "thinker" in printed
    assert str(out) in printed

    report = json.loads(out.read_text(encoding="utf-8"))
    assert set(report) == {
        "schema",
        "generated_at",
        "argv",
        "engine",
        "warmup_units",
        "seed",
        "device",
        "clip",
        "config",
        "versions",
        "budget_sec",
        "stats",
        "violations",
        "units",
        "token_sequence",
    }
    assert report["schema"] == bench.SCHEMA_VERSION == "talkover.bench/1"
    assert report["engine"] == "fake"
    assert report["seed"]["seed"] == 5
    assert report["device"] == {"configured": "cpu", "backend": "cpu", "asr_backend": None}
    assert report["clip"]["units"] == 3
    assert report["versions"]["upstream_pinned_commit"]
    assert report["config"]["engine"]["device"] == "cpu"
    assert "api_key" not in json.dumps(report["config"])

    # Four units (three fed plus the flush), each with its stages and its metrics.
    assert [unit["unit_index"] for unit in report["units"]] == [1, 2, 3, 4]
    assert report["units"][1]["stages"]["thinker"] > 0
    assert report["units"][1]["source"] == "step"
    assert report["stats"]["unit"]["count"] == 3
    assert report["stats"]["thinker"]["count"] == 4

    sequence = report["token_sequence"]
    assert [item["unit_index"] for item in sequence] == [1, 2, 3, 4]
    assert all(isinstance(item["text_token_ids"], list) for item in sequence)
    assert sequence[1]["speak_token_count"] == 25
    # No timing in the diffable part: two backends must produce the same list.
    assert not any("wall_sec" in item or "stages" in item for item in sequence)


@pytest.mark.cpu
def test_two_runs_of_the_same_seed_emit_the_same_token_sequence(tmp_path: Path) -> None:
    """The M4 comparison: same clip, same seed, diff `token_sequence` across backends."""
    config = _config_file(tmp_path)
    dumps = []
    for name in ("one.json", "two.json"):
        out = tmp_path / name
        argv = ["-c", str(config), "--engine", "fake", "--units", "2", "--out", str(out)]
        assert bench.main(argv) == 0
        dumps.append(json.loads(out.read_text(encoding="utf-8")))

    assert dumps[0]["token_sequence"] == dumps[1]["token_sequence"]
    assert dumps[0]["clip"]["sha256"] == dumps[1]["clip"]["sha256"]


@pytest.mark.cpu
def test_a_fake_run_never_claims_the_configured_device(tmp_path: Path) -> None:
    """A fake run has to work on a box whose config file names a device it does not have."""
    out = tmp_path / "bench.json"
    config = _config_file(tmp_path, device="cuda:3")

    assert (
        bench.main(["-c", str(config), "--engine", "fake", "--units", "2", "--out", str(out)]) == 0
    )

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["device"] == {"configured": "cuda:3", "backend": "cpu", "asr_backend": None}


@pytest.mark.cpu
def test_main_rejects_a_broken_config(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("engine:\n  device: potato\n", encoding="utf-8")
    assert bench.main(["-c", str(bad), "--engine", "fake"]) == 2


@pytest.mark.cpu
def test_main_refuses_a_real_run_on_an_unavailable_device(tmp_path: Path, capsys: Any) -> None:
    config = _config_file(tmp_path, device="cuda:3")
    assert bench.main(["-c", str(config), "--units", "1"]) == 2
    assert "not available" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_talkover_bench_runs_end_to_end_with_the_fake_engine(tmp_path: Path) -> None:
    out = tmp_path / "cli-bench.json"
    config = _config_file(tmp_path)

    code = cli_main(
        [
            "bench",
            "-c",
            str(config),
            "--engine",
            "fake",
            "--units",
            "3",
            "--warmup",
            "1",
            "--clip",
            bench.SYNTHETIC,
            "--out",
            str(out),
        ]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["engine"] == "fake"
    assert report["warmup_units"] == 1
    assert report["clip"]["source"] == bench.SYNTHETIC
    # `--units 3 --warmup 1` feeds four units and measures the last three.
    assert report["clip"]["units"] == 4
    assert report["stats"]["unit"]["count"] == 3


@pytest.mark.cpu
def test_talkover_bench_needs_a_config(capsys: Any) -> None:
    assert cli_main(["bench"]) == 2
    assert "config file is required" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# The real model on MPS
# --------------------------------------------------------------------------------------


@pytest.mark.mps
def test_real_bench_run_on_mps(tmp_path: Path) -> None:
    """The T1.8 acceptance run. It skips until the checkpoints are complete.

    When it passes, its numbers are the ones that go into `docs/mps-porting-notes.md`; the
    same command by hand is::

        uv run talkover bench -c configs/serve.example.yaml --units 30 --warmup 2 --asr
    """
    from talkover.config import load_config

    backend = get_backend("mps")
    if not backend.is_available():
        pytest.skip("MPS is not available on this machine")
    if not SERVE_EXAMPLE.is_file():
        pytest.skip(f"{SERVE_EXAMPLE} is missing")
    missing = _missing_checkpoints(load_config(SERVE_EXAMPLE))
    if missing:
        pytest.skip(
            "Gander checkpoints are not downloaded; missing "
            + ", ".join(missing)
            + " (run scripts/download_models.sh)"
        )

    out = tmp_path / "bench.json"
    code = bench.main(
        ["-c", str(SERVE_EXAMPLE), "--units", "5", "--warmup", "1", "--out", str(out)]
    )

    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["engine"] == "real"
    assert report["device"]["backend"] == "mps"
    assert report["stats"]["unit"]["count"] == 5
    assert report["stats"]["thinker"]["count"] >= 5
    print(
        "\n"
        + bench.format_table(report["stats"])
        + "\n"
        + bench.format_verdict(report)
        + f"\ndump: {out}"
    )
