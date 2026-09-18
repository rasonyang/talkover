#!/usr/bin/env python
"""Per-unit processing time measurement for the Gander engine (T1.8, DESIGN.md 4.4).

The model consumes one second of audio per causal unit, so the hard constraint is that
one unit finishes end to end in under 1.0 s. This script feeds a fixed clip through the
engine in 1 s units, times every unit against the device, and reports mean / p50 / p95
per stage. It also writes a JSON dump of the emitted token sequence plus the device,
config and commit metadata, so an M4 CUDA run can be diffed against the MPS run that
produced it (DESIGN.md 11, "the two backends diverge in behavior").

What is measured
----------------

- ``unit`` — wall time of one ``feed_pcm16`` round trip, measured on the event loop. This
  is the number DESIGN.md 4.4 budgets at < 1.0 s. ``flush`` is the same measurement for the
  single ``flush_pending`` that ends the run, kept apart because it runs the tail unit.
- ``step`` — the same unit measured on the inference thread with
  :meth:`DeviceBackend.synchronize` on both sides of the upstream call, so it is compute
  time rather than submission time. The difference between ``unit`` and ``step`` is the
  thread hand-over.
- ``thinker`` / ``talker_prep`` / ``talker`` / ``token2wav`` — upstream's own per-stage
  costs (``cost_llm``, ``cost_tts_prep``, ``cost_tts``, ``cost_token2wav``), taken from the
  step metrics through :class:`talkover.engine.session.UnitTiming`. Upstream reports no
  separate cost for encoding the input audio: it is inside ``cost_llm``, so no
  ``audio_encode`` row appears until upstream grows one.
- ``asr`` — the side channel (DESIGN.md 4.3), timed here rather than in the engine: each
  unit is handed to the configured ASR backend and the transcribe call is timed. Off by
  default; ``--asr`` turns it on.

The token sequence dump is the Thinker's ``generated_token_ids`` per unit plus the text it
decoded to. Speak-token *ids* are not in it: upstream hands them straight to the speech
worker and puts only the count (``n_tts_tokens``) on the step event, so the dump carries
that count and the number of waveform samples the unit produced instead.

Running it
----------

::

    uv run talkover bench -c configs/serve.example.yaml            # the real model
    uv run python scripts/bench_rtf.py -c configs/serve.example.yaml --engine fake

``--engine fake`` replaces the engine with an in-process fake that loads no weights and
touches no device. It exercises the whole harness — clip slicing, the timing hook, the
statistics, the table and the JSON schema — which is what ``tests/engine/test_bench.py``
runs and what makes this script testable while the checkpoints are still downloading.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import platform
import subprocess
import sys
import time
import wave
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # running the file directly, without the wheel
    sys.path.insert(0, str(REPO_ROOT / "src"))

from talkover.config import ConfigError, TalkoverConfig, load_config
from talkover.engine.protocol import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    UNIT_BYTES,
    UNIT_SAMPLES,
    EngineStepEvent,
)
from talkover.engine.session import EngineSession, UnitTiming

__all__ = [
    "BUDGET_SEC",
    "SCHEMA_VERSION",
    "STAGE_ORDER",
    "BenchClip",
    "FakeBenchEngine",
    "StageTimer",
    "build_parser",
    "format_table",
    "load_clip",
    "main",
    "percentile",
    "run_bench",
    "summarize",
]

#: Bumped whenever the shape of the JSON dump changes; a diff tool checks it first.
SCHEMA_VERSION = "talkover.bench/1"

#: Row order of the printed table. ``unit`` and ``step`` are measured here and in the
#: engine; the rest come from :data:`talkover.engine.session.STAGE_NAMES`.
STAGE_ORDER = (
    "unit",
    "flush",
    "step",
    "audio_encode",
    "thinker",
    "talker_prep",
    "talker",
    "token2wav",
    "asr",
)

#: The DESIGN.md 4.4 budget, in seconds. ``talker`` covers the Talker plus token2wav.
BUDGET_SEC: Mapping[str, float] = {
    "unit": 1.0,
    "thinker": 0.45,
    "talker": 0.35,
    "asr": 0.10,
}

#: The bundled Mandarin clip (see ``tests/fixtures/README.md``), used when it is present.
DEFAULT_CLIP = REPO_ROOT / "tests" / "fixtures" / "zh_order_16k.wav"

#: ``--clip`` value that forces the generated clip instead of a file.
SYNTHETIC = "synthetic"

_DEFAULT_UNITS = 5
_DEFAULT_SEED = 20260918


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------


def percentile(values: Sequence[float], q: float) -> float:
    """The ``q``-th percentile (0-100) by nearest rank, on an unsorted sequence.

    Nearest rank rather than interpolation: a bench run has a handful of units, and an
    interpolated p95 of five samples is a number no measurement produced.
    """
    if not values:
        raise ValueError("percentile of an empty sequence")
    ordered = sorted(values)
    rank = math.ceil(q / 100.0 * len(ordered))
    return ordered[max(1, min(rank, len(ordered))) - 1]


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    """Count, mean, p50, p95, min and max of one stage's samples, in seconds."""
    samples = [float(value) for value in values]
    if not samples:
        return {"count": 0}
    return {
        "count": len(samples),
        "mean": sum(samples) / len(samples),
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "min": min(samples),
        "max": max(samples),
    }


# --------------------------------------------------------------------------------------
# Collecting the timings
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class StageTimer:
    """Collects :class:`UnitTiming` records and the stage times measured outside the engine.

    It is handed to :class:`~talkover.engine.session.EngineSession` as ``on_unit_timing``,
    so :meth:`record` runs on the inference thread: it only appends, and appending to a
    list is atomic under the GIL.
    """

    #: Every record the engine reported, in arrival order.
    timings: list[UnitTiming] = field(default_factory=list)
    #: Stage name -> samples measured by the harness itself (``unit``, ``asr``).
    external: dict[str, list[float]] = field(default_factory=dict)

    def record(self, timing: UnitTiming) -> None:
        """The ``on_unit_timing`` callback."""
        self.timings.append(timing)

    def add(self, stage: str, seconds: float) -> None:
        """Record one sample of a stage the engine does not measure."""
        self.external.setdefault(stage, []).append(float(seconds))

    def samples(self) -> dict[str, list[float]]:
        """Stage name -> every sample collected for it."""
        collected: dict[str, list[float]] = {k: list(v) for k, v in self.external.items()}
        for timing in self.timings:
            if timing.source == "step":
                collected.setdefault("step", []).append(timing.wall_sec)
            for stage, seconds in timing.stages.items():
                collected.setdefault(stage, []).append(seconds)
        return collected

    def stats(self) -> dict[str, dict[str, float | int]]:
        """Per-stage statistics, in :data:`STAGE_ORDER` first and then anything else."""
        samples = self.samples()
        ordered = [name for name in STAGE_ORDER if samples.get(name)]
        ordered += [name for name in sorted(samples) if name not in STAGE_ORDER and samples[name]]
        return {name: summarize(samples[name]) for name in ordered}


# --------------------------------------------------------------------------------------
# The clip
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchClip:
    """The fixed input: whole 1 s units of 16 kHz mono pcm16, plus how they were made."""

    units: tuple[bytes, ...]
    source: str
    path: str | None = None
    #: The clip was shorter than ``--units`` and was repeated from the start.
    looped: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "path": self.path,
            "sample_rate": INPUT_SAMPLE_RATE,
            "units": len(self.units),
            "unit_ms": 1000,
            "looped": self.looped,
            "sha256": hashlib.sha256(b"".join(self.units)).hexdigest(),
        }


def _read_wav(path: Path) -> bytes:
    """Read a mono 16 kHz pcm16 wav file into raw frames.

    Raises:
        ValueError: if the file is not the format the engine takes.
    """
    with wave.open(str(path), "rb") as handle:
        params = handle.getparams()
        if params.nchannels != 1 or params.sampwidth != 2:
            raise ValueError(
                f"{path}: the bench clip must be mono pcm16, got {params.nchannels} channel(s) "
                f"at {params.sampwidth * 8} bits"
            )
        if params.framerate != INPUT_SAMPLE_RATE:
            raise ValueError(
                f"{path}: the bench clip must be {INPUT_SAMPLE_RATE} Hz, got {params.framerate}"
            )
        return handle.readframes(params.nframes)


def _synthetic_frames(units: int, seed: int) -> bytes:
    """A deterministic stand-in clip: a 220 Hz tone, syllable-shaped, plus faint noise.

    It is not speech and the model has nothing to say about it. It exists so the harness
    runs on a machine without the fixture, and so the input is reproducible from ``--seed``
    alone; a real measurement uses the bundled clip.
    """
    rng = np.random.default_rng(seed)
    total = units * UNIT_SAMPLES
    t = np.arange(total, dtype=np.float32) / INPUT_SAMPLE_RATE
    tone = np.sin(2 * np.pi * 220.0 * t, dtype=np.float32)
    syllables = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t, dtype=np.float32)
    noise = rng.standard_normal(total).astype(np.float32) * 0.01
    wave_f32 = np.clip(0.3 * tone * syllables + noise, -1.0, 1.0)
    return (wave_f32 * 32767.0).astype("<i2").tobytes()


def load_clip(clip: str | None, units: int, seed: int) -> BenchClip:
    """Build the fixed input for ``units`` units.

    ``clip`` is a wav path, the literal ``synthetic``, or ``None`` for "the bundled
    fixture if it is there, synthetic otherwise". A clip shorter than ``units`` is
    repeated from the start, which is recorded in the dump.
    """
    if units <= 0:
        raise ValueError(f"--units must be positive, got {units}")

    if clip == SYNTHETIC:
        frames, source, path = _synthetic_frames(units, seed), SYNTHETIC, None
    else:
        candidate = Path(clip).expanduser() if clip else DEFAULT_CLIP
        if clip is None and not candidate.is_file():
            frames, source, path = _synthetic_frames(units, seed), SYNTHETIC, None
        else:
            frames, source, path = _read_wav(candidate), "file", str(candidate)

    whole = len(frames) // UNIT_BYTES
    if whole == 0:
        raise ValueError(
            f"the bench clip holds {len(frames)} bytes, less than the {UNIT_BYTES} of one unit"
        )
    sliced = [frames[i * UNIT_BYTES : (i + 1) * UNIT_BYTES] for i in range(whole)]
    looped = units > whole
    chosen = [sliced[i % whole] for i in range(units)]
    return BenchClip(units=tuple(chosen), source=source, path=path, looped=looped)


# --------------------------------------------------------------------------------------
# The fake engine
# --------------------------------------------------------------------------------------


class FakeBenchEngine:
    """An ``EngineProtocol`` implementation that loads nothing and runs anywhere.

    It produces a deterministic unit sequence from ``seed``: the first unit listens, the
    rest speak, and the last one ends the turn. Stage costs are synthetic constants, so a
    fake run's table and JSON dump are reproducible byte for byte apart from the measured
    ``unit`` / ``step`` rows. It reports them through the same ``on_unit_timing`` callback
    the real :class:`~talkover.engine.session.EngineSession` uses, so the harness below has
    no idea which engine it is driving.
    """

    #: Synthetic per-stage seconds, roughly the shape DESIGN.md 4.4 budgets.
    STAGE_COSTS: Mapping[str, float] = {
        "thinker": 0.20,
        "talker_prep": 0.01,
        "talker": 0.12,
        "token2wav": 0.08,
    }

    def __init__(
        self,
        *,
        seed: int = _DEFAULT_SEED,
        on_unit_timing: Any = None,
        speak_from_unit: int = 2,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._on_unit_timing = on_unit_timing
        self._speak_from_unit = speak_from_unit
        self._events: asyncio.Queue[EngineStepEvent | None] = asyncio.Queue()
        self._index = 0
        self._unit_id = 0
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def start(self) -> None:
        self._ready = True

    async def feed_pcm16(self, pcm16_16k_1s: bytes | np.ndarray) -> None:
        size = len(pcm16_16k_1s) if isinstance(pcm16_16k_1s, bytes) else pcm16_16k_1s.size
        if size not in (UNIT_BYTES, UNIT_SAMPLES):
            raise ValueError(f"not one 1 s unit: {size}")
        await self._step(end_of_turn=False)

    async def flush_pending(self) -> None:
        await self._step(end_of_turn=True)

    async def interrupt_output(self) -> None:
        return None

    async def set_task_slate(self, text: str) -> None:
        return None

    async def submit_text_turn(self, text: str) -> None:
        return None

    async def feed_tool_response(self, response: Mapping[str, Any]) -> None:
        return None

    async def feed_worker_delivery(self, delivery: Mapping[str, Any]) -> None:
        return None

    async def events(self) -> AsyncIterator[EngineStepEvent]:
        while True:
            item = await self._events.get()
            if item is None:
                return
            yield item

    async def stop(self) -> None:
        self._ready = False
        await self._events.put(None)

    async def _step(self, *, end_of_turn: bool) -> None:
        """Emit one deterministic unit and report its (synthetic) timing."""
        self._index += 1
        is_listen = self._index < self._speak_from_unit
        token_ids = tuple(int(value) for value in self._rng.integers(1000, 2000, size=4))
        waveform = None
        unit_id = None
        if not is_listen:
            self._unit_id += 1
            unit_id = self._unit_id
            waveform = (self._rng.standard_normal(OUTPUT_SAMPLE_RATE) * 0.05).astype(np.float32)
        stages = {} if is_listen else dict(self.STAGE_COSTS)
        if is_listen:
            stages = {"thinker": self.STAGE_COSTS["thinker"]}
        metrics = {
            "cost_llm": stages.get("thinker", 0.0),
            "cost_tts": stages.get("talker", 0.0),
            "cost_token2wav": stages.get("token2wav", 0.0),
            "n_tokens": len(token_ids),
            "n_tts_tokens": 0 if is_listen else 25,
            "fake": True,
        }
        event = EngineStepEvent(
            unit_index=self._index,
            is_listen=is_listen,
            text="" if is_listen else f"unit-{self._index}",
            end_of_turn=end_of_turn,
            interrupted=False,
            audio_waveform=waveform,
            unit_id=unit_id,
            metrics=metrics,
            step_wall_time_sec=sum(stages.values()),
        )
        await self._events.put(event)
        if self._on_unit_timing is not None:
            self._on_unit_timing(
                UnitTiming(
                    unit_index=self._index,
                    source="step",
                    wall_sec=sum(stages.values()),
                    stages=stages,
                    is_listen=is_listen,
                    end_of_turn=end_of_turn,
                    unit_id=unit_id,
                    text=event.text,
                    text_token_ids=token_ids,
                    speak_token_count=int(metrics["n_tts_tokens"]),
                    audio_samples=0 if waveform is None else int(waveform.size),
                    metrics=metrics,
                )
            )


# --------------------------------------------------------------------------------------
# Metadata for the dump
# --------------------------------------------------------------------------------------


def _git_commit(path: Path) -> str | None:
    """``git rev-parse --short HEAD`` in ``path``, or ``None`` when it is not a checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _upstream_dir() -> Path | None:
    """Where ``mcpmft`` is installed from, without importing it."""
    try:
        spec = importlib.util.find_spec("mcpmft")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _versions() -> dict[str, Any]:
    """Commits and versions, so two dumps can be compared for what produced them."""
    from talkover.check import UPSTREAM_PINNED_COMMIT

    upstream = _upstream_dir()
    return {
        "talkover_commit": _git_commit(REPO_ROOT),
        "upstream_pinned_commit": UPSTREAM_PINNED_COMMIT,
        "upstream_commit": _git_commit(upstream) if upstream is not None else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
    }


def _config_summary(config: TalkoverConfig, path: str | None) -> dict[str, Any]:
    """The parts of the config a measurement depends on. No credentials are copied."""
    engine = config.engine
    return {
        "path": path,
        "engine": {
            "device": engine.device,
            "talker_device": engine.talker_device,
            "dtype": engine.dtype,
            "quantize_thinker": engine.quantize_thinker,
            "init_vision": engine.init_vision,
            "media_mode": engine.media_mode,
            "attn_implementation": engine.attn_implementation,
            "context_max_units": engine.context_max_units,
            "sliding_window_mode": engine.sliding_window_mode,
            # Added by T1.5/T1.9 as the vocoder latency knob; read defensively so an older
            # config object (without the field) still produces a dump.
            "token2wav_timesteps": getattr(engine, "token2wav_timesteps", None),
            "base_model": engine.base_model,
            "thinker_checkpoint": engine.thinker_checkpoint,
            "talker_checkpoint": engine.talker_checkpoint,
        },
        "asr": {
            "backend": config.asr.backend,
            "model": config.asr.model,
            "device": config.asr.device,
            "compute_type": config.asr.compute_type,
        },
        "realtime": {
            "memory_slate_max_tokens": config.realtime.memory_slate_max_tokens,
            "trailing_silence_sec": config.realtime.trailing_silence_sec,
        },
    }


def _jsonable(value: Any) -> Any:
    """Keep only what survives a JSON round trip; upstream metrics nest freely."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    return repr(value)


# --------------------------------------------------------------------------------------
# Seeding
# --------------------------------------------------------------------------------------


def seed_device(backend: Any, seed: int) -> dict[str, Any]:
    """Pin the RNG from ``seed`` and push it into the device through the backend.

    ``torch.manual_seed`` seeds the host generator and every device generator torch knows
    about; the device state is then read back and written through
    :meth:`DeviceBackend.set_rng_state` — the one device RNG call the engine is allowed to
    make (DESIGN.md 4.2) — so a CUDA run at M4 starts from the same point as the MPS run
    it is compared against. The state itself is fingerprinted into the dump.
    """
    import torch

    torch.manual_seed(seed)
    info: dict[str, Any] = {"seed": seed}
    try:
        state = backend.rng_state()
        backend.set_rng_state(state)
    except Exception as exc:  # noqa: BLE001 - a device without an RNG must not stop a run
        info["rng_state"] = f"unavailable: {type(exc).__name__}: {exc}"
        return info
    raw = bytes(np.asarray(state.cpu() if hasattr(state, "cpu") else state).tobytes())
    info["rng_state_bytes"] = len(raw)
    info["rng_state_sha256"] = hashlib.sha256(raw).hexdigest()
    return info


# --------------------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------------------


async def _drain(engine: Any, collected: list[EngineStepEvent]) -> None:
    """Collect every event the engine emits until it stops."""
    async for event in engine.events():
        collected.append(event)


async def run_bench(
    *,
    clip: BenchClip,
    engine: Any,
    timer: StageTimer,
    asr: Any = None,
    warmup: int = 0,
) -> list[EngineStepEvent]:
    """Feed ``clip`` through ``engine`` one unit at a time, timing each round trip.

    The ``unit`` samples are taken here, on the event loop, because that is where the
    real-time budget is spent: it includes the hand-over to the inference thread that the
    engine's own ``step`` measurement excludes. Warm-up units are fed first and their
    samples are dropped, since the first unit of a session pays for lazy device
    allocations and the first kernel compilations.
    """
    collected: list[EngineStepEvent] = []
    consumer = asyncio.create_task(_drain(engine, collected))
    try:
        for index, unit in enumerate(clip.units, start=1):
            started = time.perf_counter()
            await engine.feed_pcm16(unit)
            elapsed = time.perf_counter() - started
            warming = index <= warmup
            if not warming:
                timer.add("unit", elapsed)
            if asr is not None:
                asr_started = time.perf_counter()
                await asyncio.to_thread(asr.transcribe, unit, start_ms=(index - 1) * 1000)
                if not warming:
                    timer.add("asr", time.perf_counter() - asr_started)
        flush_started = time.perf_counter()
        await engine.flush_pending()
        timer.add("flush", time.perf_counter() - flush_started)
    finally:
        await engine.stop()
        await consumer
    if warmup:
        # Drop the warm-up records so they never reach the statistics or the dump.
        timer.timings[:] = [t for t in timer.timings if t.unit_index > warmup]
        collected = [event for event in collected if event.unit_index > warmup]
    return collected


def token_sequence(timings: Sequence[UnitTiming]) -> list[dict[str, Any]]:
    """The diffable part of the dump: what the model emitted, with no timings in it.

    Two runs of the same clip and seed on two backends should produce the same list;
    ``jq .token_sequence`` on both dumps is the M4 comparison.
    """
    sequence = []
    for timing in timings:
        if timing.source != "step":
            continue
        sequence.append(
            {
                "unit_index": timing.unit_index,
                "unit_id": timing.unit_id,
                "is_listen": timing.is_listen,
                "end_of_turn": timing.end_of_turn,
                "text": timing.text,
                "text_token_ids": list(timing.text_token_ids),
                "speak_token_count": timing.speak_token_count,
                "audio_samples": timing.audio_samples,
            }
        )
    return sequence


def build_report(
    *,
    config: TalkoverConfig,
    config_path: str | None,
    clip: BenchClip,
    timer: StageTimer,
    backend: Any,
    engine_kind: str,
    seed_info: Mapping[str, Any],
    argv: Sequence[str],
    warmup: int,
    asr_backend: str | None,
) -> dict[str, Any]:
    """Assemble the JSON dump. Its shape is pinned by ``tests/engine/test_bench.py``."""
    stats = timer.stats()
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "argv": list(argv),
        "engine": engine_kind,
        "warmup_units": warmup,
        "seed": dict(seed_info),
        "device": {
            "configured": config.engine.device,
            "backend": getattr(backend, "name", "unknown"),
            "asr_backend": asr_backend,
        },
        "clip": clip.describe(),
        "config": _config_summary(config, config_path),
        "versions": _versions(),
        "budget_sec": dict(BUDGET_SEC),
        "stats": stats,
        "violations": budget_violations(stats),
        "units": [
            {
                "unit_index": timing.unit_index,
                "source": timing.source,
                "wall_sec": timing.wall_sec,
                "stages": dict(timing.stages),
                "is_listen": timing.is_listen,
                "end_of_turn": timing.end_of_turn,
                "unit_id": timing.unit_id,
                "generation_id": timing.generation_id,
                "audio_samples": timing.audio_samples,
                "metrics": _jsonable(timing.metrics),
            }
            for timing in timer.timings
        ],
        "token_sequence": token_sequence(timer.timings),
    }


def budget_violations(stats: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every stage whose p95 is over its DESIGN.md 4.4 budget."""
    violations = []
    for stage, budget in BUDGET_SEC.items():
        row = stats.get(stage)
        if not row or not row.get("count"):
            continue
        p95 = float(row["p95"])
        if p95 > budget:
            violations.append({"stage": stage, "p95_sec": p95, "budget_sec": budget})
    return violations


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------


def format_table(stats: Mapping[str, Mapping[str, Any]]) -> str:
    """The per-stage table, in milliseconds, with the budget column where there is one."""
    header = (
        f"{'stage':<12} {'n':>3} {'mean ms':>9} {'p50 ms':>9} "
        f"{'p95 ms':>9} {'max ms':>9} {'budget ms':>10}"
    )
    lines = [header, "-" * len(header)]
    for stage, row in stats.items():
        if not row.get("count"):
            continue
        budget = BUDGET_SEC.get(stage)
        lines.append(
            f"{stage:<12} {row['count']:>3} "
            f"{row['mean'] * 1000:>9.1f} {row['p50'] * 1000:>9.1f} "
            f"{row['p95'] * 1000:>9.1f} {row['max'] * 1000:>9.1f} "
            + (f"{budget * 1000:>10.0f}" if budget is not None else f"{'-':>10}")
        )
    return "\n".join(lines)


def format_verdict(report: Mapping[str, Any]) -> str:
    """One line per budget breach, or the line that says there were none."""
    violations = report["violations"]
    if not violations:
        return "budget: every measured stage is within DESIGN.md 4.4"
    return "\n".join(
        f"budget: {item['stage']} p95 {item['p95_sec'] * 1000:.1f} ms "
        f"exceeds {item['budget_sec'] * 1000:.0f} ms"
        for item in violations
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bench_rtf.py",
        description="measure per-unit and per-stage engine time (T1.8, DESIGN.md 4.4)",
    )
    parser.add_argument("-c", "--config", default="configs/serve.example.yaml")
    parser.add_argument(
        "--out",
        default=None,
        help="where to write the JSON dump; the default is bench_rtf.<backend>.json",
    )
    parser.add_argument(
        "--units", type=int, default=_DEFAULT_UNITS, help="how many 1 s units to feed"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="units fed before measurement starts; their samples are dropped",
    )
    parser.add_argument(
        "--engine",
        choices=("real", "fake"),
        default="real",
        help="'fake' runs the harness with no weights and no device",
    )
    parser.add_argument(
        "--clip",
        default=None,
        help=f"a 16 kHz mono pcm16 wav, or '{SYNTHETIC}'; the default is the bundled fixture",
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--asr",
        action="store_true",
        help="also time the ASR side channel on every unit (DESIGN.md 4.3)",
    )
    parser.add_argument(
        "--talker-thread",
        action="store_true",
        help="run the Talker on its own thread (T1.5); needs engine.ref_audio_path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the bench. 0 on success, 2 on a bad config or an unusable device."""
    args = build_parser().parse_args(list(argv) if argv is not None else None)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2

    from talkover.engine.backend import get_backend

    # The fake engine measures the harness, not a device, so it never claims one: a fake
    # run has to work on a CI box whose config file says `device: mps`.
    backend = get_backend("cpu" if args.engine == "fake" else config.engine.device)
    if args.engine == "real" and not backend.is_available():
        print(
            f"bench: engine.device {config.engine.device!r} is not available on this machine "
            f"(backend {backend.name!r}); run `talkover check -c {args.config}`",
            file=sys.stderr,
        )
        return 2

    try:
        clip = load_clip(args.clip, args.units + args.warmup, args.seed)
    except (OSError, ValueError, wave.Error) as exc:
        print(f"bench: {exc}", file=sys.stderr)
        return 2

    seed_info = seed_device(backend, args.seed)
    timer = StageTimer()

    asr = None
    asr_backend_name: str | None = None
    if args.asr:
        from talkover.engine.asr import AsrError, create_asr

        try:
            asr = create_asr(config.asr, backend)
        except AsrError as exc:
            print(f"bench: ASR is not usable, skipping the asr stage: {exc}", file=sys.stderr)
        else:
            asr_backend_name = getattr(asr, "name", config.asr.backend)

    if args.engine == "fake":
        engine: Any = FakeBenchEngine(seed=args.seed, on_unit_timing=timer.record)
    else:
        talker_factory = None
        if args.talker_thread:
            from talkover.engine.talker import talker_factory_from_config

            try:
                talker_factory = talker_factory_from_config(config)
            except (ValueError, NotImplementedError) as exc:
                print(f"bench: {exc}", file=sys.stderr)
                return 2
        engine = EngineSession(
            config,
            backend=backend,
            talker_factory=talker_factory,
            on_unit_timing=timer.record,
        )

    async def drive() -> list[EngineStepEvent]:
        await engine.start()
        return await run_bench(
            clip=clip,
            engine=engine,
            timer=timer,
            asr=asr,
            warmup=args.warmup,
        )

    try:
        asyncio.run(drive())
    finally:
        if asr is not None:
            asr.close()

    report = build_report(
        config=config,
        config_path=args.config,
        clip=clip,
        timer=timer,
        backend=backend,
        engine_kind=args.engine,
        seed_info=seed_info,
        argv=list(argv) if argv is not None else sys.argv[1:],
        warmup=args.warmup,
        asr_backend=asr_backend_name,
    )

    out = Path(args.out) if args.out else Path(f"bench_rtf.{backend.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        f"engine={args.engine} backend={backend.name} device={config.engine.device} "
        f"units={len(clip.units) - args.warmup} clip={clip.source} seed={args.seed}"
    )
    print(format_table(report["stats"]))
    print(format_verdict(report))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
