"""GPU/host memory estimation before startup, to avoid OOM halfway through loading.

The model follows the budget table in DESIGN.md section 4.5. It is a static estimate
computed from the typed configuration only: no weights are read, no model is
instantiated, and nothing here may touch a device-specific torch API (a lint guard in
`tests/engine/test_backend.py` enforces that for the whole engine package outside
`engine/backend/`). torch is never imported here.

Components, all at bf16 unless the configuration says otherwise:

* Thinker backbone + audio encoder, plus the vision encoder when `engine.init_vision`
  is true, optionally int8-quantized through `engine.quantize_thinker`.
* Talker + token2wav.
* KV cache, sized from `engine.context_max_units` plus a fixed prior-context allowance.
* Activations and the device runtime context.
* ASR, counted only when it lands in the same memory pool as the model.

Verdict thresholds
------------------
DESIGN.md section 4.5 gives the component ranges and calls full bf16 on the A10
"borderline", but it does not define a numeric threshold. This module defines one:

* `over_budget` — the estimated total exceeds the budget.
* `borderline`  — the total is at or above `BORDERLINE_RATIO` (0.85) of the budget.
  Fifteen percent headroom is what the fragmentation and allocator slack of a long
  running session need; below that, a 10-minute call is not expected to survive.
* `ok`          — everything else.

Budgets: 24 GiB for a single A10 on CUDA (DESIGN.md 4.0), the machine's unified memory
size on MPS, and host memory on CPU. `estimate_memory(..., budget_bytes=...)` overrides
the detected value so tests stay deterministic.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum

from talkover.config import AsrConfig, EngineConfig, TalkoverConfig

__all__ = [
    "BORDERLINE_RATIO",
    "CUDA_A10_BUDGET_BYTES",
    "MemoryEstimate",
    "Verdict",
    "device_budget_bytes",
    "estimate_memory",
    "format_estimate",
]

GIB = 1024**3


# --------------------------------------------------------------------------------------
# Constants. Every number below is a static estimate taken from the table in
# DESIGN.md section 4.5 ("A10 Single-Card VRAM Budget") and split into the pieces the
# configuration can switch on. The sums reproduce that table:
#   Thinker with vision 17.75 GiB (table: 17-18 GB), Talker + token2wav 1.75 GiB
#   (table: 1.5-2 GB), KV cache 1.96 GiB at 128 units (table: 2-3 GB), activations
#   1.25 GiB (table: 1-1.5 GB).
# --------------------------------------------------------------------------------------

# MiniCPM-o 4.5 backbone plus the audio encoder, bf16.
THINKER_BF16_BYTES = int(16.75 * GIB)
# SigLIP-class vision encoder, bf16. Skipped when `init_vision: false` (DESIGN.md 4.5,
# fallback step 2).
VISION_BF16_BYTES = int(1.00 * GIB)
# Talker plus token2wav, bf16.
TALKER_BF16_BYTES = int(1.75 * GIB)
# Activations plus the device runtime context.
ACTIVATION_BYTES = int(1.25 * GIB)

# KV cache, bf16: 2 (K and V) * 36 layers * 8 KV heads * 128 head_dim * 2 bytes.
KV_BF16_BYTES_PER_TOKEN = 2 * 36 * 8 * 128 * 2
# One 1 s causal unit costs roughly this many tokens of context (audio tokens plus the
# text the Cerebellum emits for that unit).
KV_TOKENS_PER_UNIT = 100
# DESIGN.md 4.5 sizes the cache as "128 units + 1500 tokens of prior context".
KV_PRIOR_CONTEXT_TOKENS = 1500

# int8 quantization of the Thinker's linear layers (torchao / bitsandbytes). Embeddings,
# norms and the audio encoder stay at bf16, so the saving is less than half.
THINKER_INT8_RATIO = 0.55

# whisper large-v3-turbo weights, by ASR compute type.
ASR_WEIGHT_BYTES = {
    "float32": int(3.24 * GIB),
    "float16": int(1.62 * GIB),
    "bfloat16": int(1.62 * GIB),
    "int8_float16": int(0.81 * GIB),
    "int8": int(0.81 * GIB),
}
ASR_DEFAULT_WEIGHT_BYTES = ASR_WEIGHT_BYTES["float16"]
# Encoder activations and the ASR runtime's own context, when it shares the device.
ASR_RUNTIME_BYTES = int(0.90 * GIB)

# bf16 is the reference precision; other dtypes scale the weight and KV numbers.
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

CUDA_A10_BUDGET_BYTES = 24 * GIB
"""Single-A10 VRAM budget (DESIGN.md 4.0: "A10, 24 GB VRAM, single card")."""

BORDERLINE_RATIO = 0.85
"""Fraction of the budget at or above which a configuration is called `borderline`."""

_FALLBACK_HOST_BYTES = 16 * GIB


class Verdict(StrEnum):
    """How the estimated total compares to the device budget."""

    OK = "ok"
    BORDERLINE = "borderline"
    OVER_BUDGET = "over_budget"


@dataclass(frozen=True, slots=True)
class MemoryEstimate:
    """Per-component byte estimate for one configuration, plus the budget verdict."""

    device: str
    dtype: str
    thinker_bytes: int
    vision_bytes: int
    talker_bytes: int
    kv_cache_bytes: int
    activation_bytes: int
    asr_bytes: int
    total_bytes: int
    budget_bytes: int
    verdict: Verdict
    asr_on_device: bool
    notes: tuple[str, ...] = ()

    @property
    def components(self) -> tuple[tuple[str, int], ...]:
        """The per-component numbers in report order, including zero-sized ones."""
        return (
            ("thinker", self.thinker_bytes),
            ("vision_encoder", self.vision_bytes),
            ("talker_token2wav", self.talker_bytes),
            ("kv_cache", self.kv_cache_bytes),
            ("activations", self.activation_bytes),
            ("asr", self.asr_bytes),
        )

    @property
    def headroom_bytes(self) -> int:
        """Budget minus total; negative when the configuration does not fit."""
        return self.budget_bytes - self.total_bytes

    @property
    def utilization(self) -> float:
        """Total as a fraction of the budget."""
        return self.total_bytes / self.budget_bytes if self.budget_bytes > 0 else float("inf")

    @property
    def fits(self) -> bool:
        return self.verdict is not Verdict.OVER_BUDGET


# --------------------------------------------------------------------------------------
# Budget detection
# --------------------------------------------------------------------------------------


def _host_memory_bytes() -> int | None:
    """Physical RAM of this machine, or None when it cannot be determined."""
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return int(out.stdout.strip())
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError):
        return None
    if pages is None or page_size is None or pages <= 0 or page_size <= 0:
        return None
    return int(pages) * int(page_size)


def device_budget_bytes(device: str) -> tuple[int, str | None]:
    """Memory budget for `device`, with an optional note about how it was obtained.

    CUDA assumes a single A10 (DESIGN.md 4.0); VRAM is not queried here because this
    module must not reach for a device-specific torch API, and the estimate has to run
    before any device is initialized. MPS and CPU use the machine's physical memory,
    which on Apple Silicon is the unified pool the model shares.
    """
    kind = device.split(":", 1)[0]
    if kind == "cuda":
        return CUDA_A10_BUDGET_BYTES, "CUDA budget assumes a single 24 GB A10 (DESIGN.md 4.0)"
    host = _host_memory_bytes()
    if host is None:
        assumed = _fmt(_FALLBACK_HOST_BYTES)
        return _FALLBACK_HOST_BYTES, f"could not detect host memory; assuming {assumed}"
    if kind == "mps":
        return host, "MPS budget is the machine's unified memory, shared with the OS"
    return host, None


# --------------------------------------------------------------------------------------
# Estimation
# --------------------------------------------------------------------------------------


def _dtype_scale(dtype: str) -> float:
    return DTYPE_BYTES.get(dtype, 2) / 2


def _asr_shares_device(engine: EngineConfig, asr: AsrConfig) -> bool:
    """Whether the ASR backend draws on the same memory pool as the model.

    On MPS the pool is unified, so an ASR backend pinned to `cpu` still competes for the
    same physical RAM. On CUDA, moving ASR to the CPU frees VRAM outright, which is the
    first step of the fallback ladder in DESIGN.md 4.5.
    """
    engine_kind = engine.device_kind
    asr_kind = asr.device.split(":", 1)[0]
    if engine_kind == "mps":
        return True
    return asr_kind == engine_kind


def _asr_bytes(asr: AsrConfig) -> int:
    weights = ASR_WEIGHT_BYTES.get(asr.compute_type, ASR_DEFAULT_WEIGHT_BYTES)
    return weights + ASR_RUNTIME_BYTES


def _verdict(total: int, budget: int) -> Verdict:
    if budget <= 0 or total > budget:
        return Verdict.OVER_BUDGET
    if total >= budget * BORDERLINE_RATIO:
        return Verdict.BORDERLINE
    return Verdict.OK


def estimate_memory(config: TalkoverConfig, *, budget_bytes: int | None = None) -> MemoryEstimate:
    """Estimate device memory for `config` without loading anything.

    `budget_bytes` overrides the detected device budget; tests pass it so the result does
    not depend on the machine the suite runs on.
    """
    engine = config.engine
    asr = config.asr
    scale = _dtype_scale(engine.dtype)
    notes: list[str] = []

    thinker = int(THINKER_BF16_BYTES * scale)
    if engine.quantize_thinker == "int8":
        thinker = int(THINKER_BF16_BYTES * THINKER_INT8_RATIO)
        notes.append("thinker linear layers quantized to int8")

    vision = int(VISION_BF16_BYTES * scale) if engine.init_vision else 0
    if not engine.init_vision:
        notes.append("vision encoder not loaded (init_vision: false)")

    talker = int(TALKER_BF16_BYTES * scale)

    kv_tokens = engine.context_max_units * KV_TOKENS_PER_UNIT + KV_PRIOR_CONTEXT_TOKENS
    kv_cache = int(kv_tokens * KV_BF16_BYTES_PER_TOKEN * scale)

    activations = ACTIVATION_BYTES

    asr_on_device = _asr_shares_device(engine, asr)
    asr_bytes = _asr_bytes(asr) if asr_on_device else 0
    if not asr_on_device:
        notes.append(f"ASR runs on {asr.device}, outside the {engine.device_kind} memory pool")

    total = thinker + vision + talker + kv_cache + activations + asr_bytes

    if budget_bytes is None:
        budget, budget_note = device_budget_bytes(engine.device)
        if budget_note:
            notes.append(budget_note)
    else:
        budget = budget_bytes

    return MemoryEstimate(
        device=engine.device,
        dtype=engine.dtype,
        thinker_bytes=thinker,
        vision_bytes=vision,
        talker_bytes=talker,
        kv_cache_bytes=kv_cache,
        activation_bytes=activations,
        asr_bytes=asr_bytes,
        total_bytes=total,
        budget_bytes=budget,
        verdict=_verdict(total, budget),
        asr_on_device=asr_on_device,
        notes=tuple(notes),
    )


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _fmt(num_bytes: int) -> str:
    return f"{num_bytes / GIB:.2f} GiB"


_VERDICT_TEXT = {
    Verdict.OK: "ok",
    Verdict.BORDERLINE: "borderline - little headroom for a long session",
    Verdict.OVER_BUDGET: "OVER BUDGET - startup is expected to OOM",
}


def format_estimate(result: MemoryEstimate) -> str:
    """Render an estimate as the plain-text block `talkover check` prints."""
    lines = [f"memory estimate for device {result.device} ({result.dtype}):"]
    width = max(len(name) for name, _ in result.components)
    for name, value in result.components:
        suffix = ""
        if value == 0:
            suffix = " (not loaded)" if name != "asr" else " (off-device)"
        lines.append(f"  {name.ljust(width)}  {_fmt(value).rjust(9)}{suffix}")
    lines.append(f"  {'total'.ljust(width)}  {_fmt(result.total_bytes).rjust(9)}")
    lines.append(f"  {'budget'.ljust(width)}  {_fmt(result.budget_bytes).rjust(9)}")
    lines.append(
        f"  {'headroom'.ljust(width)}  {_fmt(result.headroom_bytes).rjust(9)} "
        f"({result.utilization:.0%} used)"
    )
    lines.append(f"verdict: {_VERDICT_TEXT[result.verdict]}")
    for note in result.notes:
        lines.append(f"  note: {note}")
    return "\n".join(lines)
