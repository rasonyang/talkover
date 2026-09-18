"""Tests for the pre-startup memory estimate (DESIGN.md 4.5). All cases are GPU-free."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from talkover.config import parse_config, read_yaml
from talkover.engine.memory import (
    BORDERLINE_RATIO,
    CUDA_A10_BUDGET_BYTES,
    GIB,
    MemoryEstimate,
    Verdict,
    device_budget_bytes,
    estimate_memory,
    format_estimate,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_CONFIG = REPO_ROOT / "configs" / "serve.example.yaml"
MEMORY_MODULE = REPO_ROOT / "src" / "talkover" / "engine" / "memory.py"

MPS_BUDGET_BYTES = 128 * GIB


def _example(**engine_overrides: Any) -> dict[str, Any]:
    """The example config as a raw mapping, with `engine` / `asr` keys overridden.

    Keys are given as dotted paths, e.g. `_example(**{"engine.device": "cuda"})`.
    """
    data = copy.deepcopy(read_yaml(EXAMPLE_CONFIG))
    for dotted, value in engine_overrides.items():
        section, _, key = dotted.partition(".")
        data.setdefault(section, {})[key] = value
    return data


def _config(**overrides: Any):
    return parse_config(_example(**overrides))


# --- Acceptance cases from tasks.md T1.2 ------------------------------------------


@pytest.mark.cpu
def test_cuda_with_asr_on_cuda_is_over_budget() -> None:
    config = _config(**{"engine.device": "cuda"})
    assert config.asr.device == "cuda"
    result = estimate_memory(config, budget_bytes=CUDA_A10_BUDGET_BYTES)
    assert result.asr_on_device is True
    assert result.asr_bytes > 0
    assert result.verdict is Verdict.OVER_BUDGET
    assert result.total_bytes > result.budget_bytes
    assert result.headroom_bytes < 0
    assert result.fits is False


@pytest.mark.cpu
def test_cuda_with_asr_on_cpu_is_borderline() -> None:
    config = _config(**{"engine.device": "cuda", "asr.device": "cpu"})
    result = estimate_memory(config, budget_bytes=CUDA_A10_BUDGET_BYTES)
    assert result.asr_on_device is False
    assert result.asr_bytes == 0
    assert result.verdict is Verdict.BORDERLINE
    assert result.total_bytes <= result.budget_bytes
    assert result.total_bytes >= result.budget_bytes * BORDERLINE_RATIO
    assert result.fits is True


@pytest.mark.cpu
def test_mps_with_128gb_is_ok() -> None:
    config = _config(**{"engine.device": "mps"})
    result = estimate_memory(config, budget_bytes=MPS_BUDGET_BYTES)
    assert result.verdict is Verdict.OK
    # Unified memory: an ASR backend on the CPU still draws on the same pool.
    assert result.asr_on_device is True


@pytest.mark.cpu
def test_mps_counts_cpu_asr_against_unified_memory() -> None:
    config = _config(**{"engine.device": "mps", "asr.device": "cpu"})
    result = estimate_memory(config, budget_bytes=MPS_BUDGET_BYTES)
    assert result.asr_on_device is True
    assert result.asr_bytes > 0


# --- Component behaviour ----------------------------------------------------------


@pytest.mark.cpu
def test_components_sum_to_total() -> None:
    result = estimate_memory(_config(), budget_bytes=MPS_BUDGET_BYTES)
    assert sum(value for _, value in result.components) == result.total_bytes


@pytest.mark.cpu
def test_vision_encoder_only_counted_when_enabled() -> None:
    off = estimate_memory(_config(**{"engine.init_vision": False}), budget_bytes=MPS_BUDGET_BYTES)
    on = estimate_memory(_config(**{"engine.init_vision": True}), budget_bytes=MPS_BUDGET_BYTES)
    assert off.vision_bytes == 0
    assert on.vision_bytes > 0
    assert on.total_bytes - off.total_bytes == on.vision_bytes
    assert any("vision encoder not loaded" in note for note in off.notes)


@pytest.mark.cpu
def test_int8_thinker_shrinks_the_thinker() -> None:
    base = estimate_memory(_config(), budget_bytes=CUDA_A10_BUDGET_BYTES)
    quantized = estimate_memory(
        _config(**{"engine.quantize_thinker": "int8"}), budget_bytes=CUDA_A10_BUDGET_BYTES
    )
    assert quantized.thinker_bytes < base.thinker_bytes
    assert quantized.total_bytes < base.total_bytes


@pytest.mark.cpu
def test_fallback_ladder_brings_a10_back_to_ok() -> None:
    """DESIGN.md 4.5: ASR to CPU, then int8 Thinker, then no vision encoder."""
    config = _config(
        **{
            "engine.device": "cuda",
            "engine.quantize_thinker": "int8",
            "engine.init_vision": False,
            "asr.device": "cpu",
        }
    )
    result = estimate_memory(config, budget_bytes=CUDA_A10_BUDGET_BYTES)
    assert result.verdict is Verdict.OK


@pytest.mark.cpu
def test_kv_cache_scales_with_context_max_units() -> None:
    small = estimate_memory(
        _config(**{"engine.context_max_units": 64}), budget_bytes=MPS_BUDGET_BYTES
    )
    large = estimate_memory(
        _config(**{"engine.context_max_units": 256}), budget_bytes=MPS_BUDGET_BYTES
    )
    assert large.kv_cache_bytes > small.kv_cache_bytes
    # Only the per-unit part scales; the prior-context allowance is fixed.
    assert large.kv_cache_bytes < 4 * small.kv_cache_bytes


@pytest.mark.cpu
def test_float32_doubles_weight_and_kv_estimates() -> None:
    bf16 = estimate_memory(_config(**{"engine.dtype": "bfloat16"}), budget_bytes=MPS_BUDGET_BYTES)
    fp32 = estimate_memory(_config(**{"engine.dtype": "float32"}), budget_bytes=MPS_BUDGET_BYTES)
    assert fp32.thinker_bytes == 2 * bf16.thinker_bytes
    assert fp32.talker_bytes == 2 * bf16.talker_bytes
    assert fp32.kv_cache_bytes == 2 * bf16.kv_cache_bytes
    # Activations and ASR are not scaled by the model dtype.
    assert fp32.activation_bytes == bf16.activation_bytes
    assert fp32.asr_bytes == bf16.asr_bytes


@pytest.mark.cpu
def test_design_table_ranges_are_respected() -> None:
    """The component estimates must stay inside the ranges in DESIGN.md 4.5."""
    result = estimate_memory(
        _config(**{"engine.device": "cuda", "engine.init_vision": True, "asr.device": "cpu"}),
        budget_bytes=CUDA_A10_BUDGET_BYTES,
    )
    thinker = result.thinker_bytes + result.vision_bytes
    assert 17 * GIB <= thinker <= 18 * GIB
    assert 1.5 * GIB <= result.talker_bytes <= 2 * GIB
    assert 1.9 * GIB <= result.kv_cache_bytes <= 3 * GIB
    assert 1 * GIB <= result.activation_bytes <= 1.5 * GIB
    assert 22 * GIB <= result.total_bytes <= 24 * GIB


# --- Verdict, budgets and reporting -----------------------------------------------


@pytest.mark.cpu
def test_verdict_thresholds_are_exact() -> None:
    result = estimate_memory(_config(), budget_bytes=MPS_BUDGET_BYTES)
    total = result.total_bytes
    assert estimate_memory(_config(), budget_bytes=total).verdict is Verdict.BORDERLINE
    assert estimate_memory(_config(), budget_bytes=total - 1).verdict is Verdict.OVER_BUDGET
    generous = int(total / BORDERLINE_RATIO) + 2 * GIB
    assert estimate_memory(_config(), budget_bytes=generous).verdict is Verdict.OK


@pytest.mark.cpu
def test_device_budget_bytes_for_cuda_is_the_a10_budget() -> None:
    budget, note = device_budget_bytes("cuda:1")
    assert budget == CUDA_A10_BUDGET_BYTES
    assert note is not None


@pytest.mark.cpu
def test_device_budget_bytes_for_host_devices_is_plausible() -> None:
    for device in ("mps", "cpu"):
        budget, _ = device_budget_bytes(device)
        assert budget >= 4 * GIB


@pytest.mark.cpu
def test_estimate_without_override_uses_the_detected_budget() -> None:
    result = estimate_memory(_config(**{"engine.device": "cuda"}))
    assert result.budget_bytes == CUDA_A10_BUDGET_BYTES


@pytest.mark.cpu
def test_result_is_frozen() -> None:
    result = estimate_memory(_config(), budget_bytes=MPS_BUDGET_BYTES)
    assert isinstance(result, MemoryEstimate)
    with pytest.raises(AttributeError):
        result.total_bytes = 0  # type: ignore[misc]


@pytest.mark.cpu
def test_format_estimate_is_readable() -> None:
    result = estimate_memory(
        _config(**{"engine.device": "cuda"}), budget_bytes=CUDA_A10_BUDGET_BYTES
    )
    text = format_estimate(result)
    for name, _ in result.components:
        assert name in text
    assert "budget" in text
    assert "verdict" in text
    assert "OVER BUDGET" in text
    assert text == text.strip()


# --- Lint guard -------------------------------------------------------------------


@pytest.mark.cpu
def test_memory_module_never_references_device_apis() -> None:
    source = MEMORY_MODULE.read_text(encoding="utf-8")
    for forbidden in ("torch.cuda", "torch.mps", "import torch"):
        assert forbidden not in source, f"{MEMORY_MODULE} must not reference {forbidden}"
