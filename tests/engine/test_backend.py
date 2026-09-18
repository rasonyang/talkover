"""Tests for the DeviceBackend protocol, its three implementations, and get_backend."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from talkover.engine.backend import DeviceBackend, get_backend
from talkover.engine.backend.cpu import CpuBackend
from talkover.engine.backend.cuda import CudaBackend

ENGINE_DIR = Path(__file__).resolve().parents[2] / "src" / "talkover" / "engine"
FORBIDDEN_OUTSIDE_BACKEND = ("torch.cuda", "torch.mps")


def _assert_rng_round_trip(backend: DeviceBackend) -> None:
    device = backend.device()
    state = backend.rng_state()
    first = torch.randn(64, device=device)
    backend.set_rng_state(state)
    second = torch.randn(64, device=device)
    assert torch.equal(first, second)


# --- CpuBackend ------------------------------------------------------------------


@pytest.mark.cpu
def test_cpu_backend_identity() -> None:
    backend = CpuBackend()
    assert backend.name == "cpu"
    assert backend.is_available() is True
    assert backend.device() == torch.device("cpu")
    assert backend.device(0) == torch.device("cpu")
    assert backend.autocast_dtype() is torch.bfloat16
    assert backend.synchronize() is None


@pytest.mark.cpu
def test_cpu_backend_rejects_other_index() -> None:
    with pytest.raises(ValueError, match="index"):
        CpuBackend().device(1)


@pytest.mark.cpu
def test_cpu_backend_rng_state_round_trip() -> None:
    _assert_rng_round_trip(CpuBackend())


@pytest.mark.cpu
def test_cpu_backend_satisfies_protocol() -> None:
    assert isinstance(CpuBackend(), DeviceBackend)
    assert isinstance(CudaBackend(3), DeviceBackend)


# --- get_backend parsing ---------------------------------------------------------


@pytest.mark.cpu
def test_get_backend_cpu() -> None:
    backend = get_backend("cpu")
    assert isinstance(backend, CpuBackend)
    assert backend.name == "cpu"


@pytest.mark.cpu
def test_get_backend_mps() -> None:
    from talkover.engine.backend.mps import MpsBackend

    backend = get_backend("mps")
    assert isinstance(backend, MpsBackend)
    assert backend.name == "mps"


@pytest.mark.cpu
@pytest.mark.parametrize(
    ("spec", "index"),
    [("cuda", 0), ("cuda:0", 0), ("cuda:1", 1), ("cuda:7", 7), (" CUDA:2 ", 2)],
)
def test_get_backend_cuda_index(spec: str, index: int) -> None:
    backend = get_backend(spec)
    assert isinstance(backend, CudaBackend)
    assert backend.name == "cuda"
    assert backend.index == index
    assert backend.device() == torch.device("cuda", index)


@pytest.mark.cpu
@pytest.mark.parametrize(
    "spec",
    ["", "gpu", "cuda:", "cuda:-1", "cuda:a", "cuda:0:1", "mps:0", "cpu:0", "metal", "CUDA 0"],
)
def test_get_backend_rejects_unknown_spec(spec: str) -> None:
    with pytest.raises(ValueError, match="unknown device"):
        get_backend(spec)


@pytest.mark.cpu
def test_get_backend_rejects_non_string() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        get_backend(0)  # type: ignore[arg-type]


@pytest.mark.cpu
def test_cuda_backend_rejects_negative_index() -> None:
    with pytest.raises(ValueError, match="negative"):
        CudaBackend(-1)


# --- Lint guard ------------------------------------------------------------------


@pytest.mark.cpu
def test_engine_outside_backend_has_no_direct_device_calls() -> None:
    """Device specifics stay in backend/mps.py and backend/cuda.py (DESIGN.md 4.2)."""
    offenders: list[str] = []
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        if (ENGINE_DIR / "backend") in path.parents or path.parent == ENGINE_DIR / "backend":
            continue
        source = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_OUTSIDE_BACKEND:
            if token in source:
                offenders.append(f"{path.relative_to(ENGINE_DIR)}: {token}")
    assert not offenders, (
        "engine code outside backend/ must go through DeviceBackend: " + ", ".join(offenders)
    )


# --- MpsBackend ------------------------------------------------------------------


@pytest.mark.mps
def test_mps_backend() -> None:
    from talkover.engine.backend.mps import MpsBackend

    backend = MpsBackend()
    if not backend.is_available():
        pytest.skip("MPS device is not available")

    assert backend.name == "mps"
    assert backend.device() == torch.device("mps")
    assert backend.autocast_dtype() is torch.bfloat16
    with pytest.raises(ValueError, match="index"):
        backend.device(1)

    tensor = torch.ones(8, device=backend.device(), dtype=backend.autocast_dtype())
    backend.synchronize()
    assert tensor.device.type == "mps"
    assert tensor.sum().item() == 8

    _assert_rng_round_trip(backend)


# --- CudaBackend -----------------------------------------------------------------


@pytest.mark.cuda
def test_cuda_backend() -> None:
    backend = CudaBackend(0)
    if not backend.is_available():
        pytest.skip("CUDA device is not available")

    assert backend.name == "cuda"
    assert backend.device() == torch.device("cuda", 0)
    assert backend.autocast_dtype() is torch.bfloat16

    tensor = torch.ones(8, device=backend.device(), dtype=backend.autocast_dtype())
    backend.synchronize()
    assert tensor.device.type == "cuda"
    assert tensor.sum().item() == 8

    _assert_rng_round_trip(backend)
