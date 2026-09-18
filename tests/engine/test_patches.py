"""Tests for the upstream monkeypatches (T1.3) and the import-order hook in session.py.

The import-order behaviour is only real in a fresh interpreter, so those cases run in a
subprocess. The rest exercise the shim and the backend selection in-process.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from talkover.engine.backend import get_backend
from talkover.engine.backend.shims import device_placement_shim


def _run(source: str, **env: str) -> subprocess.CompletedProcess[str]:
    """Run ``source`` in a fresh interpreter with the project environment."""
    child_env = {**os.environ, **env}
    child_env.pop("PYTHONWARNINGS", None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
        env=child_env,
        check=False,
    )


# --- device_placement_shim -------------------------------------------------------


@pytest.mark.cpu
def test_shim_retargets_module_placement_and_availability_gate() -> None:
    """A stand-in for upstream's load tail: the gate passes and "cuda" lands on the backend."""
    backend = get_backend("cpu")
    module = torch.nn.Linear(2, 2)

    def upstream_tail() -> torch.device:
        # Mirrors mcpmft/infer/common.py lines 83-89.
        if not torch.cuda.is_available():
            raise RuntimeError("Gander inference requires CUDA")
        module.to("cuda")
        return next(module.parameters()).device

    with pytest.raises(RuntimeError, match="requires CUDA"):
        upstream_tail()

    with device_placement_shim(backend):
        assert upstream_tail() == backend.device()

    # Both globals are restored on exit.
    with pytest.raises(RuntimeError, match="requires CUDA"):
        upstream_tail()


@pytest.mark.cpu
def test_shim_restores_globals_on_exception() -> None:
    original_to = torch.nn.Module.to
    original_is_available = torch.cuda.is_available
    with pytest.raises(ValueError, match="boom"), device_placement_shim(get_backend("cpu")):
        raise ValueError("boom")
    assert torch.nn.Module.to is original_to
    assert torch.cuda.is_available is original_is_available


@pytest.mark.cpu
def test_shim_is_a_no_op_for_cuda() -> None:
    original_to = torch.nn.Module.to
    with device_placement_shim(get_backend("cuda")):
        assert torch.nn.Module.to is original_to


@pytest.mark.cpu
def test_shim_leaves_other_destinations_alone() -> None:
    module = torch.nn.Linear(2, 2)
    with device_placement_shim(get_backend("cpu")):
        module.to(torch.float64)
    assert next(module.parameters()).dtype is torch.float64


# --- import order ----------------------------------------------------------------


@pytest.mark.cpu
def test_session_import_patches_before_upstream_import() -> None:
    """Acceptance: session imports on a CUDA-less machine and upstream then imports cleanly."""
    result = _run(
        """
        import sys

        assert not [m for m in sys.modules if m.startswith("mcpmft.infer")]
        import talkover.engine.session  # noqa: F401

        import mcpmft.infer.common as common
        import mcpmft.infer.realtime as realtime

        assert getattr(common.load_for_infer, "__talkover_patched__", False)
        assert realtime.DuplexLiveSession is not None
        print("OK")
        """,
        TALKOVER_DEVICE="cpu",
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.cpu
def test_apply_patches_rejects_a_late_call() -> None:
    result = _run(
        """
        import mcpmft.infer.common  # imported too early, on purpose

        from talkover.engine.backend import get_backend
        from talkover.engine.patches import PatchOrderError, apply_patches

        try:
            apply_patches(get_backend("cpu"))
        except PatchOrderError as exc:
            assert "mcpmft.infer.common" in str(exc)
            print("RAISED")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout


@pytest.mark.cpu
def test_apply_patches_is_idempotent_and_retargets() -> None:
    result = _run(
        """
        from talkover.engine.backend import get_backend
        from talkover.engine.patches import active_backend, apply_patches, patches_applied

        assert not patches_applied()
        apply_patches(get_backend("cpu"))
        import mcpmft.infer.common as common

        first = common.load_for_infer
        assert patches_applied()

        apply_patches(get_backend("cpu"))
        assert common.load_for_infer is first, "patch must not be wrapped twice"

        apply_patches(get_backend("cuda:1"))
        assert common.load_for_infer is first
        assert active_backend().name == "cuda"
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.cpu
def test_active_backend_before_apply_patches_raises() -> None:
    result = _run(
        """
        from talkover.engine.patches import active_backend

        try:
            active_backend()
        except RuntimeError as exc:
            assert "apply_patches" in str(exc)
            print("RAISED")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "RAISED" in result.stdout


# --- import-time backend selection -----------------------------------------------


@pytest.mark.cpu
def test_default_backend_honours_the_environment_variable() -> None:
    result = _run(
        """
        from talkover.engine.session import default_backend

        backend = default_backend()
        assert backend.name == "cuda", backend.name
        assert backend.index == 1, backend.index
        print("OK")
        """,
        TALKOVER_DEVICE="cuda:1",
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


@pytest.mark.cpu
def test_default_backend_rejects_an_unknown_device() -> None:
    result = _run(
        """
        from talkover.engine.session import default_backend  # noqa: F401
        """,
        TALKOVER_DEVICE="rocm",
    )
    assert result.returncode != 0
    assert "unknown device" in result.stderr


@pytest.mark.cpu
def test_default_backend_auto_detects_an_available_device() -> None:
    result = _run(
        """
        import os

        os.environ.pop("TALKOVER_DEVICE", None)
        from talkover.engine.session import default_backend

        backend = default_backend()
        assert backend.name in {"mps", "cuda", "cpu"}
        assert backend.is_available()
        print(backend.name)
        """,
        TALKOVER_DEVICE="",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() in {"mps", "cuda", "cpu"}


@pytest.mark.mps
def test_default_backend_picks_mps_on_apple_silicon() -> None:
    if not get_backend("mps").is_available():
        pytest.skip("MPS device is not available")
    result = _run(
        """
        from talkover.engine.session import default_backend

        print(default_backend().name)
        """,
        TALKOVER_DEVICE="",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "mps"
