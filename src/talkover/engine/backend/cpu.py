"""CPU device backend implementation, for CI smoke tests only."""

from __future__ import annotations

from typing import Any

import torch


class CpuBackend:
    """Device backend backed by the CPU.

    Only meant for GPU-free smoke tests; it is far too slow for the real-time budget.
    """

    name = "cpu"

    def is_available(self) -> bool:
        """The CPU is always available."""
        return True

    def device(self, index: int | None = None) -> torch.device:
        """Return the CPU device. Any index other than 0 is rejected."""
        if index not in (None, 0):
            raise ValueError(f"CPU backend has no device index {index}")
        return torch.device("cpu")

    def synchronize(self) -> None:
        """No-op: CPU execution is already synchronous."""

    def rng_state(self) -> Any:
        return torch.get_rng_state()

    def set_rng_state(self, state: Any) -> None:
        torch.set_rng_state(state)

    def autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16
