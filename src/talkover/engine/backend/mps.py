"""Apple Silicon Metal device backend implementation."""

from __future__ import annotations

from typing import Any

import torch


class MpsBackend:
    """Device backend backed by Apple Silicon Metal (MPS).

    This is the only module in the engine package allowed to touch ``torch.mps``
    and ``torch.backends.mps``.
    """

    name = "mps"

    def is_available(self) -> bool:
        return torch.backends.mps.is_available()

    def device(self, index: int | None = None) -> torch.device:
        """Return the MPS device. MPS exposes a single device, so only index 0 is valid."""
        if index not in (None, 0):
            raise ValueError(f"MPS backend has no device index {index}")
        return torch.device("mps")

    def synchronize(self) -> None:
        torch.mps.synchronize()

    def rng_state(self) -> Any:
        return torch.mps.get_rng_state()

    def set_rng_state(self, state: Any) -> None:
        torch.mps.set_rng_state(state)

    def autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16
