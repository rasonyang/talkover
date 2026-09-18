"""NVIDIA CUDA device backend implementation, supporting a detached Talker on multiple GPUs."""

from __future__ import annotations

from typing import Any

import torch


class CudaBackend:
    """Device backend backed by a single NVIDIA CUDA device.

    The device ordinal is fixed at construction time (``get_backend("cuda:1")``), so
    ``synchronize`` and the RNG state always apply to that card. A detached Talker on a
    second card is served by a second ``CudaBackend`` instance with its own index.

    This is the only module in the engine package allowed to touch ``torch.cuda``.
    """

    name = "cuda"

    def __init__(self, index: int = 0) -> None:
        if index < 0:
            raise ValueError(f"CUDA device index must not be negative: {index}")
        self.index = index

    def __repr__(self) -> str:
        return f"CudaBackend(index={self.index})"

    def is_available(self) -> bool:
        return torch.cuda.is_available() and self.index < torch.cuda.device_count()

    def device(self, index: int | None = None) -> torch.device:
        """Return this backend's device, or the one named by an explicit index."""
        return torch.device("cuda", self.index if index is None else index)

    def synchronize(self) -> None:
        torch.cuda.synchronize(self.device())

    def rng_state(self) -> Any:
        return torch.cuda.get_rng_state(self.device())

    def set_rng_state(self, state: Any) -> None:
        torch.cuda.set_rng_state(state, self.device())

    def autocast_dtype(self) -> torch.dtype:
        return torch.bfloat16
