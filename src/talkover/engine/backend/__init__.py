"""Device backend abstraction: the DeviceBackend protocol and backend selection.

Every device operation in :mod:`talkover.engine` goes through :class:`DeviceBackend`.
``torch.cuda``, ``torch.mps`` and ``torch.backends.mps`` are referenced only inside
:mod:`talkover.engine.backend.cuda` and :mod:`talkover.engine.backend.mps`; those modules
are imported lazily so that importing this package never touches an unavailable device.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from talkover.engine.backend.cpu import CpuBackend
    from talkover.engine.backend.cuda import CudaBackend
    from talkover.engine.backend.mps import MpsBackend

__all__ = [
    "CpuBackend",
    "CudaBackend",
    "DeviceBackend",
    "MpsBackend",
    "get_backend",
]

_CUDA_SPEC = re.compile(r"^cuda(?::(\d+))?$")


@runtime_checkable
class DeviceBackend(Protocol):
    """Interface every compute device used by the engine is reached through.

    Implementations accept ``index=None`` on :meth:`device` to mean "the device this
    backend was built for", which is how ``cuda:<n>`` selection is carried around.
    """

    name: str

    def is_available(self) -> bool:
        """Whether this device is usable in the current process."""
        ...

    def device(self, index: int = 0) -> torch.device:
        """Return the ``torch.device`` this backend runs on."""
        ...

    def synchronize(self) -> None:
        """Block until all work queued on this device has finished."""
        ...

    def rng_state(self) -> Any:
        """Return the device RNG state, for reproducible token sequences."""
        ...

    def set_rng_state(self, state: Any) -> None:
        """Restore a state previously returned by :meth:`rng_state`."""
        ...

    def autocast_dtype(self) -> torch.dtype:
        """The autocast dtype for this device (bf16 everywhere so far)."""
        ...


def get_backend(device: str) -> DeviceBackend:
    """Build the backend named by ``device``.

    Accepted spellings: ``mps``, ``cpu``, ``cuda`` and ``cuda:<n>``. Availability is not
    checked here; call :meth:`DeviceBackend.is_available` (``talkover check`` does).

    Raises:
        ValueError: if ``device`` is not one of the accepted spellings.
        TypeError: if ``device`` is not a string.
    """
    if not isinstance(device, str):
        raise TypeError(f"device must be a string, got {type(device).__name__}")
    spec = device.strip().lower()

    if spec == "cpu":
        from talkover.engine.backend.cpu import CpuBackend

        return CpuBackend()

    if spec == "mps":
        from talkover.engine.backend.mps import MpsBackend

        return MpsBackend()

    match = _CUDA_SPEC.match(spec)
    if match is not None:
        from talkover.engine.backend.cuda import CudaBackend

        return CudaBackend(int(match.group(1) or 0))

    raise ValueError(
        f"unknown device {device!r}: expected one of 'mps', 'cuda', 'cuda:<n>' or 'cpu'"
    )


def __getattr__(name: str) -> Any:
    """Expose the backend classes without importing device-specific modules eagerly."""
    if name == "CpuBackend":
        from talkover.engine.backend.cpu import CpuBackend

        return CpuBackend
    if name == "MpsBackend":
        from talkover.engine.backend.mps import MpsBackend

        return MpsBackend
    if name == "CudaBackend":
        from talkover.engine.backend.cuda import CudaBackend

        return CudaBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
