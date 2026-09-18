"""Scoped torch shims used by the upstream monkeypatches (see ``talkover.engine.patches``).

Upstream ``load_for_infer`` hard-codes CUDA: it refuses to run unless the CUDA
availability check passes, and then moves the model to the literal string ``"cuda"``.
Neither statement can be replaced on its own, because both live in the middle of the
function body and resolve ``torch`` through a function-local ``import torch``. The only
way to retarget them without forking upstream is to patch the two torch globals they
reach, for the duration of that one call.

Those globals are device specific, so the helper lives in ``engine/backend/`` — the only
place in the engine package allowed to name device-specific torch attributes (DESIGN.md
4.2). The patches are installed by a context manager and always restored, and they are
process-wide while active, so a block must wrap a single model-loading call and nothing
else.

:func:`talker_device_shim` does the same job for the Talker stack (T1.5): upstream's
``detached_talker.py`` and the third-party ``stepaudio2`` vocoder are written for CUDA
only, and the Talker thread holds that block for its whole life. Its docstring explains
why a long-lived, process-wide block is safe there.
"""

from __future__ import annotations

import contextlib
import functools
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from talkover.engine.backend import DeviceBackend

__all__ = ["device_placement_shim", "talker_device_shim"]


def _retarget(value: Any, target: torch.device) -> Any:
    """Map a CUDA destination onto ``target``; anything else is returned unchanged."""
    if isinstance(value, str) and (value == "cuda" or value.startswith("cuda:")):
        return target
    if isinstance(value, torch.device) and value.type == "cuda":
        return target
    return value


@contextmanager
def device_placement_shim(backend: DeviceBackend) -> Iterator[None]:
    """Make an upstream CUDA-only load path run on ``backend``'s device.

    While the block is active:

    - ``torch.cuda.is_available`` reports ``True``, so upstream's availability gate passes.
    - ``torch.nn.Module.to`` rewrites a CUDA destination to ``backend.device()``.

    Both originals are restored on exit, including on exception. The shim is a no-op for a
    CUDA backend, which needs no retargeting.
    """
    if backend.name == "cuda":
        yield
        return

    target = backend.device()
    original_is_available = torch.cuda.is_available
    original_to = torch.nn.Module.to

    def patched_is_available() -> bool:
        return True

    def patched_to(self: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        new_args = tuple(_retarget(arg, target) for arg in args)
        new_kwargs = dict(kwargs)
        if "device" in new_kwargs:
            new_kwargs["device"] = _retarget(new_kwargs["device"], target)
        return original_to(self, *new_args, **new_kwargs)

    torch.cuda.is_available = patched_is_available
    torch.nn.Module.to = patched_to
    try:
        yield
    finally:
        torch.cuda.is_available = original_is_available
        torch.nn.Module.to = original_to


_CUDA_FACTORIES = (
    "tensor",
    "as_tensor",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "linspace",
    "rand",
    "randn",
    "zeros_like",
    "ones_like",
    "empty_like",
    "full_like",
)

_AUTOCAST_DTYPES = (torch.float16, torch.bfloat16)

_talker_lock = threading.RLock()
_talker_depth = 0
_talker_backend_name: str | None = None
_talker_restore: list[Callable[[], None]] = []


def _retarget_device_kwarg(kwargs: dict[str, Any], target: torch.device) -> dict[str, Any]:
    """Return ``kwargs`` with a CUDA ``device=`` entry rewritten to ``target``."""
    device = kwargs.get("device")
    if device is None:
        return kwargs
    retargeted = _retarget(device, target)
    if retargeted is device:
        return kwargs
    patched = dict(kwargs)
    patched["device"] = retargeted
    return patched


def _install_talker_shim(backend: DeviceBackend) -> list[Callable[[], None]]:
    """Patch the CUDA-only call sites the Talker stack reaches; return the undo list."""
    target = backend.device()
    restore: list[Callable[[], None]] = []

    # 1. torch factory functions called with device="cuda" (stepaudio2 token2wav).
    for name in _CUDA_FACTORIES:
        original = getattr(torch, name, None)
        if original is None:  # pragma: no cover - depends on the torch build
            continue

        def make(original: Any = original) -> Any:
            @functools.wraps(original)
            def factory(*args: Any, **kwargs: Any) -> Any:
                return original(*args, **_retarget_device_kwarg(kwargs, target))

            return factory

        setattr(torch, name, make())
        restore.append(functools.partial(setattr, torch, name, original))

    # 2. ``.cuda()`` on tensors and modules. MPS has no float64, so a float64 buffer
    #    (``np.hamming`` in token2wav) is narrowed to float32 on the way over.
    original_tensor_cuda = torch.Tensor.cuda
    original_module_cuda = torch.nn.Module.cuda
    narrow_float64 = target.type == "mps"

    def tensor_cuda(self: torch.Tensor, device: Any = None, **kwargs: Any) -> torch.Tensor:
        if narrow_float64 and self.dtype == torch.float64:
            return self.to(device=target, dtype=torch.float32)
        return self.to(target)

    def module_cuda(self: torch.nn.Module, device: Any = None) -> torch.nn.Module:
        return self.to(target)

    torch.Tensor.cuda = tensor_cuda  # type: ignore[method-assign]
    torch.nn.Module.cuda = module_cuda  # type: ignore[method-assign]
    restore.append(functools.partial(setattr, torch.Tensor, "cuda", original_tensor_cuda))
    restore.append(functools.partial(setattr, torch.nn.Module, "cuda", original_module_cuda))

    # 3. ``torch.amp.autocast("cuda", ...)``: retarget, and disable it outright when the
    #    device cannot autocast the requested dtype (MPS only supports fp16 / bf16).
    original_autocast = torch.amp.autocast

    def autocast(device_type: str = "cuda", dtype: Any = None, **kwargs: Any) -> Any:
        if device_type == "cuda":
            if target.type != "cuda" and dtype is not None and dtype not in _AUTOCAST_DTYPES:
                return contextlib.nullcontext()
            device_type = target.type
        return original_autocast(device_type, dtype=dtype, **kwargs)

    torch.amp.autocast = autocast  # type: ignore[assignment]
    restore.append(functools.partial(setattr, torch.amp, "autocast", original_autocast))

    # 4. The device-scope, RNG and synchronize calls in upstream ``detached_talker.py``.
    original_cuda_device = torch.cuda.device
    original_synchronize = torch.cuda.synchronize
    original_get_rng_state = torch.cuda.get_rng_state
    original_set_rng_state = torch.cuda.set_rng_state

    @contextmanager
    def cuda_device(device: Any = None) -> Iterator[None]:
        yield

    def synchronize(device: Any = None) -> None:
        backend.synchronize()

    def get_rng_state(device: Any = None) -> Any:
        return backend.rng_state()

    def set_rng_state(state: Any, device: Any = None) -> None:
        backend.set_rng_state(state)

    torch.cuda.device = cuda_device  # type: ignore[assignment]
    torch.cuda.synchronize = synchronize  # type: ignore[assignment]
    torch.cuda.get_rng_state = get_rng_state  # type: ignore[assignment]
    torch.cuda.set_rng_state = set_rng_state  # type: ignore[assignment]
    restore.append(functools.partial(setattr, torch.cuda, "device", original_cuda_device))
    restore.append(functools.partial(setattr, torch.cuda, "synchronize", original_synchronize))
    restore.append(functools.partial(setattr, torch.cuda, "get_rng_state", original_get_rng_state))
    restore.append(functools.partial(setattr, torch.cuda, "set_rng_state", original_set_rng_state))
    return restore


@contextmanager
def talker_device_shim(backend: DeviceBackend) -> Iterator[None]:
    """Run the Talker and token2wav stack on ``backend``'s device instead of CUDA.

    The Talker stack reaches CUDA from two directions, neither of which can be replaced
    statement by statement without forking:

    - upstream ``mcpmft/infer/detached_talker.py`` wraps every runtime call in a CUDA
      device scope and reads the CUDA RNG state around ``warm_token2wav``;
    - the third-party vocoder ``stepaudio2.token2wav`` hard-codes ``.cuda()``,
      ``device="cuda"`` and ``torch.amp.autocast("cuda", ...)``.

    While the block is active those call sites are redirected to ``backend``: the device
    scope becomes a no-op, the RNG and synchronize calls go through
    :class:`~talkover.engine.backend.DeviceBackend`, ``device="cuda"`` and ``.cuda()``
    resolve to ``backend.device()``, and an autocast the device cannot do is disabled.

    The patches are process-wide while active and the Talker thread holds the block for
    its whole life, so the Thinker thread sees them too. That is safe because every
    replacement is a correct implementation for any thread on a non-CUDA backend: nothing
    else in the process has a working CUDA device to reach. The shim is a no-op on a CUDA
    backend, and it nests (the loader enters it while the Talker thread holds it).

    Raises:
        RuntimeError: if a second backend is requested while the shim is active.
    """
    global _talker_depth, _talker_backend_name
    if backend.name == "cuda":
        yield
        return

    with _talker_lock:
        if _talker_depth and _talker_backend_name != backend.name:
            raise RuntimeError(
                "talker_device_shim is already active for backend "
                f"{_talker_backend_name!r}; it cannot also target {backend.name!r}"
            )
        if _talker_depth == 0:
            _talker_restore.extend(_install_talker_shim(backend))
            _talker_backend_name = backend.name
        _talker_depth += 1
    try:
        yield
    finally:
        with _talker_lock:
            _talker_depth -= 1
            if _talker_depth == 0:
                for undo in reversed(_talker_restore):
                    undo()
                _talker_restore.clear()
                _talker_backend_name = None
