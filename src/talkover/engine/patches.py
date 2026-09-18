"""Minimal monkeypatches for upstream mcpmft / gander_runtime, kept in one place.

Upstream (``../Omni-Interaction-Agent``, pinned at ``cf43838``) is never forked. Every
binding point that blocks Talkover on a machine without CUDA is patched here, each by its
own function whose docstring names the upstream file, the line range, and the minimal
upstream patch it stands for. The same diffs are recorded in ``docs/mps-porting-notes.md``.

Ordering rule: :func:`apply_patches` must run before anything else imports ``mcpmft.infer``,
so that no module has captured an unpatched reference (``from ... import load_for_infer``).
Importing :mod:`talkover.engine.session` is what enforces that; the first call to
:func:`apply_patches` raises :class:`PatchOrderError` if ``mcpmft.infer`` is already loaded.

Patches are installed once per process and are idempotent. The active backend is held in a
module-level slot the patches read at call time, so calling :func:`apply_patches` again with
the configured backend (T1.4, once the YAML config is known) only swaps that slot.
"""

from __future__ import annotations

import functools
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from talkover.engine.backend import DeviceBackend

__all__ = [
    "PatchOrderError",
    "active_backend",
    "apply_patches",
    "patches_applied",
    "speech_worker_override",
]

_UPSTREAM_INFER = "mcpmft.infer"
_MARKER = "__talkover_patched__"

_lock = threading.RLock()
_active_backend: DeviceBackend | None = None
_applied = False


class PatchOrderError(RuntimeError):
    """Raised when upstream inference modules were imported before the patches ran."""


def patches_applied() -> bool:
    """Whether :func:`apply_patches` has already installed the patches in this process."""
    return _applied


def active_backend() -> DeviceBackend:
    """The backend the installed patches currently target.

    Raises:
        RuntimeError: if :func:`apply_patches` has not run yet.
    """
    if _active_backend is None:
        raise RuntimeError("talkover.engine.patches.apply_patches() has not been called")
    return _active_backend


def _check_import_order() -> None:
    """Fail loudly if upstream inference modules were imported before the patches."""
    loaded = sorted(
        name
        for name in sys.modules
        if name == _UPSTREAM_INFER or name.startswith(f"{_UPSTREAM_INFER}.")
    )
    if loaded:
        raise PatchOrderError(
            "talkover.engine.patches.apply_patches() must run before any mcpmft.infer "
            f"import, but these modules are already loaded: {', '.join(loaded)}. "
            "Import talkover.engine.session (which applies the patches) first."
        )


def apply_patches(backend: DeviceBackend) -> None:
    """Install every upstream patch, targeting ``backend``. Safe to call repeatedly.

    The first call must happen before ``mcpmft.infer`` is imported. Later calls only
    retarget the already installed patches, which is how the configured device replaces
    the one auto-detected at import time.
    """
    global _active_backend, _applied
    with _lock:
        _active_backend = backend
        if _applied:
            return
        _check_import_order()
        _patch_load_for_infer_device()
        _applied = True


def _patch_load_for_infer_device() -> None:
    """Retarget the device handling at the end of upstream ``load_for_infer``.

    Upstream file: ``minicpm_ft/mcpmft/infer/common.py`` lines 83-89 (commit ``cf43838``).
    The tail of ``load_for_infer`` refuses to run unless the CUDA availability check
    passes, and then moves the model to the hard-coded string ``"cuda"``. On a machine
    without CUDA that raises ``RuntimeError("Gander inference requires CUDA")`` as soon as
    the model is constructed.

    Minimal upstream patch: drop the availability check and take the destination from a
    device resolver instead of the literal ``"cuda"``::

        -    if not <the CUDA availability gate>:
        -        raise RuntimeError("Gander inference requires CUDA")
         if inference_args.device_map is None:
        -        model.to("cuda")
        +        model.to(infer_device())

    The full unified diff is in ``docs/mps-porting-notes.md`` (2026-09-17, Patch 1); it is
    kept out of this file because the lint guard in ``tests/engine/test_backend.py`` forbids
    device-specific torch attribute names outside ``engine/backend/``, including in
    docstrings.

    Talkover cannot replace those two statements individually: they sit inside the
    function body and resolve ``torch`` through a function-local import. So
    ``load_for_infer`` is wrapped, and the wrapper runs the original inside
    :func:`talkover.engine.backend.shims.device_placement_shim`, which makes the
    availability gate pass and rewrites the CUDA destination to ``backend.device()`` for
    the duration of that one call. The shim lives under ``engine/backend/`` because it is
    the only layer allowed to name device-specific torch attributes (DESIGN.md 4.2).

    The wrapper is installed at most once; re-running :func:`apply_patches` is a no-op
    because the wrapper reads the active backend when it is called.
    """
    import mcpmft.infer.common as upstream_common

    from talkover.engine.backend.shims import device_placement_shim

    original = upstream_common.load_for_infer
    if getattr(original, _MARKER, False):
        return

    @functools.wraps(original)
    def load_for_infer(*args: Any, **kwargs: Any) -> Any:
        with device_placement_shim(active_backend()):
            return original(*args, **kwargs)

    setattr(load_for_infer, _MARKER, True)
    upstream_common.load_for_infer = load_for_infer


@contextmanager
def speech_worker_override(worker: Any) -> Iterator[None]:
    """Make one ``DuplexLiveSession`` construction adopt an existing Talker thread.

    Upstream file: ``minicpm_ft/mcpmft/infer/realtime.py`` lines 206-209 (commit
    ``cf43838``). ``DuplexLiveSession.__init__`` builds its background Talker itself::

        self.speech_worker = (
            AsyncTalkerWorker(detached_talker) if detached_talker is not None else None
        )

    Talkover brings its own worker (:class:`talkover.engine.talker.TalkerThread`), which
    adds the ``DeviceBackend.synchronize`` separation between the Thinker step and the
    Talker thread (T1.5). There is no parameter for it, and assigning ``speech_worker``
    after construction would leave upstream's worker thread started and then thrown away,
    so the class the constructor looks up is replaced for the duration of that one call.

    Minimal upstream patch: let the caller supply the worker::

        -    def __init__(self, bundle, *, ..., detached_talker=None, ...):
        +    def __init__(self, bundle, *, ..., detached_talker=None,
        +                 speech_worker_factory=AsyncTalkerWorker, ...):
        -        self.speech_worker = (
        -            AsyncTalkerWorker(detached_talker) if detached_talker is not None else None
        -        )
        +        self.speech_worker = (
        +            speech_worker_factory(detached_talker) if detached_talker is not None else None
        +        )

    The override is scoped: it must wrap a single ``DuplexLiveSession(...)`` call, because
    it is process-wide while active and ignores the runtime upstream passes to it.
    """
    import mcpmft.infer.realtime as upstream_realtime

    original = upstream_realtime.AsyncTalkerWorker

    def factory(_detached_talker: Any) -> Any:
        return worker

    upstream_realtime.AsyncTalkerWorker = factory
    try:
        yield
    finally:
        upstream_realtime.AsyncTalkerWorker = original
