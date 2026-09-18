"""In-process Talker thread, separated by the device backend's synchronize.

Gander splits speech generation in two. The Thinker decides, per one-second unit, whether
to listen or speak and emits a handful of speak tokens plus their hidden states; the
Talker turns that condition into 25 S3 audio tokens, and token2wav vocodes them into a
24 kHz waveform. Upstream runs those two stages in one of two ways:

- **in step** (``DuplexParams.generate_audio=True``, ``detached_talker=None``): the Talker
  and token2wav run inside ``OnlineRunner.streaming_generate``, on the Thinker's thread,
  and the waveform comes back on the step event. This is what T1.4 uses;
- **detached** (``mcpmft/infer/detached_talker.py``): a ``DetachedTalkerRuntime`` holds its
  own Talker KV and token2wav caches, an ``AsyncTalkerWorker`` drives it on a background
  thread, and ``DuplexLiveSession`` hands it the speak tokens through a queue. Upstream
  built that for multi-GPU CUDA and the class is CUDA-only.

Talkover (DESIGN.md 4.0 / 4.2) wants the second shape on **one** device: the Thinker step
must not wait for the vocoder, but there is only one GPU. So this module reuses upstream's
runtime — no fork — with two adapters:

- :func:`talkover.engine.backend.shims.talker_device_shim` redirects the CUDA-only call
  sites — upstream's device scopes, the device RNG state read around ``warm_token2wav`` and
  its device synchronize, plus ``stepaudio2``'s hard-coded ``.cuda()``, ``device="cuda"``
  and ``autocast("cuda")`` — onto the configured :class:`DeviceBackend`. The call sites are
  listed by file and line in ``docs/mps-porting-notes.md``, which is where they may be
  spelled out: the lint guard in ``tests/engine/test_backend.py`` forbids device-specific
  torch attribute names anywhere in the engine outside ``backend/``, docstrings included;
- :class:`TalkerThread` replaces upstream's ``AsyncTalkerWorker`` with a worker that
  separates the two stages with ``backend.synchronize()``: once on the Thinker thread when
  a request is handed over (so the hidden states are materialized before the Talker thread
  reads them) and once on the Talker thread when a request is finished (so a waveform is
  materialized before it is published). Those two calls are the only device work the
  Thinker ever blocks on for the Talker.

:class:`TalkerThread` keeps upstream's worker interface (``submit`` / ``cancel`` / ``poll``
/ ``drain_outputs`` / ``state`` / ``wait_until_drained`` / ``close`` / ``generation_id``),
so ``DuplexLiveSession`` drives it unchanged; ``talkover.engine.patches`` has the scoped
override that puts it in place of upstream's worker.

The multi-GPU branch is kept reachable but not implemented: when ``engine.talker_device``
names a device other than ``engine.device``, :func:`build_talker_runtime` raises
``NotImplementedError`` pointing at M4, where upstream
``DetachedTalkerRuntime.from_thinker_model`` (CUDA only) is the path to take.

Nothing here imports ``mcpmft`` at module level: importing this module must not defeat the
patch-order rule in :mod:`talkover.engine.patches`.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from talkover.config import TalkoverConfig
from talkover.engine.backend import DeviceBackend
from talkover.engine.backend.shims import talker_device_shim
from talkover.engine.protocol import OUTPUT_SAMPLE_RATE

__all__ = [
    "SILENCE_TOKEN_ID",
    "SPEECH_TOKENS_PER_UNIT",
    "TalkerChunk",
    "TalkerDone",
    "TalkerFailed",
    "TalkerInterrupted",
    "TalkerOutput",
    "TalkerRequest",
    "TalkerRuntime",
    "TalkerThread",
    "build_talker_runtime",
    "build_token2wav",
    "talker_factory_from_config",
]

LOGGER = logging.getLogger(__name__)

#: S3 token the vocoder treats as silence (upstream ``DetachedTalkerConfig.silence_token_id``).
SILENCE_TOKEN_ID = 4218
#: S3 tokens the Talker emits per one-second unit (upstream ``speech_tokens_per_unit``).
SPEECH_TOKENS_PER_UNIT = 25

_CLOSE_JOIN_TIMEOUT_SEC = 30.0


# --------------------------------------------------------------------------------------
# What crosses the queue
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TalkerRequest:
    """One unit of speak tokens handed from the Thinker to the Talker.

    The field names match upstream ``SpeechSynthesisRequest``, so an upstream runtime
    reads this object without a conversion.
    """

    generation_id: int
    unit_id: int
    current_time: int | None
    token_ids: tuple[int, ...]
    hidden_states: Any
    end_of_turn: bool


@dataclass(frozen=True, slots=True)
class TalkerChunk:
    """A finished piece of waveform, always 1-D float32 at :data:`OUTPUT_SAMPLE_RATE`."""

    generation_id: int
    unit_id: int
    sequence: int
    waveform: np.ndarray
    sample_rate: int = OUTPUT_SAMPLE_RATE
    current_time: int | None = None
    end_of_turn: bool = False
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TalkerDone:
    """One request finished; :attr:`metrics` carries the runtime's per-stage timings."""

    generation_id: int
    unit_id: int
    end_of_turn: bool
    metrics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TalkerInterrupted:
    """Everything queued under :attr:`cancelled_generation_id` was dropped."""

    generation_id: int
    cancelled_generation_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class TalkerFailed:
    """The runtime raised while synthesizing; the thread stays usable."""

    generation_id: int
    unit_id: int
    message: str


TalkerOutput = TalkerChunk | TalkerDone | TalkerInterrupted | TalkerFailed


@runtime_checkable
class TalkerRuntime(Protocol):
    """The device-side half of the Talker, as :class:`TalkerThread` drives it.

    Upstream ``DetachedTalkerRuntime`` satisfies this; so does the fake the ``cpu`` tests
    use. ``synthesize`` must call ``emit`` for every finished waveform chunk and must stop
    (by raising) when ``cancel_event`` is set.
    """

    def reset(self) -> None:
        """Drop the Talker KV cache and the token2wav stream state."""
        ...

    def synthesize(
        self,
        request: Any,
        cancel_event: threading.Event,
        emit: Callable[[Any], None],
    ) -> Mapping[str, Any]:
        """Turn one request into waveform chunks and return its metrics."""
        ...


def _as_float32_waveform(value: Any) -> np.ndarray:
    """Normalize a chunk waveform (numpy, torch tensor or bytes) to 1-D float32."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(value), dtype="<i2").astype(np.float32) / 32768.0
    detach = getattr(value, "detach", None)
    if callable(detach):  # a torch tensor, without naming torch here
        value = detach().to("cpu").numpy()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if np.issubdtype(np.asarray(value).dtype, np.integer):
        array = array / 32768.0
    return array


def _to_chunk(value: Any, *, request: TalkerRequest, sequence: int) -> TalkerChunk:
    """Convert whatever the runtime emitted into a :class:`TalkerChunk`."""
    waveform = _as_float32_waveform(getattr(value, "waveform", value))
    return TalkerChunk(
        generation_id=int(getattr(value, "generation_id", request.generation_id)),
        unit_id=int(getattr(value, "unit_id", request.unit_id)),
        sequence=int(getattr(value, "sequence", sequence)),
        waveform=waveform,
        sample_rate=int(getattr(value, "sample_rate", OUTPUT_SAMPLE_RATE)),
        current_time=getattr(value, "current_time", request.current_time),
        end_of_turn=bool(getattr(value, "end_of_turn", False)),
        metrics=dict(getattr(value, "metrics", {}) or {}),
    )


@dataclass(frozen=True, slots=True)
class _Queued:
    """A request together with the cancellation epoch it was submitted under."""

    request: TalkerRequest
    cancel_event: threading.Event


# --------------------------------------------------------------------------------------
# The thread
# --------------------------------------------------------------------------------------


class TalkerThread:
    """Runs a :class:`TalkerRuntime` on its own thread, on the Thinker's device.

    One instance serves one session. Requests are served in submission order; a
    :meth:`cancel` opens a new generation and everything still queued under the old one is
    dropped, which is how barge-in stops the vocoder within a unit.

    The interface is the one ``mcpmft.infer.realtime.DuplexLiveSession`` expects from
    ``AsyncTalkerWorker``, so upstream drives this class unchanged.

    Args:
        runtime: the device-side Talker; see :class:`TalkerRuntime`.
        backend: the device backend the runtime runs on. Its ``synchronize`` is what
            separates the Thinker step from the Talker thread.
        name: thread name, for logs and profiles.
        hold_device_shim: keep :func:`talker_device_shim` entered for the life of the
            thread. True for a real runtime (upstream's runtime and ``stepaudio2`` are
            CUDA-only); the fake runtimes in the tests do not need it.
    """

    def __init__(
        self,
        runtime: TalkerRuntime,
        backend: DeviceBackend,
        *,
        name: str = "talkover-talker",
        hold_device_shim: bool = True,
    ) -> None:
        self._runtime = runtime
        self._backend = backend
        self._hold_device_shim = hold_device_shim
        self._requests: queue.Queue[_Queued | None] = queue.Queue()
        self._outputs: queue.Queue[TalkerOutput] = queue.Queue()
        self._lock = threading.Lock()
        self._drained = threading.Condition(self._lock)
        self._generation_id = 1
        self._cancel_event = threading.Event()
        self._pending = 0
        self._active = False
        self._closed = False
        self._runtime_generation: int | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    # -- introspection -------------------------------------------------------------

    @property
    def runtime(self) -> TalkerRuntime:
        """The device-side runtime this thread drives."""
        return self._runtime

    @property
    def backend(self) -> DeviceBackend:
        """The device backend the Talker runs on."""
        return self._backend

    @property
    def generation_id(self) -> int:
        """The current output generation; :meth:`cancel` bumps it."""
        with self._lock:
            return self._generation_id

    def state(self) -> dict[str, Any]:
        """The state dict upstream reports as ``metrics["talker"]``."""
        with self._lock:
            return {
                "active": self._active,
                "drained": not self._active and self._pending == 0,
                "pending_requests": self._pending,
                "generation_id": self._generation_id,
                "pending_output_events": self._outputs.qsize(),
            }

    # -- the Thinker side ----------------------------------------------------------

    def submit(
        self,
        *,
        unit_id: int,
        current_time: int | None,
        token_ids: Sequence[int],
        hidden_states: Any,
        end_of_turn: bool,
    ) -> TalkerRequest:
        """Hand one unit of speak tokens to the Talker thread.

        Called on the Thinker thread at the end of a step. The device queue is drained
        first, so the hidden states this request points at are finished before the Talker
        thread starts reading them; that ``synchronize`` is the whole of the Thinker's
        wait on the Talker.

        Raises:
            RuntimeError: if the thread has been closed.
        """
        self._backend.synchronize()
        with self._lock:
            if self._closed:
                raise RuntimeError("the Talker thread is closed")
            request = TalkerRequest(
                generation_id=self._generation_id,
                unit_id=int(unit_id),
                current_time=current_time,
                token_ids=tuple(int(value) for value in token_ids),
                hidden_states=hidden_states,
                end_of_turn=bool(end_of_turn),
            )
            queued = _Queued(request=request, cancel_event=self._cancel_event)
            self._pending += 1
        self._requests.put(queued)
        return request

    def cancel(self, reason: str) -> TalkerInterrupted:
        """Drop everything queued for the current generation and open the next one."""
        with self._lock:
            cancelled = self._generation_id
            self._cancel_event.set()
            self._generation_id += 1
            self._cancel_event = threading.Event()
            event = TalkerInterrupted(
                generation_id=self._generation_id,
                cancelled_generation_id=cancelled,
                reason=reason,
            )
        LOGGER.info(
            "talker cancelled generation=%d next_generation=%d reason=%s",
            cancelled,
            event.generation_id,
            reason,
        )
        self._outputs.put(event)
        return event

    # -- the consumer side ---------------------------------------------------------

    def poll(self, timeout: float = 0.0) -> TalkerOutput | None:
        """Take the next output, or ``None`` when none arrives within ``timeout``."""
        try:
            return self._outputs.get(timeout=max(0.0, timeout))
        except queue.Empty:
            return None

    def drain_outputs(self) -> list[TalkerOutput]:
        """Take every output produced so far, in order."""
        outputs: list[TalkerOutput] = []
        while True:
            value = self.poll()
            if value is None:
                return outputs
            outputs.append(value)

    def wait_until_drained(self, timeout: float | None = None) -> bool:
        """Block until nothing is queued or running; ``False`` on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._drained:
            while self._active or self._pending:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._drained.wait(remaining)
            return True

    # -- lifecycle -----------------------------------------------------------------

    def close(self, *, drain: bool = False, timeout: float = _CLOSE_JOIN_TIMEOUT_SEC) -> None:
        """Stop the thread. With ``drain`` it first waits for the queue to empty.

        Safe to call more than once.

        Raises:
            RuntimeError: if the thread is still alive after ``timeout``.
        """
        drained = self.wait_until_drained(timeout) if drain else False
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if not drained:
                self._cancel_event.set()
        self._requests.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():  # pragma: no cover - only on a wedged device call
            raise RuntimeError("the Talker thread did not stop before the timeout")

    # -- the thread ----------------------------------------------------------------

    def _run(self) -> None:
        """Serve requests until :meth:`close`, with the device shim held throughout."""
        if not self._hold_device_shim:
            self._serve()
            return
        with talker_device_shim(self._backend):
            self._serve()

    def _serve(self) -> None:
        while True:
            queued = self._requests.get()
            if queued is None:
                return
            with self._drained:
                self._pending -= 1
                if queued.cancel_event.is_set():
                    self._drained.notify_all()
                    continue
                self._active = True
            try:
                self._synthesize(queued)
            finally:
                with self._drained:
                    self._active = False
                    self._drained.notify_all()

    def _synthesize(self, queued: _Queued) -> None:
        """Run one request, publish its chunks, and report how it ended."""
        request = queued.request
        sequence = 0

        def emit(value: Any) -> None:
            nonlocal sequence
            if queued.cancel_event.is_set():
                return
            sequence += 1
            self._outputs.put(_to_chunk(value, request=request, sequence=sequence))

        try:
            if self._runtime_generation != request.generation_id:
                self._runtime.reset()
                self._runtime_generation = request.generation_id
            metrics = self._runtime.synthesize(request, queued.cancel_event, emit)
            # The waveform must be off the device before the consumer is told the request
            # is finished; this is the second half of the Thinker/Talker separation.
            self._backend.synchronize()
        except BaseException as exc:  # a failure is reported; the thread stays usable
            if queued.cancel_event.is_set():
                LOGGER.info("talker request cancelled mid-synthesis: %s", exc)
            else:
                LOGGER.exception(
                    "talker failed for generation=%d unit=%d",
                    request.generation_id,
                    request.unit_id,
                )
            self._reset_after_failure()
            if not queued.cancel_event.is_set():
                self._outputs.put(
                    TalkerFailed(
                        generation_id=request.generation_id,
                        unit_id=request.unit_id,
                        message=str(exc),
                    )
                )
            return

        if queued.cancel_event.is_set():
            return
        if sequence == 0:
            LOGGER.warning(
                "talker produced no audio generation=%d unit=%d end_of_turn=%s",
                request.generation_id,
                request.unit_id,
                request.end_of_turn,
            )
        self._outputs.put(
            TalkerDone(
                generation_id=request.generation_id,
                unit_id=request.unit_id,
                end_of_turn=request.end_of_turn,
                metrics=dict(metrics or {}),
            )
        )

    def _reset_after_failure(self) -> None:
        """Put the runtime back into a usable state after a failed request."""
        self._runtime_generation = None
        try:
            self._runtime.reset()
        except Exception:  # pragma: no cover - reset failing is already terminal
            LOGGER.exception("resetting the Talker runtime after a failure also failed")


# --------------------------------------------------------------------------------------
# Building the real runtime
# --------------------------------------------------------------------------------------


def build_token2wav(
    token2wav_dir: str | Path,
    backend: DeviceBackend,
    *,
    float16: bool = False,
    n_timesteps: int = 10,
) -> Any:
    """Load the ``stepaudio2`` vocoder onto ``backend``'s device.

    ``stepaudio2.Token2wav`` places every module and buffer with ``.cuda()``; the load runs
    inside :func:`talker_device_shim`, which retargets those calls. The returned object is
    the upstream vocoder, unmodified, with its weights on ``backend.device()``.

    Every later call into it — ``set_stream_cache``, ``stream``, ``__call__`` — allocates
    with ``device="cuda"`` too, so the caller must hold the same shim. :class:`TalkerThread`
    holds it for the life of its thread, which is where the vocoder is meant to run.

    Raises:
        FileNotFoundError: if the assets directory does not exist.
        ImportError: if ``stepaudio2`` is not installed.
    """
    directory = Path(token2wav_dir).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"token2wav assets not found: {directory}")
    from stepaudio2 import Token2wav

    with talker_device_shim(backend):
        return Token2wav(str(directory), float16=bool(float16), n_timesteps=int(n_timesteps))


def build_talker_runtime(
    *,
    thinker_model: Any,
    backend: DeviceBackend,
    base_model: str | Path,
    talker_checkpoint: str | Path,
    token2wav_dir: str | Path,
    prompt_wav_path: str | Path,
    n_timesteps: int = 10,
    float16: bool = False,
    config: Any | None = None,
    warm: bool = True,
) -> Any:
    """Build an upstream ``DetachedTalkerRuntime`` on ``backend``'s device.

    This is upstream's ``DetachedTalkerRuntime.from_thinker_model`` with its CUDA-only
    parts replaced, and nothing else: that classmethod refuses to run without CUDA and
    rejects a non-CUDA device before it gets to the load, so it cannot be called here.
    Everything it does afterwards — building ``MiniCPMTTS`` from the Thinker's
    ``tts_config``, overlaying the base-model tensors with the Talker checkpoint, attaching
    token2wav — is device neutral once :func:`talker_device_shim` is in place, so the
    resulting object is upstream's, not a fork of it.

    Args:
        thinker_model: the loaded Thinker; its config, dtype and defining module are read.
        backend: the device backend the Talker runs on, the Thinker's own.
        base_model: the MiniCPM-o checkpoint holding the frozen TTS tensors.
        talker_checkpoint: the Gander Talker checkpoint overlaid on top of it.
        token2wav_dir: the vocoder assets.
        prompt_wav_path: the reference wav that fixes the voice.
        warm: run one throwaway vocoder chunk so the first real unit is not the one that
            pays for kernel compilation. The RNG state is saved and restored around it
            through the backend.

    Raises:
        RuntimeError: if the Talker checkpoint contributes no tensors.
    """
    import importlib
    from copy import deepcopy

    import torch
    from mcpmft.modeling.load import load_prefixed_submodule_state_dict

    device = backend.device()
    model_module = importlib.import_module(type(thinker_model).__module__)
    tts_class = model_module.MiniCPMTTS
    tts_config = deepcopy(thinker_model.config.tts_config)
    if getattr(thinker_model.config, "_attn_implementation", None) == "flash_attention_2":
        tts_config.attn_implementation = "flash_attention_2"
    else:
        tts_config.attn_implementation = "eager"
    dtype = next(thinker_model.parameters()).dtype

    with talker_device_shim(backend):
        tts = tts_class(config=tts_config, audio_tokenizer=None)
        tts.to(device=device, dtype=dtype)
        base_loaded = load_prefixed_submodule_state_dict(
            tts, str(base_model), prefix="tts.", strict=True
        )
        overlay_loaded = load_prefixed_submodule_state_dict(
            tts, str(talker_checkpoint), prefix="tts.", strict=False
        )
        if overlay_loaded <= 0:
            raise RuntimeError(f"No Talker tensors loaded from {talker_checkpoint}")
        LOGGER.info(
            "loaded Talker on %s with %d base and %d fine-tuned tensors",
            device,
            base_loaded,
            overlay_loaded,
        )
        tts.eval()
        tts.config.audio_tokenizer_type = "s3tokenizer_step_audio"
        tts.audio_tokenizer = build_token2wav(
            token2wav_dir, backend, float16=float16, n_timesteps=n_timesteps
        )

        from mcpmft.infer.detached_talker import DetachedTalkerRuntime

        runtime = DetachedTalkerRuntime(
            tts=tts,
            model_module=model_module,
            device=str(device),
            prompt_wav_path=str(prompt_wav_path),
            config=config,
        )
        if warm:
            with torch.inference_mode():
                runtime.warm_token2wav()
    return runtime


def talker_factory_from_config(
    config: TalkoverConfig,
    *,
    token2wav_dir: str | None = None,
    ref_audio_path: str | None = None,
    warm: bool = True,
) -> Callable[[Any, DeviceBackend], TalkerThread]:
    """Return the factory :func:`talkover.engine.session.build_live_session` calls.

    The factory runs on the engine's inference thread, right after the checkpoints are
    loaded, because the Talker is built from the loaded Thinker. It needs a reference wav:
    upstream's runtime has no fallback voice, so ``engine.ref_audio_path`` (or the
    ``ref_audio_path`` argument) must point at one.

    Raises:
        NotImplementedError: if ``engine.talker_device`` names another device. That is the
            M4 multi-GPU branch, where upstream ``DetachedTalkerRuntime.from_thinker_model``
            places the Talker on a second CUDA card.
        ValueError: if no reference wav or no token2wav directory is configured.
    """
    engine = config.engine
    if engine.talker_device is not None and engine.talker_device != engine.device:
        raise NotImplementedError(
            "engine.talker_device pointing at a second device is the M4 multi-GPU branch: "
            "it needs upstream DetachedTalkerRuntime.from_thinker_model, which is CUDA "
            f"only; got talker_device={engine.talker_device!r} with device={engine.device!r}"
        )
    reference = ref_audio_path or engine.ref_audio_path
    if not reference:
        raise ValueError(
            "the in-process Talker thread needs engine.ref_audio_path: upstream's Talker "
            "runtime has no built-in reference voice"
        )
    from talkover.engine.session import default_token2wav_dir

    assets = token2wav_dir or engine.token2wav_dir or default_token2wav_dir(config)
    if not assets:
        raise ValueError(
            "the in-process Talker thread found no token2wav assets: set "
            "engine.token2wav_dir, or make sure <engine.base_model>/assets/token2wav exists"
        )

    def factory(thinker_model: Any, backend: DeviceBackend) -> TalkerThread:
        runtime = build_talker_runtime(
            thinker_model=thinker_model,
            backend=backend,
            base_model=engine.base_model,
            talker_checkpoint=engine.talker_checkpoint,
            token2wav_dir=assets,
            prompt_wav_path=reference,
            warm=warm,
        )
        return TalkerThread(runtime, backend)

    return factory
