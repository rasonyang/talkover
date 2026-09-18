"""Tests for the in-process Talker thread (T1.5).

Three layers, all marked:

- `cpu`: :class:`TalkerThread` driven with a fake runtime — queue order, the
  ``synchronize`` separation, cancellation epochs, failure handling and shutdown, with no
  weights and no device work;
- `cpu`: the device shim (:func:`talker_device_shim`) against ``CpuBackend``, and the
  wiring into :class:`EngineSession`, which turns Talker chunks into ``is_audio_chunk``
  events;
- `mps`: the real ``stepaudio2`` vocoder on the Metal device, driven through the Talker
  thread with a fixed S3 token sequence, which must come back as a 24 kHz waveform while
  the submitting thread stays free.

The `mps` case needs only the Talker checkpoint's assets (``assets/token2wav`` and a
reference wav); it does not load the Thinker or the AR Talker, so it runs before those
checkpoints exist. The AR stage itself (`build_talker_runtime`) stays untested until they
do.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from talkover.config import TalkoverConfig, load_config, parse_config
from talkover.engine import talker as talker_module
from talkover.engine.backend import DeviceBackend, get_backend
from talkover.engine.backend.shims import talker_device_shim
from talkover.engine.protocol import OUTPUT_SAMPLE_RATE
from talkover.engine.session import EngineSession, default_token2wav_dir
from talkover.engine.talker import (
    SILENCE_TOKEN_ID,
    SPEECH_TOKENS_PER_UNIT,
    TalkerChunk,
    TalkerDone,
    TalkerFailed,
    TalkerInterrupted,
    TalkerRuntime,
    TalkerThread,
    build_token2wav,
    talker_factory_from_config,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_EXAMPLE = REPO_ROOT / "configs" / "serve.example.yaml"


# --------------------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------------------


class FakeRuntime:
    """A Talker runtime that emits scripted waveforms without touching a device.

    Args:
        chunks_per_request: how many waveform chunks one request produces.
        samples: samples in each chunk.
        fail_on_unit: raise instead of synthesizing this unit, once.
        gate: when set, every ``synthesize`` waits for it before emitting.
    """

    def __init__(
        self,
        *,
        chunks_per_request: int = 1,
        samples: int = 8,
        fail_on_unit: int | None = None,
        gate: threading.Event | None = None,
    ) -> None:
        self.chunks_per_request = chunks_per_request
        self.samples = samples
        self.fail_on_unit = fail_on_unit
        self.gate = gate
        self.resets = 0
        self.started = threading.Event()
        self.synthesized: list[int] = []
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.resets += 1

    def synthesize(
        self,
        request: Any,
        cancel_event: threading.Event,
        emit: Any,
    ) -> dict[str, Any]:
        self.started.set()
        if self.gate is not None:
            self.gate.wait(10.0)
        if self.fail_on_unit == request.unit_id:
            self.fail_on_unit = None
            raise RuntimeError(f"fake talker failure on unit {request.unit_id}")
        with self._lock:
            self.synthesized.append(request.unit_id)
        for sequence in range(self.chunks_per_request):
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
            emit(
                SimpleNamespace(
                    generation_id=request.generation_id,
                    unit_id=request.unit_id,
                    sequence=sequence + 1,
                    waveform=np.full(self.samples, 0.25, dtype=np.float32),
                    current_time=request.current_time,
                    end_of_turn=request.end_of_turn and sequence + 1 == self.chunks_per_request,
                    metrics={"chunk": sequence + 1},
                )
            )
        return {"audio_chunks": self.chunks_per_request, "unit_id": request.unit_id}


class CountingBackend:
    """Wraps a backend and counts ``synchronize`` calls per thread."""

    def __init__(self, inner: DeviceBackend) -> None:
        self._inner = inner
        self.name = inner.name
        self.synchronize_threads: list[str] = []
        self._lock = threading.Lock()

    def is_available(self) -> bool:
        return self._inner.is_available()

    def device(self, index: int | None = None) -> torch.device:
        return self._inner.device(index)

    def synchronize(self) -> None:
        with self._lock:
            self.synchronize_threads.append(threading.current_thread().name)
        self._inner.synchronize()

    def rng_state(self) -> Any:
        return self._inner.rng_state()

    def set_rng_state(self, state: Any) -> None:
        self._inner.set_rng_state(state)

    def autocast_dtype(self) -> torch.dtype:
        return self._inner.autocast_dtype()


def _thread(runtime: Any, backend: DeviceBackend | None = None) -> TalkerThread:
    return TalkerThread(
        runtime,
        backend if backend is not None else get_backend("cpu"),
        hold_device_shim=False,
    )


def _submit(thread: TalkerThread, unit_id: int, *, end_of_turn: bool = False) -> Any:
    return thread.submit(
        unit_id=unit_id,
        current_time=unit_id * 1000,
        token_ids=(1, 2, 3),
        hidden_states=np.zeros((3, 4), dtype=np.float32),
        end_of_turn=end_of_turn,
    )


def _collect(thread: TalkerThread, count: int, timeout: float = 5.0) -> list[Any]:
    """Take exactly ``count`` outputs, or fail the test."""
    deadline = time.monotonic() + timeout
    outputs: list[Any] = []
    while len(outputs) < count:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError(f"only {len(outputs)} of {count} outputs arrived: {outputs}")
        value = thread.poll(remaining)
        if value is not None:
            outputs.append(value)
    return outputs


# --------------------------------------------------------------------------------------
# The thread and its queue
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_requests_are_served_in_order() -> None:
    runtime = FakeRuntime(chunks_per_request=2)
    thread = _thread(runtime)
    try:
        for unit_id in (1, 2, 3):
            _submit(thread, unit_id, end_of_turn=unit_id == 3)
        outputs = _collect(thread, 9)
    finally:
        thread.close()

    assert runtime.synthesized == [1, 2, 3]
    chunks = [value for value in outputs if isinstance(value, TalkerChunk)]
    done = [value for value in outputs if isinstance(value, TalkerDone)]
    assert [(chunk.unit_id, chunk.sequence) for chunk in chunks] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (3, 1),
        (3, 2),
    ]
    assert [value.unit_id for value in done] == [1, 2, 3]
    assert [value.end_of_turn for value in done] == [False, False, True]
    assert chunks[0].sample_rate == OUTPUT_SAMPLE_RATE
    assert chunks[0].waveform.dtype == np.float32
    assert chunks[0].waveform.shape == (8,)
    assert done[-1].metrics["audio_chunks"] == 2


@pytest.mark.cpu
def test_runtime_is_reset_once_per_generation() -> None:
    runtime = FakeRuntime()
    thread = _thread(runtime)
    try:
        _submit(thread, 1)
        _submit(thread, 2)
        _collect(thread, 4)
        assert runtime.resets == 1
        thread.cancel("barge_in")
        _submit(thread, 3)
        _collect(thread, 3)  # the cancel event plus the new unit's chunk and done
    finally:
        thread.close()
    assert runtime.resets == 2


@pytest.mark.cpu
def test_submit_synchronizes_on_the_calling_thread() -> None:
    backend = CountingBackend(get_backend("cpu"))
    runtime = FakeRuntime()
    thread = _thread(runtime, backend)
    caller = threading.current_thread().name
    try:
        _submit(thread, 1)
        _collect(thread, 2)
    finally:
        thread.close()

    # One hand-over synchronize on the caller (the Thinker), one on the Talker thread
    # after the request is finished. Nothing else blocks the caller.
    assert backend.synchronize_threads.count(caller) == 1
    assert [name for name in backend.synchronize_threads if name != caller] == ["talkover-talker"]


@pytest.mark.cpu
def test_submit_does_not_wait_for_synthesis() -> None:
    gate = threading.Event()
    runtime = FakeRuntime(gate=gate)
    thread = _thread(runtime)
    try:
        started = time.perf_counter()
        _submit(thread, 1)
        _submit(thread, 2)
        submit_sec = time.perf_counter() - started
        assert runtime.started.wait(5.0)
        assert submit_sec < 0.1
        assert thread.state()["active"] is True
        gate.set()
        _collect(thread, 4)
    finally:
        gate.set()
        thread.close()
    assert thread.state()["drained"] is True


@pytest.mark.cpu
def test_cancel_drops_queued_requests_and_opens_a_generation() -> None:
    gate = threading.Event()
    runtime = FakeRuntime(gate=gate)
    thread = _thread(runtime)
    try:
        _submit(thread, 1)
        assert runtime.started.wait(5.0)
        _submit(thread, 2)
        _submit(thread, 3)
        assert thread.generation_id == 1
        event = thread.cancel("barge_in")
        gate.set()
        assert isinstance(event, TalkerInterrupted)
        assert event.cancelled_generation_id == 1
        assert thread.generation_id == 2
        assert thread.wait_until_drained(5.0) is True
        outputs = thread.drain_outputs()
    finally:
        gate.set()
        thread.close()

    assert [value for value in outputs if isinstance(value, TalkerInterrupted)] == [event]
    # Units 2 and 3 never ran, and the cancelled unit 1 published neither chunk nor done.
    assert runtime.synthesized in ([], [1])
    assert [value for value in outputs if isinstance(value, TalkerDone)] == []
    assert [value for value in outputs if isinstance(value, TalkerChunk)] == []


@pytest.mark.cpu
def test_a_failed_request_is_reported_and_the_thread_stays_usable() -> None:
    runtime = FakeRuntime(fail_on_unit=1)
    thread = _thread(runtime)
    try:
        _submit(thread, 1)
        _submit(thread, 2)
        outputs = _collect(thread, 3)
    finally:
        thread.close()

    failed = outputs[0]
    assert isinstance(failed, TalkerFailed)
    assert failed.unit_id == 1
    assert "fake talker failure" in failed.message
    assert isinstance(outputs[1], TalkerChunk)
    assert isinstance(outputs[2], TalkerDone)
    assert runtime.synthesized == [2]
    # Once when the generation opened, once after the failure, once for the retry.
    assert runtime.resets == 3


@pytest.mark.cpu
def test_close_is_idempotent_and_refuses_later_submits() -> None:
    thread = _thread(FakeRuntime())
    _submit(thread, 1)
    assert thread.wait_until_drained(5.0) is True
    thread.close(drain=True)
    thread.close()
    with pytest.raises(RuntimeError, match="closed"):
        _submit(thread, 2)


@pytest.mark.cpu
def test_state_reports_pending_work() -> None:
    gate = threading.Event()
    runtime = FakeRuntime(gate=gate)
    thread = _thread(runtime)
    try:
        assert thread.state() == {
            "active": False,
            "drained": True,
            "pending_requests": 0,
            "generation_id": 1,
            "pending_output_events": 0,
        }
        _submit(thread, 1)
        _submit(thread, 2)
        assert runtime.started.wait(5.0)
        state = thread.state()
        assert state["drained"] is False
        assert state["pending_requests"] == 1
        gate.set()
        _collect(thread, 4)
    finally:
        gate.set()
        thread.close()


@pytest.mark.cpu
def test_fake_runtime_satisfies_the_runtime_protocol() -> None:
    assert isinstance(FakeRuntime(), TalkerRuntime)


# --------------------------------------------------------------------------------------
# The device shim
# --------------------------------------------------------------------------------------


@pytest.mark.cpu
def test_talker_device_shim_retargets_cuda_placement() -> None:
    backend = get_backend("cpu")
    device = backend.device()
    with talker_device_shim(backend):
        assert torch.zeros(4, device="cuda").device == device
        assert torch.tensor([1, 2], device=torch.device("cuda", 0)).device == device
        assert torch.ones(2, 2).cuda().device == device
        assert torch.nn.Linear(2, 2).cuda().weight.device == device
        # A float64 buffer survives: only MPS, which has no float64, narrows it.
        assert torch.from_numpy(np.hamming(8)).cuda().dtype is torch.float64
        assert torch.zeros(2).device == device  # an untouched call still works


@pytest.mark.cpu
def test_talker_device_shim_routes_device_calls_through_the_backend() -> None:
    backend = CountingBackend(get_backend("cpu"))
    with talker_device_shim(backend):
        with torch.cuda.device(torch.device("cuda", 0)):
            torch.cuda.synchronize(torch.device("cuda", 0))
        state = torch.cuda.get_rng_state()
        first = torch.randn(8)
        torch.cuda.set_rng_state(state)
        second = torch.randn(8)
    assert backend.synchronize_threads == [threading.current_thread().name]
    assert torch.equal(first, second)


@pytest.mark.cpu
def test_talker_device_shim_disables_an_unsupported_autocast() -> None:
    backend = get_backend("cpu")
    with talker_device_shim(backend):
        with torch.amp.autocast("cuda", dtype=torch.float32):
            assert torch.is_autocast_enabled("cpu") is False
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            assert torch.is_autocast_enabled("cpu") is True


@pytest.mark.cpu
def test_talker_device_shim_nests_and_restores() -> None:
    backend = get_backend("cpu")
    original_zeros = torch.zeros
    original_cuda = torch.Tensor.cuda
    with talker_device_shim(backend):
        assert torch.zeros is not original_zeros
        with talker_device_shim(backend):
            assert torch.zeros(1, device="cuda").device == backend.device()
        assert torch.zeros is not original_zeros  # the inner exit must not restore
    assert torch.zeros is original_zeros
    assert torch.Tensor.cuda is original_cuda


@pytest.mark.cpu
def test_talker_device_shim_restores_after_an_exception() -> None:
    backend = get_backend("cpu")
    original_zeros = torch.zeros
    with pytest.raises(ValueError, match="boom"), talker_device_shim(backend):
        raise ValueError("boom")
    assert torch.zeros is original_zeros


@pytest.mark.cpu
def test_talker_device_shim_rejects_a_second_backend() -> None:
    cpu = get_backend("cpu")
    mps = get_backend("mps")
    with talker_device_shim(cpu), pytest.raises(RuntimeError, match="cannot also target"):
        talker_device_shim(mps).__enter__()


@pytest.mark.cpu
def test_talker_device_shim_is_a_noop_on_cuda() -> None:
    original_zeros = torch.zeros
    with talker_device_shim(get_backend("cuda:1")):
        assert torch.zeros is original_zeros


# --------------------------------------------------------------------------------------
# The factory and the M4 branch
# --------------------------------------------------------------------------------------


def _config(**engine_overrides: Any) -> TalkoverConfig:
    engine: dict[str, Any] = {"device": "cpu"}
    engine.update(engine_overrides)
    return parse_config({"engine": engine})


@pytest.mark.cpu
def test_talker_factory_refuses_a_second_device() -> None:
    config = _config(device="cuda:0", talker_device="cuda:1")
    with pytest.raises(NotImplementedError, match="M4"):
        talker_factory_from_config(config)


@pytest.mark.cpu
def test_talker_factory_requires_a_reference_wav(tmp_path: Path) -> None:
    config = _config(ref_audio_path=None, token2wav_dir=str(tmp_path))
    with pytest.raises(ValueError, match="ref_audio_path"):
        talker_factory_from_config(config)


@pytest.mark.cpu
def test_talker_factory_requires_token2wav_assets(tmp_path: Path) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    config = _config(base_model="", ref_audio_path=str(reference))
    with pytest.raises(ValueError, match="token2wav"):
        talker_factory_from_config(config)


@pytest.mark.cpu
def test_build_token2wav_passes_the_configured_step_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`engine.token2wav_timesteps` reaches the upstream vocoder constructor (DESIGN.md 9)."""
    seen: dict[str, Any] = {}

    class FakeToken2wav:
        def __init__(self, directory: str, *, float16: bool = False, n_timesteps: int = 10) -> None:
            seen.update(directory=directory, float16=float16, n_timesteps=n_timesteps)

    monkeypatch.setitem(sys.modules, "stepaudio2", SimpleNamespace(Token2wav=FakeToken2wav))
    backend = get_backend("cpu")

    build_token2wav(tmp_path, backend)
    assert seen["n_timesteps"] == 10  # upstream's default when nothing asks otherwise

    build_token2wav(tmp_path, backend, n_timesteps=2)
    assert seen == {"directory": str(tmp_path), "float16": False, "n_timesteps": 2}


@pytest.mark.cpu
def test_the_talker_factory_forwards_token2wav_timesteps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reference = tmp_path / "ref.wav"
    reference.write_bytes(b"RIFF")
    config = _config(
        base_model="",
        token2wav_dir=str(tmp_path),
        ref_audio_path=str(reference),
        token2wav_timesteps=5,
    )
    seen: dict[str, Any] = {}

    def fake_build_talker_runtime(**kwargs: Any) -> Any:
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(talker_module, "build_talker_runtime", fake_build_talker_runtime)
    monkeypatch.setattr(talker_module, "TalkerThread", lambda runtime, backend: runtime)

    factory = talker_factory_from_config(config)
    factory(SimpleNamespace(), get_backend("cpu"))

    assert seen["n_timesteps"] == 5
    assert seen["token2wav_dir"] == str(tmp_path)
    assert seen["prompt_wav_path"] == str(reference)


# --------------------------------------------------------------------------------------
# The EngineSession wiring
# --------------------------------------------------------------------------------------


class StubStepEvent:
    """The upstream step event fields `EngineSession.from_upstream` reads."""

    def __init__(self, index: int, *, is_listen: bool = False) -> None:
        self.index = index
        self.is_listen = is_listen
        self.text = f"unit-{index}"
        self.end_of_turn = False
        self.interrupted = False
        self.audio_waveform = None
        self.current_time = None
        self.unit_id = index
        self.generation_id = 1
        self.metrics: dict[str, Any] = {}


class StubLiveSessionWithTalker:
    """A stub upstream session whose `speech_worker` is a real Talker thread."""

    def __init__(self, talker: TalkerThread) -> None:
        self.speech_worker = talker
        self.step_index = 0
        self.closed = False

    def feed_pcm16(self, data: bytes) -> list[StubStepEvent]:
        self.step_index += 1
        self.speech_worker.submit(
            unit_id=self.step_index,
            current_time=self.step_index * 1000,
            token_ids=(7, 8),
            hidden_states=np.zeros((2, 4), dtype=np.float32),
            end_of_turn=False,
        )
        self.speech_worker.wait_until_drained(5.0)
        return [StubStepEvent(self.step_index)]

    def close(self, **kwargs: Any) -> None:
        self.closed = True
        self.speech_worker.close(**kwargs)


@pytest.mark.cpu
async def test_engine_session_publishes_talker_audio_as_its_own_event() -> None:
    talker = _thread(FakeRuntime(chunks_per_request=2))
    stub = StubLiveSessionWithTalker(talker)
    session = EngineSession(
        _config(),
        backend=get_backend("cpu"),
        session_factory=lambda: stub,
    )
    await session.start()
    try:
        await session.feed_pcm16(b"\x00" * 32000)
        stream = session.events()
        events = [await anext(stream) for _ in range(3)]
        await stream.aclose()
    finally:
        await session.stop()

    unit, first, second = events
    assert unit.is_audio_chunk is False
    assert unit.text == "unit-1"
    assert unit.audio_waveform is None
    for chunk_event in (first, second):
        assert chunk_event.is_audio_chunk is True
        assert chunk_event.text == ""
        assert chunk_event.end_of_turn is False
        assert chunk_event.unit_id == 1
        assert chunk_event.sample_rate == OUTPUT_SAMPLE_RATE
        assert chunk_event.has_audio is True
    assert first.metrics["chunk"] == 1
    assert second.metrics["chunk"] == 2
    assert stub.closed is True


# --------------------------------------------------------------------------------------
# The real vocoder on MPS
# --------------------------------------------------------------------------------------


class Token2wavRuntime:
    """Talker runtime that vocodes a given S3 token sequence, skipping the AR stage.

    The AR Talker needs the base model and the Talker checkpoint; token2wav needs neither,
    so this drives the second half of the Talker on the real device with a fixed token
    sequence. It mirrors what upstream ``DetachedTalkerRuntime._tokens_to_waveforms`` does
    with the vocoder: keep a look-ahead buffer, hand it whole chunks, emit what comes back.
    """

    def __init__(self, token2wav: Any, prompt_wav: str) -> None:
        self._token2wav = token2wav
        self._prompt_wav = prompt_wav
        self._pre_lookahead = int(token2wav.flow.pre_lookahead_len)
        self._buffer: list[int] = []

    def reset(self) -> None:
        self._token2wav.cache = None
        flow_cache, hift_cache = self._token2wav.set_stream_cache(self._prompt_wav)
        self._token2wav.stream_cache = flow_cache
        self._token2wav.hift_cache_dict = hift_cache
        self._buffer = [SILENCE_TOKEN_ID] * self._pre_lookahead

    def synthesize(self, request: Any, cancel_event: threading.Event, emit: Any) -> dict[str, Any]:
        started = time.perf_counter()
        self._buffer.extend(request.token_ids)
        window = SPEECH_TOKENS_PER_UNIT + self._pre_lookahead
        emitted = 0
        while len(self._buffer) >= window:
            if cancel_event.is_set():
                raise RuntimeError("cancelled")
            pcm = self._token2wav.stream(self._buffer[:window], prompt_wav=self._prompt_wav)
            self._buffer = self._buffer[SPEECH_TOKENS_PER_UNIT:]
            emitted += 1
            emit(
                SimpleNamespace(
                    generation_id=request.generation_id,
                    unit_id=request.unit_id,
                    sequence=emitted,
                    waveform=pcm,
                    current_time=request.current_time,
                    end_of_turn=request.end_of_turn,
                    metrics={},
                )
            )
        return {"audio_chunks": emitted, "cost_token2wav": time.perf_counter() - started}


def _talker_assets() -> tuple[str, str]:
    """The token2wav directory and reference wav, or skip with what is missing."""
    if not SERVE_EXAMPLE.is_file():
        pytest.skip(f"{SERVE_EXAMPLE} is missing")
    config = load_config(SERVE_EXAMPLE)
    token2wav_dir = default_token2wav_dir(config)
    reference = config.engine.ref_audio_path or (
        Path(config.engine.talker_checkpoint) / "assets" / "ref_audio.wav"
    )
    missing = []
    if token2wav_dir is None:
        missing.append(f"token2wav assets under {config.engine.base_model}")
    if not Path(reference).is_file():
        missing.append(f"reference wav {reference}")
    if missing:
        pytest.skip(
            "Talker assets are not downloaded: "
            + ", ".join(missing)
            + " (run scripts/download_models.sh)"
        )
    return str(token2wav_dir), str(reference)


@pytest.mark.mps
def test_token2wav_streams_a_24khz_waveform_on_mps() -> None:
    backend = get_backend("mps")
    if not backend.is_available():
        pytest.skip("MPS is not available on this machine")
    token2wav_dir, reference = _talker_assets()
    try:
        from talkover.engine.talker import build_token2wav
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"the Talker stack is not importable: {exc}")
    try:
        token2wav = build_token2wav(token2wav_dir, backend)
    except ImportError as exc:
        pytest.skip(f"stepaudio2 is not importable: {exc}")

    thread = TalkerThread(Token2wavRuntime(token2wav, reference), backend)
    tokens = tuple(range(100, 100 + SPEECH_TOKENS_PER_UNIT))
    try:
        sync_started = time.perf_counter()
        backend.synchronize()
        sync_sec = time.perf_counter() - sync_started

        submit_started = time.perf_counter()
        thread.submit(
            unit_id=1,
            current_time=1000,
            token_ids=tokens,
            hidden_states=None,
            end_of_turn=False,
        )
        submit_sec = time.perf_counter() - submit_started
        first = _collect(thread, 2, timeout=300.0)
        first_sec = time.perf_counter() - submit_started

        warm_started = time.perf_counter()
        thread.submit(
            unit_id=2,
            current_time=2000,
            token_ids=tokens,
            hidden_states=None,
            end_of_turn=True,
        )
        warm_submit_sec = time.perf_counter() - warm_started
        second = _collect(thread, 2, timeout=300.0)
        warm_sec = time.perf_counter() - warm_started
    finally:
        thread.close()

    chunk = first[0]
    assert isinstance(chunk, TalkerChunk)
    assert isinstance(first[1], TalkerDone)
    assert chunk.sample_rate == OUTPUT_SAMPLE_RATE
    assert chunk.waveform.dtype == np.float32
    # 25 S3 tokens are one second of speech at 24 kHz.
    assert chunk.waveform.size == OUTPUT_SAMPLE_RATE
    assert float(np.abs(chunk.waveform).max()) > 1e-3
    assert float(np.abs(chunk.waveform).max()) <= 1.0
    assert isinstance(second[0], TalkerChunk)
    assert second[0].waveform.size == OUTPUT_SAMPLE_RATE

    # Handing a unit over costs the Thinker a synchronize and a queue put, nothing more.
    assert submit_sec < sync_sec + 0.05
    assert warm_submit_sec < sync_sec + 0.05
    assert submit_sec < first_sec / 4
    print(
        f"\ntoken2wav on mps: synchronize {sync_sec * 1000:.2f} ms; "
        f"hand-over {submit_sec * 1000:.2f} ms (warm {warm_submit_sec * 1000:.2f} ms); "
        f"first chunk {first_sec:.2f} s; warm unit {warm_sec:.2f} s"
    )
