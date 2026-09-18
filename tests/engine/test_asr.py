"""Tests for the ASR package: backend dispatch, the energy VAD, and a real transcription.

Markers: `cpu` cases load no model at all, `mps` runs mlx-whisper on the bundled clip, and
`cuda` runs faster-whisper and skips when the device or the package is missing.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from talkover.config import AsrConfig, EngineConfig, resolve_asr_auto
from talkover.engine.asr import (
    SAMPLE_RATE,
    AsrError,
    AsrResult,
    AsrStream,
    BaseAsrBackend,
    EnergyVad,
    SpeechStarted,
    SpeechStopped,
    Transcript,
    TranscriptSegment,
    VadSettings,
    create_asr,
    pcm16_to_float32,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "zh_order_16k.wav"

# Runtime data: the sentence spoken in the fixture is
# "您好，我的订单号是八六四二，请帮我查一下物流。" (see tests/fixtures/README.md).
EXPECTED_SUBSTRINGS = ("订单", "物流")
EXPECTED_DIGITS = ("八六四二", "8642", "86 42", "八六四二号")


# --- helpers -------------------------------------------------------------------


class FakeDeviceBackend:
    """Stand-in for `DeviceBackend`: `create_asr` only reads `name` and `index`."""

    def __init__(self, name: str, index: int = 0) -> None:
        self.name = name
        self.index = index


class FakeAsrBackend(BaseAsrBackend):
    """Records every segment it is asked to transcribe and returns a fixed text."""

    name = "fake"

    def __init__(self, text: str = "hello") -> None:
        self.text = text
        self.calls: list[tuple[int, int]] = []

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        audio = self._prepare(pcm)
        duration_ms = int(audio.size * 1000 / SAMPLE_RATE)
        self.calls.append((start_ms, duration_ms))
        return AsrResult(
            text=self.text,
            segments=(TranscriptSegment(self.text, start_ms, start_ms + duration_ms),),
            language="zh",
            duration_ms=duration_ms,
        )


def _pcm16(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def _tone(ms: int, freq: float = 440.0, amplitude: float = 0.3) -> np.ndarray:
    n = int(SAMPLE_RATE * ms / 1000)
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _read_fixture() -> bytes:
    with wave.open(str(FIXTURE), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == SAMPLE_RATE
        assert handle.getsampwidth() == 2
        return handle.readframes(handle.getnframes())


# --- create_asr dispatch -------------------------------------------------------


@pytest.mark.cpu
def test_create_asr_dispatches_mlx_whisper_on_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    import talkover.engine.asr.mlx_whisper as module

    seen: dict[str, AsrConfig] = {}
    monkeypatch.setattr(module, "MlxWhisperBackend", lambda cfg: seen.setdefault("cfg", cfg))

    built = create_asr(AsrConfig(), FakeDeviceBackend("mps"))

    assert built is seen["cfg"]
    assert seen["cfg"].backend == "mlx_whisper"
    assert seen["cfg"].device == "mps"
    assert seen["cfg"].compute_type == "float16"


@pytest.mark.cpu
def test_create_asr_maps_cuda_to_faster_whisper(monkeypatch: pytest.MonkeyPatch) -> None:
    import talkover.engine.asr.faster_whisper as module

    seen: dict[str, AsrConfig] = {}
    monkeypatch.setattr(module, "FasterWhisperBackend", lambda cfg: seen.setdefault("cfg", cfg))

    create_asr(AsrConfig(), FakeDeviceBackend("cuda", index=1))

    assert seen["cfg"].backend == "faster_whisper"
    assert seen["cfg"].device == "cuda:1"
    assert seen["cfg"].compute_type == "float16"


@pytest.mark.cpu
def test_create_asr_cpu_backend_uses_int8(monkeypatch: pytest.MonkeyPatch) -> None:
    import talkover.engine.asr.faster_whisper as module

    seen: dict[str, AsrConfig] = {}
    monkeypatch.setattr(module, "FasterWhisperBackend", lambda cfg: seen.setdefault("cfg", cfg))

    create_asr(AsrConfig(), FakeDeviceBackend("cpu"))

    assert seen["cfg"].backend == "faster_whisper"
    assert seen["cfg"].device == "cpu"
    assert seen["cfg"].compute_type == "int8"


@pytest.mark.cpu
def test_create_asr_matches_config_resolution() -> None:
    """`create_asr` must not invent a second rule: it reuses `resolve_asr_auto`."""
    for device in ("mps", "cuda", "cpu"):
        resolved = resolve_asr_auto(AsrConfig(), EngineConfig(device=device))
        assert resolved.backend == ("mlx_whisper" if device == "mps" else "faster_whisper")


@pytest.mark.cpu
def test_create_asr_honours_an_already_resolved_config(monkeypatch: pytest.MonkeyPatch) -> None:
    import talkover.engine.asr.faster_whisper as module

    seen: dict[str, AsrConfig] = {}
    monkeypatch.setattr(module, "FasterWhisperBackend", lambda cfg: seen.setdefault("cfg", cfg))
    config = AsrConfig(backend="faster_whisper", device="cpu", compute_type="int8")

    create_asr(config, FakeDeviceBackend("mps"))

    assert seen["cfg"] is config


@pytest.mark.cpu
def test_create_asr_rejects_an_unknown_backend() -> None:
    config = AsrConfig(backend="whisper.cpp", device="cpu", compute_type="int8")
    with pytest.raises(AsrError, match="unknown asr.backend"):
        create_asr(config, FakeDeviceBackend("cpu"))


@pytest.mark.cpu
def test_sensevoice_is_not_implemented() -> None:
    config = AsrConfig(backend="sensevoice", device="cpu", compute_type="int8")
    with pytest.raises(NotImplementedError, match="phase-two"):
        create_asr(config, FakeDeviceBackend("cpu"))


@pytest.mark.cpu
def test_faster_whisper_imports_without_the_package() -> None:
    """The module must import cleanly; only construction may fail."""
    from talkover.engine.asr.faster_whisper import FasterWhisperBackend, split_device

    assert split_device("cuda:2") == ("cuda", 2)
    assert split_device("cuda") == ("cuda", 0)
    assert split_device("cpu") == ("cpu", 0)
    with pytest.raises(AsrError, match="Metal"):
        split_device("mps")

    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        with pytest.raises(AsrError, match="cuda' extra"):
            FasterWhisperBackend(AsrConfig(device="cpu", compute_type="int8"))


# --- pcm helpers ---------------------------------------------------------------


@pytest.mark.cpu
def test_pcm16_to_float32_round_trip() -> None:
    samples = np.array([0, 16384, -16384], dtype=np.int16)
    converted = pcm16_to_float32(samples.tobytes())
    assert converted.dtype == np.float32
    assert np.allclose(converted, [0.0, 0.5, -0.5])
    assert pcm16_to_float32(converted) is converted


@pytest.mark.cpu
def test_pcm16_to_float32_rejects_a_half_sample() -> None:
    with pytest.raises(AsrError, match="whole number"):
        pcm16_to_float32(b"\x01\x02\x03")


# --- energy VAD ----------------------------------------------------------------


@pytest.mark.cpu
def test_vad_reports_start_and_stop_around_a_tone() -> None:
    vad = EnergyVad()
    audio = np.concatenate([_silence(500), _tone(1000), _silence(1000)])

    events = vad.feed(_pcm16(audio))

    assert [type(event) for event in events] == [SpeechStarted, SpeechStopped]
    started, stopped = events
    assert abs(started.at_ms - 500) <= 60
    assert abs(stopped.at_ms - 1500) <= 60
    assert vad.in_speech is False


@pytest.mark.cpu
def test_vad_is_silent_on_silence_only() -> None:
    vad = EnergyVad()
    assert vad.feed(_pcm16(_silence(3000))) == []
    assert vad.flush() == []
    assert vad.in_speech is False


@pytest.mark.cpu
def test_vad_spans_chunk_boundaries_and_flushes() -> None:
    vad = EnergyVad()
    audio = np.concatenate([_silence(300), _tone(800)])
    pcm = _pcm16(audio)
    chunk = 997 * 2  # an odd chunk size, so frames straddle chunk boundaries

    events: list[SpeechStarted | SpeechStopped] = []
    for offset in range(0, len(pcm), chunk):
        events.extend(vad.feed(pcm[offset : offset + chunk]))

    assert [type(event) for event in events] == [SpeechStarted]
    assert vad.in_speech is True

    flushed = vad.flush()
    assert [type(event) for event in flushed] == [SpeechStopped]
    assert flushed[0].at_ms >= 1000


@pytest.mark.cpu
def test_vad_ignores_a_click_shorter_than_min_speech() -> None:
    vad = EnergyVad(VadSettings(min_speech_ms=200))
    audio = np.concatenate([_silence(200), _tone(60), _silence(600)])
    assert vad.feed(_pcm16(audio)) == []


# --- streaming glue ------------------------------------------------------------


@pytest.mark.cpu
def test_stream_emits_a_transcript_after_speech_stops() -> None:
    backend = FakeAsrBackend("您好")
    stream = AsrStream(backend, language="zh")
    audio = np.concatenate([_silence(400), _tone(900), _silence(900)])

    events = stream.feed(_pcm16(audio))

    assert [type(event) for event in events] == [SpeechStarted, SpeechStopped, Transcript]
    transcript = events[-1]
    assert transcript.text == "您好"
    assert transcript.end_ms > transcript.start_ms
    assert len(backend.calls) == 1
    # The preroll means slightly more audio is transcribed than the voiced span.
    start_ms, duration_ms = backend.calls[0]
    assert start_ms <= 400
    assert 900 <= duration_ms <= 1400


@pytest.mark.cpu
async def test_stream_afeed_runs_the_model_off_the_loop() -> None:
    backend = FakeAsrBackend("好的")
    stream = AsrStream(backend)
    audio = np.concatenate([_silence(400), _tone(900), _silence(900)])

    events = await stream.afeed(_pcm16(audio))

    assert [type(event) for event in events] == [SpeechStarted, SpeechStopped, Transcript]


@pytest.mark.cpu
async def test_base_backend_atranscribe() -> None:
    backend = FakeAsrBackend("text")
    result = await backend.atranscribe(_pcm16(_tone(100)), start_ms=250)
    assert result.text == "text"
    assert result.segments[0].start_ms == 250


@pytest.mark.cpu
def test_stream_drops_empty_transcriptions() -> None:
    backend = FakeAsrBackend("")
    stream = AsrStream(backend)
    events = stream.feed(_pcm16(np.concatenate([_silence(400), _tone(900), _silence(900)])))
    assert [type(event) for event in events] == [SpeechStarted, SpeechStopped]


# --- real backends -------------------------------------------------------------


@pytest.mark.mps
def test_mlx_whisper_transcribes_the_chinese_clip() -> None:
    pytest.importorskip("mlx_whisper", reason="the 'mps' extra is not installed")
    from talkover.engine.asr.mlx_whisper import MlxWhisperBackend, resolve_model_source

    config = AsrConfig(backend="mlx_whisper", device="mps", compute_type="float16")
    source, origin = resolve_model_source(config.model)
    backend = MlxWhisperBackend(config)

    pcm = _read_fixture()
    try:
        # Warm-up. The first call may download the weights and always loads them from the
        # HF cache; neither belongs in the steady-state timing asserted below.
        backend.transcribe(pcm, start_ms=0, language="zh")
    except AsrError as exc:  # no network and no cached weights
        if origin == "hf-download":
            pytest.skip(
                f"mlx-whisper weights {source!r} are not cached and could not be downloaded: {exc}"
            )
        raise

    started = time.perf_counter()
    result = backend.transcribe(pcm, start_ms=1000, language="zh")
    elapsed = time.perf_counter() - started

    assert result.text
    assert any(part in result.text for part in EXPECTED_SUBSTRINGS), result.text
    assert any(part in result.text for part in EXPECTED_DIGITS), result.text
    assert result.segments
    first = result.segments[0]
    assert first.start_ms >= 1000  # the start_ms offset is applied
    assert first.end_ms > first.start_ms
    assert 4500 <= result.duration_ms <= 6000
    assert elapsed < 10  # warm, model already loaded; the clip is 5 s and MPS runs it at RTF ~0.1


@pytest.mark.cuda
def test_faster_whisper_transcribes_the_chinese_clip() -> None:
    pytest.importorskip("faster_whisper", reason="the 'cuda' extra is not installed")
    import torch

    if not torch.cuda.is_available():  # pragma: no cover - CUDA machines only
        pytest.skip("no CUDA device available")

    from talkover.engine.asr.faster_whisper import FasterWhisperBackend

    config = AsrConfig(backend="faster_whisper", device="cuda", compute_type="float16")
    backend = FasterWhisperBackend(config)
    try:
        result = backend.transcribe(_read_fixture(), start_ms=0, language="zh")
    finally:
        backend.close()

    assert result.text
    assert any(part in result.text for part in EXPECTED_SUBSTRINGS), result.text
    assert result.segments
