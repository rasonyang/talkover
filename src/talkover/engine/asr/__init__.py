"""ASR backend abstraction: takes pcm16 16k segments and returns text with timestamps.

ASR is a side channel (DESIGN.md 4.3): it produces the browser transcript and the trusted
text handed to Brain, while the Cerebellum consumes raw audio directly. Because of that the
backend is swappable — `mlx_whisper` on MPS, `faster_whisper` on CUDA or CPU int8.

The module defines three things:

* :class:`AsrBackend`, the protocol every backend implements. ``transcribe`` is a blocking
  call meant to run off the event loop; :meth:`BaseAsrBackend.atranscribe` wraps it in
  :func:`asyncio.to_thread` for callers that live on the loop.
* :class:`EnergyVad`, a dependency-free (numpy only) energy gate with hangover, shared by
  every backend. It emits :class:`SpeechStarted` / :class:`SpeechStopped`. Gander has no
  explicit VAD, so these boundaries are **approximate** (DESIGN.md 5.4); they exist so the
  realtime layer (T2.6) can synthesize `input_audio_buffer.speech_started` /
  `speech_stopped` without pretending to be a real VAD.
* :class:`AsrStream`, which glues the two together: feed it 16 kHz pcm16 chunks and it
  yields :class:`SpeechStarted`, :class:`SpeechStopped` and :class:`Transcript` events.

All audio crossing this API is mono pcm16 little-endian at 16 kHz.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from talkover.config import AsrConfig
    from talkover.engine.backend import DeviceBackend

__all__ = [
    "SAMPLE_RATE",
    "AsrBackend",
    "AsrError",
    "AsrEvent",
    "AsrResult",
    "AsrStream",
    "BaseAsrBackend",
    "EnergyVad",
    "SpeechStarted",
    "SpeechStopped",
    "Transcript",
    "TranscriptSegment",
    "VadSettings",
    "create_asr",
    "pcm16_to_float32",
]

SAMPLE_RATE = 16000
"""The only sample rate this package accepts, in Hz."""

_BYTES_PER_SAMPLE = 2


class AsrError(RuntimeError):
    """Raised when an ASR backend cannot be constructed or a transcription fails."""


# --------------------------------------------------------------------------------------
# Result and event types
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """One timestamped piece of a transcription result.

    Timestamps are milliseconds on the caller's timeline: they already include the
    ``start_ms`` offset passed to :meth:`AsrBackend.transcribe`.
    """

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class AsrResult:
    """What a backend returns for one pcm16 segment: the full text plus its segments."""

    text: str
    segments: tuple[TranscriptSegment, ...] = ()
    language: str | None = None
    duration_ms: int = 0


@dataclass(frozen=True, slots=True)
class SpeechStarted:
    """The VAD saw enough consecutive speech frames; ``at_ms`` is where the run began."""

    at_ms: int


@dataclass(frozen=True, slots=True)
class SpeechStopped:
    """The VAD saw a full hangover of silence; ``at_ms`` is where that silence began."""

    at_ms: int


@dataclass(frozen=True, slots=True)
class Transcript:
    """Text for the utterance that just ended, with its span on the stream timeline."""

    text: str
    start_ms: int
    end_ms: int
    segments: tuple[TranscriptSegment, ...] = ()
    language: str | None = None


AsrEvent = SpeechStarted | SpeechStopped | Transcript


# --------------------------------------------------------------------------------------
# Audio helpers
# --------------------------------------------------------------------------------------


def pcm16_to_float32(pcm: bytes | bytearray | memoryview | np.ndarray) -> np.ndarray:
    """Convert mono pcm16 little-endian bytes to a float32 array in ``[-1, 1)``.

    A float array is passed through unchanged (already normalised); an int16 array is
    scaled. This is the one place the pcm16 convention is encoded.
    """
    if isinstance(pcm, np.ndarray):
        if pcm.dtype == np.float32:
            return pcm
        if pcm.dtype == np.float64:
            return pcm.astype(np.float32)
        if pcm.dtype == np.int16:
            return pcm.astype(np.float32) / 32768.0
        raise AsrError(f"unsupported audio array dtype: {pcm.dtype}")
    data = bytes(pcm)
    if len(data) % _BYTES_PER_SAMPLE:
        raise AsrError("pcm16 input must contain a whole number of 16-bit samples")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _samples_to_ms(samples: int) -> int:
    return int(samples * 1000 / SAMPLE_RATE)


def _ms_to_samples(ms: float) -> int:
    return int(ms * SAMPLE_RATE / 1000)


# --------------------------------------------------------------------------------------
# Backend protocol
# --------------------------------------------------------------------------------------


@runtime_checkable
class AsrBackend(Protocol):
    """Interface of every ASR backend, matching upstream ``asr_process``.

    ``transcribe`` blocks: it runs a model. Callers on the event loop use
    :meth:`BaseAsrBackend.atranscribe` (or :func:`asyncio.to_thread`) instead.
    """

    name: str

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        """Transcribe one pcm16 16 kHz segment, offsetting timestamps by ``start_ms``."""
        ...

    def close(self) -> None:
        """Release model resources. Calling it twice is not an error."""
        ...


class BaseAsrBackend:
    """Shared behaviour for backends: the async wrapper, ``close`` and input validation."""

    name = "base"

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        raise NotImplementedError

    async def atranscribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        """Run :meth:`transcribe` on a worker thread so the event loop keeps running."""
        return await asyncio.to_thread(self.transcribe, pcm, start_ms=start_ms, language=language)

    def close(self) -> None:
        """Default: nothing to release."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    @staticmethod
    def _prepare(pcm: bytes | np.ndarray) -> np.ndarray:
        return pcm16_to_float32(pcm)

    @staticmethod
    def _build_result(
        segments: list[TranscriptSegment],
        *,
        language: str | None,
        duration_ms: int,
        text: str | None = None,
    ) -> AsrResult:
        joined = text if text is not None else " ".join(s.text for s in segments)
        return AsrResult(
            text=joined.strip(),
            segments=tuple(segments),
            language=language,
            duration_ms=duration_ms,
        )


# --------------------------------------------------------------------------------------
# Energy VAD
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VadSettings:
    """Tuning for :class:`EnergyVad`. Defaults suit 16 kHz telephony-style speech.

    ``rms_threshold`` is on the float32 ``[-1, 1)`` scale. ``min_speech_ms`` suppresses
    clicks; ``hangover_ms`` is how much silence ends an utterance.
    """

    frame_ms: int = 20
    rms_threshold: float = 0.01
    min_speech_ms: int = 160
    hangover_ms: int = 480


class EnergyVad:
    """Approximate speech boundaries from frame energy, with a hangover.

    This is deliberately crude: numpy only, no model, no adaptive noise tracking beyond a
    fixed RMS gate. Gander has no explicit VAD, so the realtime layer documents the events
    derived from this class as approximate (DESIGN.md 5.4).

    Feed it 16 kHz pcm16 chunks of any length; leftover samples are carried between calls
    so frame boundaries stay aligned to the stream.
    """

    def __init__(self, settings: VadSettings | None = None) -> None:
        self.settings = settings or VadSettings()
        self._frame_samples = max(1, _ms_to_samples(self.settings.frame_ms))
        self._min_speech_frames = max(
            1, round(self.settings.min_speech_ms / self.settings.frame_ms)
        )
        self._hangover_frames = max(1, round(self.settings.hangover_ms / self.settings.frame_ms))
        self._tail = np.zeros(0, dtype=np.float32)
        self._position = 0  # samples of whole frames consumed so far
        self._in_speech = False
        self._run_start = 0  # sample index where the current voiced/silent run began
        self._run_frames = 0

    @property
    def in_speech(self) -> bool:
        """Whether the VAD currently believes someone is talking."""
        return self._in_speech

    @property
    def position_ms(self) -> int:
        """Milliseconds of audio consumed as whole frames."""
        return _samples_to_ms(self._position)

    def reset(self) -> None:
        """Forget all state, including the stream position."""
        self._tail = np.zeros(0, dtype=np.float32)
        self._position = 0
        self._in_speech = False
        self._run_start = 0
        self._run_frames = 0

    def feed(self, pcm: bytes | np.ndarray) -> list[SpeechStarted | SpeechStopped]:
        """Consume a chunk and return the boundary events it produced (often none)."""
        audio = np.concatenate([self._tail, pcm16_to_float32(pcm)])
        usable = (audio.size // self._frame_samples) * self._frame_samples
        self._tail = audio[usable:].copy()
        if usable == 0:
            return []

        frames = audio[:usable].reshape(-1, self._frame_samples)
        rms = np.sqrt(np.mean(np.square(frames.astype(np.float64)), axis=1))
        voiced = rms >= self.settings.rms_threshold

        events: list[SpeechStarted | SpeechStopped] = []
        for is_voiced in voiced.tolist():
            frame_start = self._position
            self._position += self._frame_samples
            if is_voiced == self._in_speech:
                # A frame that agrees with the current state ends any contrary run.
                self._run_frames = 0
                continue
            if self._run_frames == 0:
                self._run_start = frame_start
            self._run_frames += 1
            if not self._in_speech and self._run_frames >= self._min_speech_frames:
                self._in_speech = True
                self._run_frames = 0
                events.append(SpeechStarted(at_ms=_samples_to_ms(self._run_start)))
            elif self._in_speech and self._run_frames >= self._hangover_frames:
                self._in_speech = False
                self._run_frames = 0
                events.append(SpeechStopped(at_ms=_samples_to_ms(self._run_start)))
        return events

    def flush(self) -> list[SpeechStarted | SpeechStopped]:
        """Close an open utterance at the current position, e.g. at end of input."""
        if not self._in_speech:
            return []
        self._in_speech = False
        at_ms = _samples_to_ms(self._run_start if self._run_frames else self._position)
        self._run_frames = 0
        return [SpeechStopped(at_ms=at_ms)]


# --------------------------------------------------------------------------------------
# Streaming glue
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Utterance:
    start_ms: int
    frames: list[np.ndarray] = field(default_factory=list)


class AsrStream:
    """Feed pcm16 chunks in, get :class:`AsrEvent` values out.

    The stream buffers audio from :class:`SpeechStarted` (minus ``preroll_ms``) until
    :class:`SpeechStopped`, then transcribes that span and emits a :class:`Transcript`.
    Use :meth:`feed` on a worker thread, or :meth:`afeed` from the event loop — the latter
    runs the blocking model call through :func:`asyncio.to_thread`.
    """

    def __init__(
        self,
        backend: AsrBackend,
        *,
        vad: EnergyVad | None = None,
        language: str | None = None,
        preroll_ms: int = 200,
        max_utterance_ms: int = 30_000,
    ) -> None:
        self.backend = backend
        self.vad = vad or EnergyVad()
        self.language = language
        self._preroll_samples = _ms_to_samples(preroll_ms)
        self._max_samples = _ms_to_samples(max_utterance_ms)
        self._history = np.zeros(0, dtype=np.float32)
        self._history_start = 0
        self._utterance: _Utterance | None = None

    def feed(self, pcm: bytes | np.ndarray) -> list[AsrEvent]:
        """Consume a chunk; blocking, because a finished utterance is transcribed here."""
        events: list[AsrEvent] = []
        for event, audio in self._advance(pcm):
            events.append(event)
            if audio is not None:
                transcript = self._transcribe(event, audio)
                if transcript is not None:
                    events.append(transcript)
        return events

    async def afeed(self, pcm: bytes | np.ndarray) -> list[AsrEvent]:
        """Same as :meth:`feed`, with the model call moved off the event loop."""
        events: list[AsrEvent] = []
        for event, audio in self._advance(pcm):
            events.append(event)
            if audio is not None:
                transcript = await asyncio.to_thread(self._transcribe, event, audio)
                if transcript is not None:
                    events.append(transcript)
        return events

    def flush(self) -> list[AsrEvent]:
        """Close an open utterance and transcribe it. Blocking, like :meth:`feed`."""
        events: list[AsrEvent] = []
        for event in self.vad.flush():
            events.append(event)
            audio = self._close_utterance(event.at_ms)
            if audio is not None:
                transcript = self._transcribe(event, audio)
                if transcript is not None:
                    events.append(transcript)
        return events

    def reset(self) -> None:
        """Drop all buffered audio and VAD state."""
        self.vad.reset()
        self._history = np.zeros(0, dtype=np.float32)
        self._history_start = 0
        self._utterance = None

    # -- internals ---------------------------------------------------------------

    def _advance(
        self, pcm: bytes | np.ndarray
    ) -> list[tuple[SpeechStarted | SpeechStopped, np.ndarray | None]]:
        audio = pcm16_to_float32(pcm)
        self._history = np.concatenate([self._history, audio])
        events = self.vad.feed(audio)

        out: list[tuple[SpeechStarted | SpeechStopped, np.ndarray | None]] = []
        for event in events:
            if isinstance(event, SpeechStarted):
                start = max(0, _ms_to_samples(event.at_ms) - self._preroll_samples)
                self._utterance = _Utterance(start_ms=_samples_to_ms(start))
                out.append((event, None))
            else:
                out.append((event, self._close_utterance(event.at_ms)))
        self._trim_history()
        return out

    def _close_utterance(self, stop_ms: int) -> np.ndarray | None:
        utterance = self._utterance
        self._utterance = None
        if utterance is None:
            return None
        begin = _ms_to_samples(utterance.start_ms) - self._history_start
        end = _ms_to_samples(stop_ms) - self._history_start
        begin = max(0, min(begin, self._history.size))
        end = max(begin, min(end, self._history.size))
        span = self._history[begin:end]
        if span.size == 0:
            return None
        return span[-self._max_samples :].copy()

    def _trim_history(self) -> None:
        keep_from = _ms_to_samples(self.vad.position_ms) - self._preroll_samples
        if self._utterance is not None:
            keep_from = min(keep_from, _ms_to_samples(self._utterance.start_ms))
        drop = max(0, keep_from - self._history_start)
        if drop > 0:
            self._history = self._history[drop:].copy()
            self._history_start += drop

    def _transcribe(
        self, event: SpeechStarted | SpeechStopped, audio: np.ndarray
    ) -> Transcript | None:
        start_ms = max(0, event.at_ms - _samples_to_ms(audio.size))
        result = self.backend.transcribe(audio, start_ms=start_ms, language=self.language)
        if not result.text:
            return None
        return Transcript(
            text=result.text,
            start_ms=start_ms,
            end_ms=event.at_ms,
            segments=result.segments,
            language=result.language,
        )


# --------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------


def create_asr(config: AsrConfig, backend: DeviceBackend) -> AsrBackend:
    """Build the ASR backend named by ``config.backend`` for the engine's device.

    ``config.backend`` is normally already concrete: :func:`talkover.config.load_config`
    resolves ``auto`` at load time. A literal ``auto`` is still handled here for configs
    built in code, by reusing :func:`talkover.config.resolve_asr_auto` against the engine
    device ``backend`` stands for — the rule is never duplicated.

    Raises:
        AsrError: if the backend name is unknown or its dependency is missing.
    """
    from talkover.config import EngineConfig, resolve_asr_auto

    if config.backend == "auto" or config.device == "auto" or config.compute_type == "auto":
        device = getattr(backend, "name", "cpu")
        index = getattr(backend, "index", None)
        if device == "cuda" and index:
            device = f"cuda:{index}"
        config = resolve_asr_auto(config, EngineConfig(device=device))

    if config.backend == "mlx_whisper":
        from talkover.engine.asr.mlx_whisper import MlxWhisperBackend

        return MlxWhisperBackend(config)
    if config.backend == "faster_whisper":
        from talkover.engine.asr.faster_whisper import FasterWhisperBackend

        return FasterWhisperBackend(config)
    if config.backend == "sensevoice":
        from talkover.engine.asr.sensevoice import SenseVoiceBackend

        return SenseVoiceBackend(config)
    raise AsrError(
        f"unknown asr.backend {config.backend!r}: expected 'mlx_whisper', "
        "'faster_whisper' or 'sensevoice'"
    )
