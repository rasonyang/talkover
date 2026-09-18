"""faster-whisper backend, CUDA float16 or CPU int8.

This is the CUDA default and matches upstream `asr_process` (DESIGN.md 4.3). CTranslate2
has no Metal backend, so on an Apple machine `asr.device` resolves to ``cpu`` with
``compute_type: int8``; that path also serves as the reference implementation.

``faster_whisper`` ships in the ``cuda`` extra and is usually absent on a dev Mac, so the
import is lazy and its absence raises :class:`~talkover.engine.asr.AsrError` at
construction, not at import time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from talkover.engine.asr import (
    SAMPLE_RATE,
    AsrError,
    AsrResult,
    BaseAsrBackend,
    TranscriptSegment,
)

if TYPE_CHECKING:
    from talkover.config import AsrConfig

__all__ = ["FasterWhisperBackend", "resolve_model_source", "split_device"]

# Short names mapped to the Systran CTranslate2 conversions faster-whisper expects.
_MODEL_REPOS = {
    "large-v3-turbo": "Systran/faster-whisper-large-v3-turbo",  # alias: deepdml mirror
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v2": "Systran/faster-whisper-large-v2",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
    "tiny": "Systran/faster-whisper-tiny",
}


def split_device(device: str) -> tuple[str, int]:
    """Split ``cuda:1`` into ``("cuda", 1)``; faster-whisper takes the index separately."""
    kind, _, index = device.partition(":")
    if kind not in ("cuda", "cpu"):
        raise AsrError(
            f"asr.device {device!r} is not usable by faster-whisper: expected 'cpu', "
            "'cuda' or 'cuda:<n>' (CTranslate2 has no Metal backend)"
        )
    return kind, int(index) if index.isdigit() else 0


def resolve_model_source(model: str) -> tuple[str, str]:
    """Resolve `asr.model` into ``(name_or_path, origin)``.

    A local directory under ``MODELS_DIR`` (default ``~/models``) wins; otherwise the
    Hugging Face repository id is returned and faster-whisper resolves it through its own
    cache. ``origin`` is ``"local"`` or ``"hf"``.
    """
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return str(candidate), "local"

    models_dir = Path(os.environ.get("MODELS_DIR", Path.home() / "models")).expanduser()
    local = models_dir / Path(model).name
    if local.is_dir():
        return str(local), "local"

    return _MODEL_REPOS.get(model, model), "hf"


class FasterWhisperBackend(BaseAsrBackend):
    """CTranslate2 Whisper. Blocking; run :meth:`transcribe` off the event loop.

    Not validated on CUDA in this phase (M1 targets MPS); the CUDA run belongs to M4.
    """

    name = "faster_whisper"

    def __init__(self, config: AsrConfig, *, beam_size: int = 3) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AsrError(
                "asr.backend 'faster_whisper' needs the 'cuda' extra: "
                "uv sync --extra cuda (package faster-whisper)"
            ) from exc

        self.config = config
        self.beam_size = beam_size
        self.device, self.device_index = split_device(config.device)
        self.compute_type = config.compute_type
        self.model_source, self.model_origin = resolve_model_source(config.model)

        try:
            self._model = WhisperModel(
                self.model_source,
                device=self.device,
                device_index=self.device_index,
                compute_type=self.compute_type,
                num_workers=1,
            )
        except Exception as exc:  # loading failures are all fatal here
            raise AsrError(
                f"cannot load faster-whisper model {self.model_source!r} on "
                f"{config.device} ({self.compute_type}): {exc}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"FasterWhisperBackend(model={self.model_source!r}, device={self.device!r}, "
            f"index={self.device_index}, compute_type={self.compute_type!r})"
        )

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        """Transcribe one pcm16 16 kHz segment; timestamps are offset by ``start_ms``."""
        audio = self._prepare(pcm)
        duration_ms = int(audio.size * 1000 / SAMPLE_RATE)
        if audio.size == 0:
            return AsrResult(text="", duration_ms=0)

        try:
            segments_iter, info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                best_of=1,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
            )
            raw_segments = list(segments_iter)
        except Exception as exc:  # surface every model failure the same way
            raise AsrError(f"faster-whisper transcription failed: {exc}") from exc

        segments: list[TranscriptSegment] = []
        for segment in raw_segments:
            text = str(segment.text or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=start_ms + int(float(segment.start) * 1000),
                    end_ms=start_ms + int(float(segment.end) * 1000),
                )
            )
        return self._build_result(
            segments,
            language=getattr(info, "language", None),
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        """Drop the CTranslate2 model so its device memory is released."""
        self._model = None
