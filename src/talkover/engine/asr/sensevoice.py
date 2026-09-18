"""SenseVoice ONNX backend, a candidate for Chinese telephony scenarios, evaluated in phase two.

The class exists so `asr.backend: sensevoice` fails with one clear message instead of an
``unknown backend`` error, and so the phase-two implementation has a place to land. It
implements the :class:`~talkover.engine.asr.AsrBackend` protocol shape but nothing else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from talkover.engine.asr import AsrResult, BaseAsrBackend

if TYPE_CHECKING:
    from talkover.config import AsrConfig

__all__ = ["SenseVoiceBackend"]

_MESSAGE = (
    "asr.backend 'sensevoice' is not implemented: the SenseVoice ONNX backend is a "
    "phase-two candidate (DESIGN.md 4.3). Use 'mlx_whisper' on MPS or 'faster_whisper' "
    "on CUDA/CPU."
)


class SenseVoiceBackend(BaseAsrBackend):
    """Placeholder that refuses to be constructed."""

    name = "sensevoice"

    def __init__(self, config: AsrConfig) -> None:
        raise NotImplementedError(_MESSAGE)

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        raise NotImplementedError(_MESSAGE)
