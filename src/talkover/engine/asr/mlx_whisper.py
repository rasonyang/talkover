"""mlx-whisper large-v3-turbo backend, the default on MPS.

mlx-whisper runs Whisper through MLX on the Metal device directly, so this module never
touches a torch device at all: the :class:`~talkover.engine.backend.DeviceBackend` only
decides *which* backend is built (see :func:`talkover.engine.asr.create_asr`).

``mlx_whisper`` is imported lazily inside the constructor so the package imports cleanly
on a machine without the ``mps`` extra installed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

__all__ = ["MlxWhisperBackend", "resolve_model_source"]

# Short names from `asr.model` mapped to their mlx-community repository.
_MODEL_REPOS = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}


def _models_dir() -> Path:
    """The local weights directory, matching `scripts/download_models.sh`."""
    return Path(os.environ.get("MODELS_DIR", Path.home() / "models")).expanduser()


def _hf_cache_dir() -> Path:
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    return Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache/huggingface/hub"))


def repo_id_for(model: str) -> str:
    """Map an `asr.model` value to an mlx-community repository id.

    A value that already contains ``/`` is treated as a repository id and returned as is;
    an unknown short name falls back to ``mlx-community/whisper-<name>``.
    """
    if "/" in model:
        return model
    return _MODEL_REPOS.get(model, f"mlx-community/whisper-{model}")


def resolve_model_source(model: str) -> tuple[str, str]:
    """Resolve `asr.model` into ``(path_or_hf_repo, origin)``.

    Local weights win over a download: a directory under ``MODELS_DIR`` (default
    ``~/models``) is used directly, then an already-populated Hugging Face cache entry,
    and only otherwise does mlx-whisper download the repository on first use.

    ``origin`` is one of ``"local"``, ``"hf-cache"`` or ``"hf-download"``, for logging and
    for `talkover check`.
    """
    candidate = Path(model).expanduser()
    if candidate.is_dir():
        return str(candidate), "local"

    repo = repo_id_for(model)
    local_names = [model, repo.split("/", 1)[1]]
    for name in local_names:
        path = _models_dir() / name
        if path.is_dir():
            return str(path), "local"

    cached = _hf_cache_dir() / ("models--" + repo.replace("/", "--"))
    if (cached / "snapshots").is_dir():
        return repo, "hf-cache"
    return repo, "hf-download"


class MlxWhisperBackend(BaseAsrBackend):
    """Whisper through MLX. Blocking; run :meth:`transcribe` off the event loop.

    The model is loaded lazily by ``mlx_whisper`` itself on the first transcription and
    kept in its process-wide cache, so construction stays cheap and `talkover check` can
    build the backend without pulling 1.6 GB of weights.
    """

    name = "mlx_whisper"

    def __init__(self, config: AsrConfig) -> None:
        try:
            import mlx_whisper  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the installed extras
            raise AsrError(
                "asr.backend 'mlx_whisper' needs the 'mps' extra: "
                "uv sync --extra mps (package mlx-whisper)"
            ) from exc

        self.config = config
        self.model_source, self.model_origin = resolve_model_source(config.model)

    def __repr__(self) -> str:
        return f"MlxWhisperBackend(model={self.model_source!r}, origin={self.model_origin!r})"

    def transcribe(
        self,
        pcm: bytes | np.ndarray,
        *,
        start_ms: int = 0,
        language: str | None = None,
    ) -> AsrResult:
        """Transcribe one pcm16 16 kHz segment; timestamps are offset by ``start_ms``."""
        import mlx_whisper

        audio = self._prepare(pcm)
        duration_ms = int(audio.size * 1000 / SAMPLE_RATE)
        if audio.size == 0:
            return AsrResult(text="", duration_ms=0)

        try:
            raw: dict[str, Any] = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_source,
                language=language,
                temperature=0.0,
                condition_on_previous_text=False,
                word_timestamps=False,
                verbose=None,
            )
        except Exception as exc:  # surface every model failure the same way
            raise AsrError(f"mlx-whisper transcription failed: {exc}") from exc

        segments: list[TranscriptSegment] = []
        for segment in raw.get("segments") or []:
            text = str(segment.get("text") or "").strip()
            if not text:
                continue
            segments.append(
                TranscriptSegment(
                    text=text,
                    start_ms=start_ms + int(float(segment.get("start", 0.0)) * 1000),
                    end_ms=start_ms + int(float(segment.get("end", 0.0)) * 1000),
                )
            )
        return self._build_result(
            segments,
            language=raw.get("language"),
            duration_ms=duration_ms,
            text=str(raw.get("text") or ""),
        )
