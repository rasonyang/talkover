"""Audio helpers for the Realtime layer: base64 pcm16, resampling, g711, framing.

External audio is pcm16 at 24 kHz (DESIGN.md 5.2); `g711_ulaw` / `g711_alaw` are
telephony formats and are always 8 kHz. The model consumes 16 kHz mono and emits
24 kHz mono, so the input path resamples 24 kHz (or 8 kHz for g711) to 16 kHz and
the output path resamples 24 kHz to 8 kHz only when a g711 format is negotiated.

Python 3.12 removed `audioop`, so G.711 is implemented here with numpy lookup
tables derived from the ITU-T G.711 segment definitions.
"""

from __future__ import annotations

import base64
from math import gcd

import numpy as np
from scipy.signal import firwin, upfirdn

__all__ = [
    "CLIENT_SAMPLE_RATE",
    "G711_SAMPLE_RATE",
    "MODEL_SAMPLE_RATE",
    "UNIT_SAMPLES",
    "AudioFramer",
    "InputAudioStream",
    "StreamResampler",
    "alaw_decode",
    "alaw_encode",
    "decode_base64_pcm16",
    "encode_base64_pcm16",
    "encode_output_audio",
    "float_to_pcm16",
    "input_sample_rate",
    "pcm16_to_float",
    "ulaw_decode",
    "ulaw_encode",
]

#: Sample rate of `pcm16` audio exchanged with the client.
CLIENT_SAMPLE_RATE = 24000
#: Sample rate the duplex model is fed with.
MODEL_SAMPLE_RATE = 16000
#: Sample rate of `g711_ulaw` / `g711_alaw` audio.
G711_SAMPLE_RATE = 8000
#: Samples in one 1 s causal unit at the model rate.
UNIT_SAMPLES = MODEL_SAMPLE_RATE


def input_sample_rate(audio_format: str) -> int:
    """Return the sample rate implied by a Realtime input/output audio format."""
    if audio_format in ("pcm16", "audio/pcm", "audio/pcm16"):
        return CLIENT_SAMPLE_RATE
    if audio_format in ("g711_ulaw", "g711_alaw", "audio/pcmu", "audio/pcma"):
        return G711_SAMPLE_RATE
    raise ValueError(f"unsupported audio format: {audio_format!r}")


# --------------------------------------------------------------------------- #
# base64 pcm16
# --------------------------------------------------------------------------- #


def decode_base64_pcm16(payload: str) -> np.ndarray:
    """Decode a base64 string of little-endian pcm16 into an int16 array."""
    raw = base64.b64decode(payload, validate=False)
    if len(raw) % 2:
        raise ValueError("pcm16 payload has an odd number of bytes")
    return np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=True)


def encode_base64_pcm16(samples: np.ndarray) -> str:
    """Encode an int16 array as a base64 string of little-endian pcm16."""
    data = np.asarray(samples, dtype=np.int16).astype("<i2", copy=False)
    return base64.b64encode(data.tobytes()).decode("ascii")


def pcm16_to_float(samples: np.ndarray) -> np.ndarray:
    """Convert int16 samples to float32 in [-1, 1)."""
    return np.asarray(samples, dtype=np.int16).astype(np.float32) / 32768.0


def float_to_pcm16(samples: np.ndarray) -> np.ndarray:
    """Convert float samples in [-1, 1] to int16 with clipping."""
    scaled = np.asarray(samples, dtype=np.float32) * 32768.0
    return np.clip(np.rint(scaled), -32768.0, 32767.0).astype(np.int16)


# --------------------------------------------------------------------------- #
# G.711 (ITU-T), table driven
# --------------------------------------------------------------------------- #

_ULAW_BIAS = 0x84
_ULAW_CLIP = 8159
_ULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def _segment(values: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """Index of the first segment boundary that is >= value, or 8 when past the last."""
    return (values[:, None] > seg_end[None, :]).sum(axis=1).astype(np.int32)


def _build_ulaw_encode_table() -> np.ndarray:
    pcm = np.arange(-32768, 32768, dtype=np.int32)
    val = pcm >> 2  # G.711 mu-law operates on 14-bit magnitudes
    mask = np.where(val < 0, 0x7F, 0xFF).astype(np.int32)
    val = np.abs(np.where(val < 0, -val, val))
    val = np.minimum(val, _ULAW_CLIP) + (_ULAW_BIAS >> 2)
    seg = _segment(val, _ULAW_SEG_END)
    seg_c = np.minimum(seg, 7)
    uval = (seg_c << 4) | ((val >> (seg_c + 1)) & 0xF)
    uval = np.where(seg >= 8, 0x7F, uval)
    return ((uval ^ mask) & 0xFF).astype(np.uint8)


def _build_ulaw_decode_table() -> np.ndarray:
    code = (~np.arange(256, dtype=np.int32)) & 0xFF
    t = ((code & 0xF) << 3) + _ULAW_BIAS
    t = t << ((code & 0x70) >> 4)
    out = np.where(code & 0x80, _ULAW_BIAS - t, t - _ULAW_BIAS)
    return out.astype(np.int16)


def _build_alaw_encode_table() -> np.ndarray:
    pcm = np.arange(-32768, 32768, dtype=np.int32)
    val = pcm >> 3  # G.711 A-law operates on 13-bit magnitudes
    mask = np.where(val >= 0, 0xD5, 0x55).astype(np.int32)
    val = np.where(val >= 0, val, -val - 1)
    seg = _segment(val, _ALAW_SEG_END)
    seg_c = np.minimum(seg, 7)
    shift = np.where(seg_c < 2, 1, seg_c)
    aval = (seg_c << 4) | ((val >> shift) & 0xF)
    aval = np.where(seg >= 8, 0x7F, aval)
    return ((aval ^ mask) & 0xFF).astype(np.uint8)


def _build_alaw_decode_table() -> np.ndarray:
    code = np.arange(256, dtype=np.int32) ^ 0x55
    t = (code & 0xF) << 4
    seg = (code & 0x70) >> 4
    t = np.where(
        seg == 0, t + 8, np.where(seg == 1, t + 0x108, (t + 0x108) << np.maximum(seg - 1, 0))
    )
    out = np.where(code & 0x80, t, -t)
    return out.astype(np.int16)


_ULAW_ENCODE = _build_ulaw_encode_table()
_ULAW_DECODE = _build_ulaw_decode_table()
_ALAW_ENCODE = _build_alaw_encode_table()
_ALAW_DECODE = _build_alaw_decode_table()


def ulaw_encode(samples: np.ndarray) -> bytes:
    """Encode int16 pcm samples as G.711 mu-law bytes."""
    idx = np.asarray(samples, dtype=np.int16).astype(np.int32) + 32768
    return _ULAW_ENCODE[idx].tobytes()


def ulaw_decode(payload: bytes) -> np.ndarray:
    """Decode G.711 mu-law bytes into int16 pcm samples."""
    return _ULAW_DECODE[np.frombuffer(payload, dtype=np.uint8)].copy()


def alaw_encode(samples: np.ndarray) -> bytes:
    """Encode int16 pcm samples as G.711 A-law bytes."""
    idx = np.asarray(samples, dtype=np.int16).astype(np.int32) + 32768
    return _ALAW_ENCODE[idx].tobytes()


def alaw_decode(payload: bytes) -> np.ndarray:
    """Decode G.711 A-law bytes into int16 pcm samples."""
    return _ALAW_DECODE[np.frombuffer(payload, dtype=np.uint8)].copy()


# --------------------------------------------------------------------------- #
# Streaming rational resampler
# --------------------------------------------------------------------------- #


class StreamResampler:
    """Stateful polyphase resampler for a fixed rational rate ratio.

    `scipy.signal.resample_poly` is a one-shot function: applied per chunk it
    restarts the FIR state at every chunk boundary, which produces a click at
    each boundary and, because the output length is rounded per chunk, a slow
    drift against the input clock. This class instead keeps the filter history
    (the input tail) and the absolute output index across calls, and evaluates
    the exact polyphase identity

        y[j] = sum_k h[k] * x_up[j * down - k],   x_up[up * n] = x[n]

    on absolute sample indices. Chunking therefore cannot change the result:
    feeding the same signal in any chunk sizes yields the same output samples,
    and the sample count stays exactly `up / down` of the input in the long run
    (for 24 kHz -> 16 kHz, up = 2 and down = 3, so 2/3 with no rounding error).
    """

    def __init__(self, src_rate: int, dst_rate: int, half_taps: int = 16) -> None:
        if src_rate <= 0 or dst_rate <= 0:
            raise ValueError("sample rates must be positive")
        self.src_rate = src_rate
        self.dst_rate = dst_rate
        divisor = gcd(src_rate, dst_rate)
        self._up = dst_rate // divisor
        self._down = src_rate // divisor
        if self._up == 1 and self._down == 1:
            self._taps = np.array([1.0], dtype=np.float64)
        else:
            length = 2 * half_taps * max(self._up, self._down) + 1
            self._taps = self._up * firwin(
                length, 1.0 / max(self._up, self._down), window=("kaiser", 5.0)
            )
        self.reset()

    @property
    def ratio(self) -> tuple[int, int]:
        """The reduced (up, down) factors."""
        return self._up, self._down

    def reset(self) -> None:
        """Drop the filter history and restart the output clock."""
        # Pre-load the history with zeros so that the very first output samples
        # have a complete tap window (the signal is zero before time 0).
        pad = (len(self._taps) - 1) // self._up + 1
        self._buf = np.zeros(pad, dtype=np.float64)
        self._start = -pad  # absolute input index of self._buf[0]
        self._next_out = 0  # absolute index of the next output sample

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Resample one chunk, returning every output sample now determined."""
        x = np.asarray(samples, dtype=np.float64).reshape(-1)
        if self._up == 1 and self._down == 1:
            return x.astype(np.float32, copy=True)
        if x.size:
            self._buf = np.concatenate((self._buf, x))
        up, down, taps = self._up, self._down, self._taps
        last = self._start + len(self._buf) - 1
        # Largest j with floor(j * down / up) <= last, i.e. no future input needed.
        j_hi = -(-(up * (last + 1)) // down) - 1
        if j_hi < self._next_out:
            return np.zeros(0, dtype=np.float32)
        upsampled = upfirdn(taps, self._buf, up, 1)
        idx = np.arange(self._next_out, j_hi + 1, dtype=np.int64) * down - self._start * up
        out = upsampled[idx]
        self._next_out = j_hi + 1
        # Keep only the history still needed by the next output sample.
        needed = (self._next_out * down - len(taps)) // up + 1
        keep = min(max(needed - self._start, 0), len(self._buf))
        if keep:
            self._buf = self._buf[keep:]
            self._start += keep
        return out.astype(np.float32, copy=False)

    def process_pcm16(self, samples: np.ndarray) -> np.ndarray:
        """Resample int16 samples and return int16 samples."""
        return float_to_pcm16(self.process(pcm16_to_float(samples)))


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


class AudioFramer:
    """Accumulate samples and hand them out in fixed-size units.

    Used to feed the model in exact 1 s units at 16 kHz. The incomplete tail is
    retained between calls and dropped by `clear()` (`input_audio_buffer.clear`).
    """

    def __init__(self, unit_samples: int = UNIT_SAMPLES, dtype: str = "int16") -> None:
        if unit_samples <= 0:
            raise ValueError("unit_samples must be positive")
        self.unit_samples = unit_samples
        self._dtype = np.dtype(dtype)
        self._pending = np.zeros(0, dtype=self._dtype)

    @property
    def pending(self) -> int:
        """Number of buffered samples that do not yet form a full unit."""
        return int(self._pending.size)

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Add samples and return every complete unit they produced."""
        chunk = np.asarray(samples, dtype=self._dtype).reshape(-1)
        if chunk.size:
            self._pending = np.concatenate((self._pending, chunk))
        count = self._pending.size // self.unit_samples
        if count == 0:
            return []
        split = count * self.unit_samples
        units = [
            self._pending[i : i + self.unit_samples] for i in range(0, split, self.unit_samples)
        ]
        self._pending = self._pending[split:]
        return units

    def flush(self, pad: bool = True) -> np.ndarray | None:
        """Return the remainder as one unit (zero padded) and empty the buffer."""
        if self._pending.size == 0:
            return None
        tail = self._pending
        self._pending = np.zeros(0, dtype=self._dtype)
        if not pad:
            return tail
        out = np.zeros(self.unit_samples, dtype=self._dtype)
        out[: tail.size] = tail
        return out

    def clear(self) -> None:
        """Drop the incomplete unit (`input_audio_buffer.clear`)."""
        self._pending = np.zeros(0, dtype=self._dtype)


class InputAudioStream:
    """Client audio in, 1 s units at the model rate out.

    Combines format decoding, resampling and framing so the session layer can
    hand raw `input_audio_buffer.append` payloads straight in.
    """

    def __init__(self, audio_format: str = "pcm16", unit_samples: int = UNIT_SAMPLES) -> None:
        self.audio_format = audio_format
        self._src_rate = input_sample_rate(audio_format)
        self._resampler = StreamResampler(self._src_rate, MODEL_SAMPLE_RATE)
        self._framer = AudioFramer(unit_samples)

    @property
    def source_rate(self) -> int:
        """Sample rate of the client-side audio."""
        return self._src_rate

    @property
    def pending(self) -> int:
        """Buffered samples at the model rate that do not yet form a unit."""
        return self._framer.pending

    def decode(self, payload: str) -> np.ndarray:
        """Decode one base64 payload in the negotiated format into int16 samples."""
        if self.audio_format in ("g711_ulaw", "audio/pcmu"):
            return ulaw_decode(base64.b64decode(payload, validate=False))
        if self.audio_format in ("g711_alaw", "audio/pcma"):
            return alaw_decode(base64.b64decode(payload, validate=False))
        return decode_base64_pcm16(payload)

    def append(self, payload: str) -> list[np.ndarray]:
        """Decode, resample and frame one payload; return the complete units."""
        return self.push(self.decode(payload))

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Resample and frame already decoded int16 samples."""
        return self._framer.push(self._resampler.process_pcm16(samples))

    def flush(self, pad: bool = True) -> np.ndarray | None:
        """Return the remainder as a unit (`input_audio_buffer.commit`)."""
        return self._framer.flush(pad=pad)

    def clear(self) -> None:
        """Drop the incomplete unit (`input_audio_buffer.clear`)."""
        self._framer.clear()


def encode_output_audio(
    samples: np.ndarray, audio_format: str, resampler: StreamResampler | None = None
) -> str:
    """Encode model output (int16 at 24 kHz) as a base64 payload in `audio_format`.

    For a g711 format the caller must pass a `StreamResampler(24000, 8000)` and
    reuse it for the whole response so that the chunk boundaries stay seamless.
    """
    data = np.asarray(samples, dtype=np.int16)
    if audio_format in ("pcm16", "audio/pcm", "audio/pcm16"):
        return encode_base64_pcm16(data)
    if resampler is None:
        raise ValueError("a StreamResampler is required for g711 output")
    narrow = resampler.process_pcm16(data)
    if audio_format in ("g711_ulaw", "audio/pcmu"):
        return base64.b64encode(ulaw_encode(narrow)).decode("ascii")
    if audio_format in ("g711_alaw", "audio/pcma"):
        return base64.b64encode(alaw_encode(narrow)).decode("ascii")
    raise ValueError(f"unsupported audio format: {audio_format!r}")
