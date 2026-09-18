"""Tests for the Realtime audio helpers (GPU-free)."""

from __future__ import annotations

import base64
import random

import numpy as np
import pytest

from talkover.realtime import audio


def _sine(freq: float, seconds: float, rate: int) -> np.ndarray:
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return audio.float_to_pcm16(0.5 * np.sin(2 * np.pi * freq * t))


def _fft_peak_hz(samples: np.ndarray, rate: int) -> float:
    windowed = audio.pcm16_to_float(samples) * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed))
    return float(np.argmax(spectrum) * rate / samples.size)


# --------------------------------------------------------------------------- #
# base64 pcm16
# --------------------------------------------------------------------------- #


def test_base64_pcm16_round_trip() -> None:
    rng = np.random.default_rng(0)
    samples = rng.integers(-32768, 32768, size=1024).astype(np.int16)
    encoded = audio.encode_base64_pcm16(samples)
    assert isinstance(encoded, str)
    decoded = audio.decode_base64_pcm16(encoded)
    assert decoded.dtype == np.int16
    assert np.array_equal(decoded, samples)


def test_base64_pcm16_is_little_endian() -> None:
    assert audio.encode_base64_pcm16(np.array([1, -1], dtype=np.int16)) == base64.b64encode(
        b"\x01\x00\xff\xff"
    ).decode("ascii")


def test_base64_pcm16_rejects_odd_length() -> None:
    with pytest.raises(ValueError):
        audio.decode_base64_pcm16(base64.b64encode(b"\x00").decode("ascii"))


def test_pcm16_float_round_trip() -> None:
    samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
    assert np.array_equal(audio.float_to_pcm16(audio.pcm16_to_float(samples)), samples)


# --------------------------------------------------------------------------- #
# G.711
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("codec", ["ulaw", "alaw"])
def test_g711_all_code_values_round_trip(codec: str) -> None:
    decode = getattr(audio, f"{codec}_decode")
    encode = getattr(audio, f"{codec}_encode")
    codes = bytes(range(256))
    linear = decode(codes)
    assert linear.dtype == np.int16
    re_encoded = encode(linear)
    if codec == "ulaw":
        # 0x7F and 0xFF are mu-law negative and positive zero; both decode to 0,
        # so the encoder can only return the canonical 0xFF for that sample.
        assert re_encoded[:127] == codes[:127]
        assert re_encoded[128:] == codes[128:]
        assert re_encoded[127] == 0xFF
    else:
        assert re_encoded == codes
    # Encoding is idempotent on decoded values for every code.
    assert np.array_equal(decode(re_encoded), linear)


@pytest.mark.parametrize("codec", ["ulaw", "alaw"])
def test_g711_linear_round_trip_within_quantization_error(codec: str) -> None:
    decode = getattr(audio, f"{codec}_decode")
    encode = getattr(audio, f"{codec}_encode")
    rng = np.random.default_rng(7)
    samples = np.clip(rng.normal(0.0, 6000.0, 20000), -32768, 32767).astype(np.int16)
    restored = decode(encode(samples))
    error = np.abs(restored.astype(np.int64) - samples.astype(np.int64))
    # G.711 is logarithmic: the step grows with the magnitude, roughly 6 % of the
    # value at the top segment, with a floor set by the smallest segment.
    assert np.all(error <= 0.07 * np.abs(samples.astype(np.int64)) + 264)


@pytest.mark.parametrize("codec", ["ulaw", "alaw"])
def test_g711_sizes_and_silence(codec: str) -> None:
    decode = getattr(audio, f"{codec}_decode")
    encode = getattr(audio, f"{codec}_encode")
    silence = np.zeros(160, dtype=np.int16)
    payload = encode(silence)
    assert len(payload) == 160
    assert decode(payload).size == 160
    assert np.all(np.abs(decode(payload)) <= 16)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #


def test_resample_24k_to_16k_keeps_frequency_and_length() -> None:
    src = _sine(440.0, 1.0, audio.CLIENT_SAMPLE_RATE)
    resampler = audio.StreamResampler(audio.CLIENT_SAMPLE_RATE, audio.MODEL_SAMPLE_RATE)
    assert resampler.ratio == (2, 3)
    out = resampler.process_pcm16(src)
    assert out.size == src.size * 2 // 3 == audio.MODEL_SAMPLE_RATE
    assert _fft_peak_hz(out, audio.MODEL_SAMPLE_RATE) == pytest.approx(440.0, abs=2.0)


def test_resample_8k_to_16k_and_24k_to_8k_lengths() -> None:
    up = audio.StreamResampler(audio.G711_SAMPLE_RATE, audio.MODEL_SAMPLE_RATE)
    assert up.process(np.zeros(800, dtype=np.float32)).size == 1600
    down = audio.StreamResampler(audio.CLIENT_SAMPLE_RATE, audio.G711_SAMPLE_RATE)
    assert down.process(np.zeros(2400, dtype=np.float32)).size == 800


def test_resample_is_chunk_invariant() -> None:
    src = _sine(700.0, 1.0, audio.CLIENT_SAMPLE_RATE)
    whole = audio.StreamResampler(24000, 16000).process_pcm16(src)

    rng = random.Random(3)
    streaming = audio.StreamResampler(24000, 16000)
    pieces: list[np.ndarray] = []
    offset = 0
    while offset < src.size:
        size = rng.randint(1, 997)
        pieces.append(streaming.process_pcm16(src[offset : offset + size]))
        offset += size
    chunked = np.concatenate(pieces)

    assert chunked.size == whole.size
    assert np.array_equal(chunked, whole)


def test_resample_has_no_boundary_click() -> None:
    # A continuous sine fed in 10 ms chunks must stay smooth: the largest
    # sample-to-sample step may not exceed the analytic slope of the sine.
    src = _sine(300.0, 0.5, audio.CLIENT_SAMPLE_RATE)
    resampler = audio.StreamResampler(24000, 16000)
    out = np.concatenate(
        [resampler.process_pcm16(src[i : i + 240]) for i in range(0, src.size, 240)]
    )
    steady = audio.pcm16_to_float(out[400:])
    max_step = np.abs(np.diff(steady)).max()
    expected = 0.5 * 2 * np.pi * 300.0 / audio.MODEL_SAMPLE_RATE
    assert max_step < expected * 1.1


def test_resample_identity_ratio() -> None:
    src = _sine(1000.0, 0.1, audio.MODEL_SAMPLE_RATE)
    same = audio.StreamResampler(16000, 16000)
    assert np.array_equal(same.process_pcm16(src), src)


def test_resample_reset_restarts_state() -> None:
    src = _sine(440.0, 0.25, audio.CLIENT_SAMPLE_RATE)
    resampler = audio.StreamResampler(24000, 16000)
    first = resampler.process_pcm16(src)
    resampler.reset()
    second = resampler.process_pcm16(src)
    assert np.array_equal(first, second)


# --------------------------------------------------------------------------- #
# Framing
# --------------------------------------------------------------------------- #


def test_framer_yields_units_and_keeps_remainder() -> None:
    framer = audio.AudioFramer()
    samples = np.arange(int(2.5 * audio.MODEL_SAMPLE_RATE), dtype=np.int16)
    units = framer.push(samples)
    assert len(units) == 2
    assert all(unit.size == audio.UNIT_SAMPLES for unit in units)
    assert np.array_equal(units[0], samples[: audio.UNIT_SAMPLES])
    assert np.array_equal(units[1], samples[audio.UNIT_SAMPLES : 2 * audio.UNIT_SAMPLES])
    assert framer.pending == audio.MODEL_SAMPLE_RATE // 2


def test_framer_clear_drops_remainder() -> None:
    framer = audio.AudioFramer()
    framer.push(np.zeros(int(2.5 * audio.MODEL_SAMPLE_RATE), dtype=np.int16))
    framer.clear()
    assert framer.pending == 0
    assert framer.flush() is None
    # After a clear the next unit boundary starts fresh.
    assert framer.push(np.ones(audio.UNIT_SAMPLES, dtype=np.int16)) != []
    assert framer.pending == 0


def test_framer_flush_pads_remainder() -> None:
    framer = audio.AudioFramer()
    framer.push(np.ones(1000, dtype=np.int16))
    padded = framer.flush()
    assert padded is not None
    assert padded.size == audio.UNIT_SAMPLES
    assert np.all(padded[:1000] == 1)
    assert np.all(padded[1000:] == 0)
    assert framer.pending == 0


def test_framer_irregular_chunks_match_single_push() -> None:
    rng = np.random.default_rng(11)
    samples = rng.integers(-1000, 1000, size=int(2.5 * audio.MODEL_SAMPLE_RATE)).astype(np.int16)
    at_once = audio.AudioFramer().push(samples)

    chunked_framer = audio.AudioFramer()
    chunked: list[np.ndarray] = []
    step = random.Random(5)
    offset = 0
    while offset < samples.size:
        size = step.randint(1, 5000)
        chunked.extend(chunked_framer.push(samples[offset : offset + size]))
        offset += size

    assert len(chunked) == len(at_once)
    for got, expected in zip(chunked, at_once, strict=True):
        assert np.array_equal(got, expected)
    assert chunked_framer.pending == audio.MODEL_SAMPLE_RATE // 2


# --------------------------------------------------------------------------- #
# InputAudioStream and output encoding
# --------------------------------------------------------------------------- #


def test_input_sample_rate_mapping() -> None:
    assert audio.input_sample_rate("pcm16") == 24000
    assert audio.input_sample_rate("g711_ulaw") == 8000
    assert audio.input_sample_rate("g711_alaw") == 8000
    with pytest.raises(ValueError):
        audio.input_sample_rate("opus")


def test_input_stream_pcm16_2_5_seconds() -> None:
    stream = audio.InputAudioStream("pcm16")
    src = _sine(440.0, 2.5, audio.CLIENT_SAMPLE_RATE)
    units = stream.append(audio.encode_base64_pcm16(src))
    assert len(units) == 2
    assert stream.pending == audio.MODEL_SAMPLE_RATE // 2
    stream.clear()
    assert stream.pending == 0
    assert _fft_peak_hz(units[1], audio.MODEL_SAMPLE_RATE) == pytest.approx(440.0, abs=4.0)


def test_input_stream_irregular_payloads_match_single_payload() -> None:
    src = _sine(440.0, 2.5, audio.CLIENT_SAMPLE_RATE)
    at_once = audio.InputAudioStream("pcm16").append(audio.encode_base64_pcm16(src))

    stream = audio.InputAudioStream("pcm16")
    units: list[np.ndarray] = []
    rng = random.Random(9)
    offset = 0
    while offset < src.size:
        size = rng.randint(1, 3000)
        units.extend(stream.append(audio.encode_base64_pcm16(src[offset : offset + size])))
        offset += size

    assert len(units) == len(at_once)
    for got, expected in zip(units, at_once, strict=True):
        assert np.array_equal(got, expected)


@pytest.mark.parametrize("fmt", ["g711_ulaw", "g711_alaw"])
def test_input_stream_g711_is_upsampled_to_model_rate(fmt: str) -> None:
    stream = audio.InputAudioStream(fmt)
    assert stream.source_rate == audio.G711_SAMPLE_RATE
    src = _sine(440.0, 2.0, audio.G711_SAMPLE_RATE)
    encode = audio.ulaw_encode if fmt == "g711_ulaw" else audio.alaw_encode
    payload = base64.b64encode(encode(src)).decode("ascii")
    units = stream.append(payload)
    assert len(units) == 2
    assert _fft_peak_hz(units[1], audio.MODEL_SAMPLE_RATE) == pytest.approx(440.0, abs=4.0)


def test_encode_output_audio_pcm16_passthrough() -> None:
    src = _sine(440.0, 0.1, audio.CLIENT_SAMPLE_RATE)
    assert audio.encode_output_audio(src, "pcm16") == audio.encode_base64_pcm16(src)


@pytest.mark.parametrize("fmt", ["g711_ulaw", "g711_alaw"])
def test_encode_output_audio_g711_downsamples_to_8k(fmt: str) -> None:
    src = _sine(440.0, 1.0, audio.CLIENT_SAMPLE_RATE)
    resampler = audio.StreamResampler(audio.CLIENT_SAMPLE_RATE, audio.G711_SAMPLE_RATE)
    payload = base64.b64decode(audio.encode_output_audio(src, fmt, resampler))
    assert len(payload) == audio.G711_SAMPLE_RATE
    decode = audio.ulaw_decode if fmt == "g711_ulaw" else audio.alaw_decode
    assert _fft_peak_hz(decode(payload), audio.G711_SAMPLE_RATE) == pytest.approx(440.0, abs=4.0)


def test_encode_output_audio_requires_resampler_for_g711() -> None:
    with pytest.raises(ValueError):
        audio.encode_output_audio(np.zeros(10, dtype=np.int16), "g711_ulaw")
