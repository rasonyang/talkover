"""Tests for the typed configuration schema (DESIGN.md section 9)."""

from __future__ import annotations

from pathlib import Path

import pytest

from talkover.config import (
    TOKEN2WAV_TIMESTEPS_RANGE,
    AsrConfig,
    ConfigError,
    EngineConfig,
    load_brain_config,
    load_config,
    resolve_asr_auto,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVE_EXAMPLE = REPO_ROOT / "configs" / "serve.example.yaml"
BRAIN_EXAMPLE = REPO_ROOT / "configs" / "brain.business.example.yaml"


def write_config(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# Example configs
# --------------------------------------------------------------------------------------


def test_serve_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALKOVER_API_KEY", "serve-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    cfg = load_config(SERVE_EXAMPLE)

    assert cfg.server.api_key == "serve-key"
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8000
    assert cfg.engine.device == "mps"
    assert cfg.engine.talker_device is None
    assert cfg.engine.init_vision is False
    assert cfg.engine.context_max_units == 128
    assert cfg.engine.attn_implementation == "sdpa"
    assert cfg.asr.backend == "mlx_whisper"
    assert cfg.realtime.memory_slate_max_tokens == 256
    assert cfg.realtime.trailing_silence_sec == 8.0
    assert cfg.brain.provider == "business"
    assert cfg.brain.llm.kind == "anthropic"
    assert cfg.brain.llm.api_key == "deepseek-key"
    assert cfg.brain.business_api.timeout_sec == 3.0
    assert "转人工" in cfg.brain.transfer_keywords
    assert cfg.engine.token2wav_dir is None
    assert cfg.engine.ref_audio_path is None
    assert cfg.brain.spoken_numbers is False
    assert cfg.brain.max_rounds == 6
    assert cfg.brain.fallback_texts.round_cap is not None


def test_brain_example_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")

    brain = load_brain_config(BRAIN_EXAMPLE)

    assert brain.provider == "business"
    assert brain.llm.model == "deepseek-flash"
    assert brain.llm.api_key == "deepseek-key"
    assert brain.business_api.base_url == "http://127.0.0.1:9100"
    assert brain.transfer_keywords == ("转人工", "人工客服", "找个人")
    assert brain.spoken_numbers is False
    assert brain.max_rounds == 6
    # Runtime data, like transfer_keywords: spoken to the customer, so Chinese here.
    assert brain.fallback_texts.no_answer == "抱歉，我没有听清，您可以再说一遍吗？"


# --------------------------------------------------------------------------------------
# Environment expansion
# --------------------------------------------------------------------------------------


def test_env_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TALKOVER_API_KEY", "secret-123")
    monkeypatch.setenv("BRAIN_HOST", "brain.internal")
    path = write_config(
        tmp_path,
        "server:\n"
        "  api_key: ${TALKOVER_API_KEY}\n"
        "brain:\n"
        "  business_api:\n"
        "    base_url: http://${BRAIN_HOST}:9100\n"
        "  transfer_keywords: ['${TALKOVER_API_KEY}']\n",
    )

    cfg = load_config(path)

    assert cfg.server.api_key == "secret-123"
    assert cfg.brain.business_api.base_url == "http://brain.internal:9100"
    assert cfg.brain.transfer_keywords == ("secret-123",)


def test_env_expansion_unset_becomes_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TALKOVER_API_KEY", raising=False)
    path = write_config(tmp_path, "server:\n  api_key: ${TALKOVER_API_KEY}\n")

    assert load_config(path).server.api_key == ""


# --------------------------------------------------------------------------------------
# `auto` resolution
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "backend", "asr_device", "compute_type"),
    [
        ("mps", "mlx_whisper", "mps", "float16"),
        ("cuda", "faster_whisper", "cuda", "float16"),
        ("cuda:1", "faster_whisper", "cuda:1", "float16"),
        ("cpu", "faster_whisper", "cpu", "int8"),
    ],
)
def test_asr_auto_resolution(
    tmp_path: Path, device: str, backend: str, asr_device: str, compute_type: str
) -> None:
    path = write_config(
        tmp_path,
        f"engine:\n  device: {device}\n"
        "asr:\n  backend: auto\n  device: auto\n  compute_type: auto\n",
    )

    cfg = load_config(path)

    assert cfg.asr.backend == backend
    assert cfg.asr.device == asr_device
    assert cfg.asr.compute_type == compute_type


def test_asr_explicit_values_are_kept(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "engine:\n  device: cuda:0\n"
        "asr:\n  backend: faster_whisper\n  device: cpu\n  compute_type: int8\n",
    )

    cfg = load_config(path)

    assert (cfg.asr.backend, cfg.asr.device, cfg.asr.compute_type) == (
        "faster_whisper",
        "cpu",
        "int8",
    )


def test_faster_whisper_on_mps_falls_back_to_cpu() -> None:
    # CTranslate2 has no Metal backend (DESIGN.md 4.1).
    resolved = resolve_asr_auto(AsrConfig(backend="faster_whisper"), EngineConfig(device="mps"))

    assert resolved.device == "cpu"
    assert resolved.compute_type == "int8"


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("device", ["metal", "cuda:", "cuda:x", "gpu", "MPS", ""])
def test_invalid_engine_device_is_rejected(tmp_path: Path, device: str) -> None:
    path = write_config(tmp_path, f"engine:\n  device: {device!r}\n")

    with pytest.raises(ConfigError, match="engine.device"):
        load_config(path)


@pytest.mark.parametrize("device", ["mps", "cpu", "cuda", "cuda:0", "cuda:3"])
def test_valid_engine_devices_are_accepted(tmp_path: Path, device: str) -> None:
    path = write_config(tmp_path, f"engine:\n  device: {device}\n")

    assert load_config(path).engine.device == device


def test_invalid_asr_backend_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "asr:\n  backend: whisper_cpp\n")

    with pytest.raises(ConfigError, match="asr.backend"):
        load_config(path)


def test_invalid_llm_kind_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "brain:\n  llm:\n    kind: gemini\n")

    with pytest.raises(ConfigError, match="brain.llm.kind"):
        load_config(path)


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("server:\n  listen_on: 0.0.0.0:8000\n", "server.listen_on"),
        ("engine:\n  devise: mps\n", "engine.devise"),
        ("asr:\n  beam_size: 5\n", "asr.beam_size"),
        ("brain:\n  llm:\n    temperature: 0.2\n", "brain.llm.temperature"),
        ("brain:\n  business_api:\n    retries: 2\n", "brain.business_api.retries"),
        ("realtime:\n  voice: alloy\n", "realtime.voice"),
    ],
)
def test_unknown_keys_are_rejected(tmp_path: Path, text: str, match: str) -> None:
    path = write_config(tmp_path, text)

    with pytest.raises(ConfigError, match=match.replace(".", r"\.")):
        load_config(path)


def test_unknown_top_level_key_in_brain_config_is_rejected(tmp_path: Path) -> None:
    path = write_config(tmp_path, "brain:\n  provider: business\nengine:\n  device: mps\n")

    with pytest.raises(ConfigError, match="engine"):
        load_brain_config(path)


def test_attn_implementation_and_talker_device_are_accepted(tmp_path: Path) -> None:
    # M4 seams: both keys must survive the schema (docs/tasks.md M4 interface list).
    path = write_config(
        tmp_path,
        "engine:\n"
        "  device: cuda:0\n"
        "  talker_device: cuda:1\n"
        "  attn_implementation: flash_attention_2\n"
        "  quantize_thinker: int8\n",
    )

    cfg = load_config(path)

    assert cfg.engine.talker_device == "cuda:1"
    assert cfg.engine.attn_implementation == "flash_attention_2"
    assert cfg.engine.quantize_thinker == "int8"
    assert cfg.engine.device_kind == "cuda"


def test_engine_asset_paths_default_to_null_and_are_overridable(tmp_path: Path) -> None:
    # null token2wav_dir resolves to <base_model>/assets/token2wav, null ref_audio_path
    # keeps the checkpoint's voice (DESIGN.md 9).
    cfg = load_config(write_config(tmp_path, "engine:\n  device: mps\n"))
    assert cfg.engine.token2wav_dir is None
    assert cfg.engine.ref_audio_path is None

    path = write_config(
        tmp_path,
        "engine:\n  token2wav_dir: /models/token2wav\n  ref_audio_path: /voices/agent.wav\n",
    )
    cfg = load_config(path)

    assert cfg.engine.token2wav_dir == "/models/token2wav"
    assert cfg.engine.ref_audio_path == "/voices/agent.wav"


def test_token2wav_timesteps_defaults_to_upstreams_ten_and_is_validated(tmp_path: Path) -> None:
    # The vocoder's flow-matching step count (DESIGN.md 4.4 / 9): upstream's default is 10.
    cfg = load_config(write_config(tmp_path, "engine:\n  device: mps\n"))
    assert cfg.engine.token2wav_timesteps == 10

    cfg = load_config(write_config(tmp_path, "engine:\n  token2wav_timesteps: 5\n"))
    assert cfg.engine.token2wav_timesteps == 5

    low, high = TOKEN2WAV_TIMESTEPS_RANGE
    for value in (low, high):
        path = write_config(tmp_path, f"engine:\n  token2wav_timesteps: {value}\n")
        assert load_config(path).engine.token2wav_timesteps == value

    for value in (low - 1, high + 1):
        with pytest.raises(ConfigError, match="engine.token2wav_timesteps"):
            load_config(write_config(tmp_path, f"engine:\n  token2wav_timesteps: {value}\n"))

    with pytest.raises(ConfigError, match="engine.token2wav_timesteps"):
        load_config(write_config(tmp_path, "engine:\n  token2wav_timesteps: fast\n"))

    with pytest.raises(ConfigError, match="engine.token2wav_timesteps"):
        load_config(write_config(tmp_path, "engine:\n  token2wav_timesteps: true\n"))


def test_invalid_engine_asset_paths_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="engine.token2wav_dir"):
        load_config(write_config(tmp_path, "engine:\n  token2wav_dir: 3\n"))

    with pytest.raises(ConfigError, match="engine.ref_audio_path"):
        load_config(write_config(tmp_path, "engine:\n  ref_audio_path: [a]\n"))


def test_brain_loop_defaults_and_overrides(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, "brain:\n  provider: business\n"))
    assert cfg.brain.spoken_numbers is False
    assert cfg.brain.max_rounds == 6
    assert cfg.brain.fallback_texts.round_cap is None
    assert cfg.brain.fallback_texts.failure is None
    assert cfg.brain.fallback_texts.no_answer is None

    path = write_config(
        tmp_path,
        "brain:\n"
        "  spoken_numbers: true\n"
        "  max_rounds: 3\n"
        "  fallback_texts:\n"
        "    round_cap: 抱歉，帮您转人工。\n"
        "    no_answer: null\n",
    )
    cfg = load_config(path)

    assert cfg.brain.spoken_numbers is True
    assert cfg.brain.max_rounds == 3
    assert cfg.brain.fallback_texts.round_cap == "抱歉，帮您转人工。"
    assert cfg.brain.fallback_texts.failure is None
    assert cfg.brain.fallback_texts.no_answer is None


def test_invalid_brain_loop_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="brain.max_rounds"):
        load_config(write_config(tmp_path, "brain:\n  max_rounds: 0\n"))

    with pytest.raises(ConfigError, match="brain.spoken_numbers"):
        load_config(write_config(tmp_path, "brain:\n  spoken_numbers: 1\n"))

    with pytest.raises(ConfigError, match=r"brain\.fallback_texts\.failure"):
        load_config(write_config(tmp_path, "brain:\n  fallback_texts:\n    failure: 7\n"))

    with pytest.raises(ConfigError, match=r"brain\.fallback_texts\.retry"):
        load_config(write_config(tmp_path, "brain:\n  fallback_texts:\n    retry: hi\n"))


def test_realtime_defaults(tmp_path: Path) -> None:
    # Defaults match upstream gander_runtime (DESIGN.md 5.5 / 5.7).
    cfg = load_config(write_config(tmp_path, "server:\n  api_key: k\n"))

    assert cfg.realtime.memory_slate_max_tokens == 256
    assert cfg.realtime.trailing_silence_sec == 8.0


def test_realtime_values_are_overridable(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "realtime:\n  memory_slate_max_tokens: 128\n  trailing_silence_sec: 0\n",
    )

    cfg = load_config(path)

    assert cfg.realtime.memory_slate_max_tokens == 128
    assert cfg.realtime.trailing_silence_sec == 0.0


def test_invalid_realtime_values_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="realtime.memory_slate_max_tokens"):
        load_config(write_config(tmp_path, "realtime:\n  memory_slate_max_tokens: 0\n"))

    with pytest.raises(ConfigError, match="realtime.trailing_silence_sec"):
        load_config(write_config(tmp_path, "realtime:\n  trailing_silence_sec: -1\n"))


def test_unknown_realtime_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"realtime\.turn_detection"):
        load_config(write_config(tmp_path, "realtime:\n  turn_detection: server_vad\n"))


def test_wrong_types_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="engine.init_vision"):
        load_config(write_config(tmp_path, "engine:\n  init_vision: yes please\n"))

    with pytest.raises(ConfigError, match="engine.context_max_units"):
        load_config(write_config(tmp_path, "engine:\n  context_max_units: 0\n"))

    with pytest.raises(ConfigError, match="brain.transfer_keywords"):
        load_config(write_config(tmp_path, "brain:\n  transfer_keywords: 转人工\n"))


def test_invalid_listen_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="server.listen"):
        load_config(write_config(tmp_path, "server:\n  listen: 127.0.0.1\n"))


def test_top_level_must_be_a_mapping(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write_config(tmp_path, "- server\n- engine\n"))


def test_empty_file_yields_defaults(tmp_path: Path) -> None:
    cfg = load_config(write_config(tmp_path, ""))

    assert cfg.engine.device == "mps"
    assert cfg.asr.backend == "mlx_whisper"
    assert cfg.brain.transfer_keywords == ()
