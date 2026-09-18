"""Configuration loading: read the YAML config once at startup and validate it.

The layout follows DESIGN.md section 9. Values are returned as frozen dataclasses so
the rest of the code never touches raw dicts. `${ENV_VAR}` references in string values
are expanded at load time, `auto` values in the `asr` section are resolved from
`engine.device`, and unknown keys are rejected.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "AsrConfig",
    "BrainConfig",
    "BusinessApiConfig",
    "ConfigError",
    "EngineConfig",
    "FallbackTextsConfig",
    "LLMConfig",
    "RealtimeConfig",
    "ServerConfig",
    "TalkoverConfig",
    "load_brain_config",
    "load_config",
]


class ConfigError(ValueError):
    """Raised when a configuration file is malformed, incomplete or inconsistent."""


_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

_DEVICE_RE = re.compile(r"^(mps|cpu|cuda(:\d+)?)$")

ASR_BACKENDS = ("auto", "mlx_whisper", "faster_whisper", "sensevoice")
LLM_KINDS = ("anthropic", "openai_compat")
DTYPES = ("bfloat16", "float16", "float32")
QUANTIZATIONS = (None, "int8")
MEDIA_MODES = ("voice", "video")
ATTN_IMPLEMENTATIONS = ("sdpa", "eager", "flash_attention_2")


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ServerConfig:
    """`server` section: the WebSocket listener and its single Bearer key."""

    listen: str = "127.0.0.1:8000"
    api_key: str = ""

    @property
    def host(self) -> str:
        return self.listen.rsplit(":", 1)[0]

    @property
    def port(self) -> int:
        return int(self.listen.rsplit(":", 1)[1])


@dataclass(frozen=True, slots=True)
class EngineConfig:
    """`engine` section: device placement, precision and Gander checkpoints.

    `token2wav_dir` is null by default, which resolves to `<base_model>/assets/token2wav`
    (`talkover.engine.session.default_token2wav_dir`); `ref_audio_path` is null by
    default, which keeps the Talker checkpoint's own voice.
    """

    device: str = "mps"
    talker_device: str | None = None
    dtype: str = "bfloat16"
    quantize_thinker: str | None = None
    init_vision: bool = False
    base_model: str = ""
    thinker_checkpoint: str = ""
    talker_checkpoint: str = ""
    context_max_units: int = 128
    sliding_window_mode: str = "context_slate"
    media_mode: str = "voice"
    attn_implementation: str = "sdpa"
    token2wav_dir: str | None = None
    ref_audio_path: str | None = None

    @property
    def device_kind(self) -> str:
        """`mps`, `cuda` or `cpu`, with the optional CUDA index stripped."""
        return self.device.split(":", 1)[0]


@dataclass(frozen=True, slots=True)
class AsrConfig:
    """`asr` section: the side-channel transcription backend.

    `backend`, `device` and `compute_type` accept `auto` in the file; after loading
    they always hold a concrete value resolved from `engine.device`.
    """

    backend: str = "auto"
    model: str = "large-v3-turbo"
    device: str = "auto"
    compute_type: str = "auto"


@dataclass(frozen=True, slots=True)
class RealtimeConfig:
    """`realtime` section: protocol-layer limits that are not device or model settings.

    `memory_slate_max_tokens` caps the `session.instructions` text written into the task
    slate (DESIGN.md 5.5); `trailing_silence_sec` is the trailing-silence window, which is
    also the window a disconnected session may reconnect within (DESIGN.md 5.7). Both
    defaults match upstream `gander_runtime`.
    """

    memory_slate_max_tokens: int = 256
    trailing_silence_sec: float = 8.0


@dataclass(frozen=True, slots=True)
class LLMConfig:
    """`brain.llm` section: the business LLM behind the `BrainLLM` protocol."""

    kind: str = "anthropic"
    base_url: str = "https://api.deepseek.com/anthropic"
    api_key: str = ""
    model: str = "deepseek-flash"


@dataclass(frozen=True, slots=True)
class BusinessApiConfig:
    """`brain.business_api` section: the HTTP API behind `query_ticket` / `query_order`."""

    base_url: str = "http://127.0.0.1:9100"
    timeout_sec: float = 3.0


@dataclass(frozen=True, slots=True)
class FallbackTextsConfig:
    """`brain.fallback_texts` section: what the Brain says when the loop cannot answer.

    Every key is optional and defaults to `None`, which means "use the built-in English
    constant in `talkover.brain.session`" (`ROUND_CAP_RESULT_TEXT`, `FAILURE_RESULT_TEXT`,
    `NO_ANSWER_RESULT_TEXT`). Like `transfer_keywords`, the values are runtime data: they
    are spoken to the customer and may be written in any language.
    """

    round_cap: str | None = None
    failure: str | None = None
    no_answer: str | None = None


@dataclass(frozen=True, slots=True)
class BrainConfig:
    """`brain` section: the customer-service Brain provider."""

    provider: str = "business"
    llm: LLMConfig = field(default_factory=LLMConfig)
    business_api: BusinessApiConfig = field(default_factory=BusinessApiConfig)
    # Runtime data: the Chinese phrases customers say to ask for a human agent.
    transfer_keywords: tuple[str, ...] = ()
    spoken_numbers: bool = False
    max_rounds: int = 6
    fallback_texts: FallbackTextsConfig = field(default_factory=FallbackTextsConfig)


@dataclass(frozen=True, slots=True)
class TalkoverConfig:
    """The whole configuration file."""

    server: ServerConfig = field(default_factory=ServerConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    realtime: RealtimeConfig = field(default_factory=RealtimeConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)


# --------------------------------------------------------------------------------------
# Generic helpers
# --------------------------------------------------------------------------------------


def expand_env(value: Any) -> Any:
    """Expand `${ENV_VAR}` references in every string of a nested structure.

    An unset variable expands to an empty string; `talkover check` reports the
    resulting empty credentials rather than failing at parse time.
    """
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def _as_mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
    return value


def _reject_unknown(data: dict[str, Any], cls: type, path: str) -> None:
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        allowed = ", ".join(sorted(known))
        where = f"{path}." if path else ""
        raise ConfigError(
            f"unknown config key(s): {', '.join(f'{where}{k}' for k in unknown)}; "
            f"allowed keys at {path or '<top level>'}: {allowed}"
        )


def _check_str(value: Any, path: str, *, allow_none: bool = False) -> Any:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise ConfigError(f"{path}: expected a string, got {type(value).__name__}")
    return value


def _check_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected a boolean, got {type(value).__name__}")
    return value


def _check_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{path}: expected an integer, got {type(value).__name__}")
    return value


def _check_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{path}: expected a number, got {type(value).__name__}")
    return float(value)


def _check_choice(value: Any, path: str, choices: tuple[Any, ...]) -> Any:
    if value not in choices:
        rendered = ", ".join("null" if c is None else repr(c) for c in choices)
        raise ConfigError(f"{path}: expected one of {rendered}, got {value!r}")
    return value


# --------------------------------------------------------------------------------------
# Section parsers
# --------------------------------------------------------------------------------------


def _parse_server(raw: Any) -> ServerConfig:
    data = _as_mapping(raw, "server")
    _reject_unknown(data, ServerConfig, "server")
    values: dict[str, Any] = {}
    if "listen" in data:
        listen = _check_str(data["listen"], "server.listen")
        host, _, port = listen.rpartition(":")
        if not host or not port.isdigit():
            raise ConfigError(f"server.listen: expected 'host:port', got {listen!r}")
        values["listen"] = listen
    if "api_key" in data:
        values["api_key"] = _check_str(data["api_key"], "server.api_key")
    return ServerConfig(**values)


def _parse_engine(raw: Any) -> EngineConfig:
    data = _as_mapping(raw, "engine")
    _reject_unknown(data, EngineConfig, "engine")
    values: dict[str, Any] = {}

    if "device" in data:
        device = _check_str(data["device"], "engine.device")
        if not _DEVICE_RE.match(device):
            raise ConfigError(
                f"engine.device: expected 'mps', 'cuda', 'cuda:<n>' or 'cpu', got {device!r}"
            )
        values["device"] = device
    if "talker_device" in data:
        talker_device = _check_str(data["talker_device"], "engine.talker_device", allow_none=True)
        if talker_device is not None and not _DEVICE_RE.match(talker_device):
            raise ConfigError(
                "engine.talker_device: expected null, 'mps', 'cuda', 'cuda:<n>' or 'cpu', "
                f"got {talker_device!r}"
            )
        values["talker_device"] = talker_device
    if "dtype" in data:
        values["dtype"] = _check_choice(data["dtype"], "engine.dtype", DTYPES)
    if "quantize_thinker" in data:
        values["quantize_thinker"] = _check_choice(
            data["quantize_thinker"], "engine.quantize_thinker", QUANTIZATIONS
        )
    if "init_vision" in data:
        values["init_vision"] = _check_bool(data["init_vision"], "engine.init_vision")
    for key in ("base_model", "thinker_checkpoint", "talker_checkpoint"):
        if key in data:
            values[key] = _check_str(data[key], f"engine.{key}")
    if "context_max_units" in data:
        units = _check_int(data["context_max_units"], "engine.context_max_units")
        if units <= 0:
            raise ConfigError(f"engine.context_max_units: expected a positive integer, got {units}")
        values["context_max_units"] = units
    if "sliding_window_mode" in data:
        values["sliding_window_mode"] = _check_str(
            data["sliding_window_mode"], "engine.sliding_window_mode"
        )
    if "media_mode" in data:
        values["media_mode"] = _check_choice(data["media_mode"], "engine.media_mode", MEDIA_MODES)
    if "attn_implementation" in data:
        values["attn_implementation"] = _check_choice(
            data["attn_implementation"], "engine.attn_implementation", ATTN_IMPLEMENTATIONS
        )
    for key in ("token2wav_dir", "ref_audio_path"):
        if key in data:
            values[key] = _check_str(data[key], f"engine.{key}", allow_none=True)
    return EngineConfig(**values)


def _parse_asr(raw: Any, engine: EngineConfig) -> AsrConfig:
    data = _as_mapping(raw, "asr")
    _reject_unknown(data, AsrConfig, "asr")
    values: dict[str, Any] = {}
    if "backend" in data:
        values["backend"] = _check_choice(data["backend"], "asr.backend", ASR_BACKENDS)
    if "model" in data:
        values["model"] = _check_str(data["model"], "asr.model")
    if "device" in data:
        device = _check_str(data["device"], "asr.device")
        if device != "auto" and not _DEVICE_RE.match(device):
            raise ConfigError(
                f"asr.device: expected 'auto', 'mps', 'cuda', 'cuda:<n>' or 'cpu', got {device!r}"
            )
        values["device"] = device
    if "compute_type" in data:
        values["compute_type"] = _check_str(data["compute_type"], "asr.compute_type")
    return resolve_asr_auto(AsrConfig(**values), engine)


def resolve_asr_auto(asr: AsrConfig, engine: EngineConfig) -> AsrConfig:
    """Resolve `auto` in `asr.backend` / `asr.device` / `asr.compute_type`.

    DESIGN.md section 4.3: MPS uses mlx-whisper on the Metal device, CUDA uses
    faster-whisper float16 on the same card, and CPU falls back to faster-whisper int8.
    """
    backend = asr.backend
    if backend == "auto":
        backend = "mlx_whisper" if engine.device_kind == "mps" else "faster_whisper"

    device = asr.device
    if device == "auto":
        device = "mps" if backend == "mlx_whisper" else engine.device
        if backend == "faster_whisper" and engine.device_kind == "mps":
            # faster-whisper (CTranslate2) has no Metal backend.
            device = "cpu"

    compute_type = asr.compute_type
    if compute_type == "auto":
        compute_type = "int8" if device.split(":", 1)[0] == "cpu" else "float16"

    return AsrConfig(backend=backend, model=asr.model, device=device, compute_type=compute_type)


def _parse_realtime(raw: Any) -> RealtimeConfig:
    data = _as_mapping(raw, "realtime")
    _reject_unknown(data, RealtimeConfig, "realtime")
    values: dict[str, Any] = {}
    if "memory_slate_max_tokens" in data:
        tokens = _check_int(data["memory_slate_max_tokens"], "realtime.memory_slate_max_tokens")
        if tokens <= 0:
            raise ConfigError(
                f"realtime.memory_slate_max_tokens: expected a positive integer, got {tokens}"
            )
        values["memory_slate_max_tokens"] = tokens
    if "trailing_silence_sec" in data:
        silence = _check_number(data["trailing_silence_sec"], "realtime.trailing_silence_sec")
        if silence < 0:
            raise ConfigError(
                f"realtime.trailing_silence_sec: expected a non-negative number, got {silence}"
            )
        values["trailing_silence_sec"] = silence
    return RealtimeConfig(**values)


def _parse_llm(raw: Any) -> LLMConfig:
    data = _as_mapping(raw, "brain.llm")
    _reject_unknown(data, LLMConfig, "brain.llm")
    values: dict[str, Any] = {}
    if "kind" in data:
        values["kind"] = _check_choice(data["kind"], "brain.llm.kind", LLM_KINDS)
    for key in ("base_url", "api_key", "model"):
        if key in data:
            values[key] = _check_str(data[key], f"brain.llm.{key}")
    return LLMConfig(**values)


def _parse_business_api(raw: Any) -> BusinessApiConfig:
    data = _as_mapping(raw, "brain.business_api")
    _reject_unknown(data, BusinessApiConfig, "brain.business_api")
    values: dict[str, Any] = {}
    if "base_url" in data:
        values["base_url"] = _check_str(data["base_url"], "brain.business_api.base_url")
    if "timeout_sec" in data:
        timeout = _check_number(data["timeout_sec"], "brain.business_api.timeout_sec")
        if timeout <= 0:
            raise ConfigError(
                f"brain.business_api.timeout_sec: expected a positive number, got {timeout}"
            )
        values["timeout_sec"] = timeout
    return BusinessApiConfig(**values)


def _parse_fallback_texts(raw: Any) -> FallbackTextsConfig:
    data = _as_mapping(raw, "brain.fallback_texts")
    _reject_unknown(data, FallbackTextsConfig, "brain.fallback_texts")
    values: dict[str, Any] = {}
    for key in ("round_cap", "failure", "no_answer"):
        if key in data:
            values[key] = _check_str(data[key], f"brain.fallback_texts.{key}", allow_none=True)
    return FallbackTextsConfig(**values)


def _parse_brain(raw: Any) -> BrainConfig:
    data = _as_mapping(raw, "brain")
    _reject_unknown(data, BrainConfig, "brain")
    values: dict[str, Any] = {}
    if "provider" in data:
        values["provider"] = _check_str(data["provider"], "brain.provider")
    if "llm" in data:
        values["llm"] = _parse_llm(data["llm"])
    if "business_api" in data:
        values["business_api"] = _parse_business_api(data["business_api"])
    if "transfer_keywords" in data:
        keywords = data["transfer_keywords"]
        if not isinstance(keywords, list):
            raise ConfigError(
                f"brain.transfer_keywords: expected a list, got {type(keywords).__name__}"
            )
        for i, item in enumerate(keywords):
            _check_str(item, f"brain.transfer_keywords[{i}]")
        values["transfer_keywords"] = tuple(keywords)
    if "spoken_numbers" in data:
        values["spoken_numbers"] = _check_bool(data["spoken_numbers"], "brain.spoken_numbers")
    if "max_rounds" in data:
        rounds = _check_int(data["max_rounds"], "brain.max_rounds")
        if rounds <= 0:
            raise ConfigError(f"brain.max_rounds: expected a positive integer, got {rounds}")
        values["max_rounds"] = rounds
    if "fallback_texts" in data:
        values["fallback_texts"] = _parse_fallback_texts(data["fallback_texts"])
    return BrainConfig(**values)


# --------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------


def read_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file, expand `${ENV_VAR}` references and return the top-level mapping."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in config file {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"config file top level must be a mapping: {path}")
    return expand_env(data)


def parse_config(data: dict[str, Any]) -> TalkoverConfig:
    """Validate an already-expanded top-level mapping into a `TalkoverConfig`."""
    _reject_unknown(data, TalkoverConfig, "")
    engine = _parse_engine(data.get("engine"))
    return TalkoverConfig(
        server=_parse_server(data.get("server")),
        engine=engine,
        asr=_parse_asr(data.get("asr"), engine),
        realtime=_parse_realtime(data.get("realtime")),
        brain=_parse_brain(data.get("brain")),
    )


def load_config(path: str | Path) -> TalkoverConfig:
    """Load and validate a serve config file (DESIGN.md section 9)."""
    return parse_config(read_yaml(path))


def load_brain_config(path: str | Path) -> BrainConfig:
    """Load a standalone Brain config file that only carries a `brain` section."""
    data = read_yaml(path)
    unknown = sorted(set(data) - {"brain"})
    if unknown:
        raise ConfigError(
            f"unknown config key(s) in brain config: {', '.join(unknown)}; only 'brain' is allowed"
        )
    return _parse_brain(data.get("brain"))
