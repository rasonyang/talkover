"""Realtime event models, validation and wire shapes.

This module owns the protocol surface described in `docs/protocol-profile.md`:
the GA OpenAI Realtime protocol (`session.type = "realtime"`, GA event names such
as `response.output_audio.delta`) as Talkover serves it. Beta shapes are rejected,
never translated.

Public API:

- `parse_client_event(raw)` validates one client frame and returns a `ClientEvent`,
  raising `ProtocolError` with the documented `code` / `message` / `param`.
- `error_event(exc)` renders a `ProtocolError` as the wire `error` event.
- `to_wire(event)` renders any client or server event as a JSON-ready dict.
- `new_event_id` / `new_item_id` / `new_response_id` / `new_session_id` /
  `new_conversation_id` / `new_call_id` mint OpenAI-style identifiers.

The module is pure: it performs no I/O, holds no session state, and never decides
whether an id exists in a conversation. Cross-event checks (unknown `item_id`,
duplicate `call_id`, "a response is already in progress") belong to the session
state machine, which raises `ProtocolError` with the same codes.
"""

from __future__ import annotations

import dataclasses
import json
import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

__all__ = [
    "AUDIO_FORMAT_NAMES",
    "ERROR_CODE_TYPES",
    "ClientEvent",
    "ConversationCreated",
    "ConversationItem",
    "ConversationItemCreate",
    "ConversationItemCreated",
    "ConversationItemDelete",
    "ConversationItemDeleted",
    "ConversationItemInputAudioTranscriptionCompleted",
    "ConversationItemTruncate",
    "ConversationItemTruncated",
    "ErrorEvent",
    "FunctionCallOutputItem",
    "InputAudioBufferAppend",
    "InputAudioBufferClear",
    "InputAudioBufferCleared",
    "InputAudioBufferCommit",
    "InputAudioBufferCommitted",
    "InputAudioBufferSpeechStarted",
    "InputAudioBufferSpeechStopped",
    "MessageItem",
    "ProtocolError",
    "ResponseCancel",
    "ResponseContentPartAdded",
    "ResponseContentPartDone",
    "ResponseCreate",
    "ResponseCreated",
    "ResponseDone",
    "ResponseFunctionCallArgumentsDelta",
    "ResponseFunctionCallArgumentsDone",
    "ResponseOptions",
    "ResponseOutputAudioDelta",
    "ResponseOutputAudioDone",
    "ResponseOutputAudioTranscriptDelta",
    "ResponseOutputAudioTranscriptDone",
    "ResponseOutputItemAdded",
    "ResponseOutputItemDone",
    "ServerEvent",
    "SessionConfig",
    "SessionCreated",
    "SessionUpdate",
    "SessionUpdated",
    "audio_format_name",
    "error_event",
    "estimate_tokens",
    "new_call_id",
    "new_conversation_id",
    "new_event_id",
    "new_item_id",
    "new_response_id",
    "new_session_id",
    "parse_client_event",
    "to_wire",
    "truncate_instructions",
]


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------

_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
_ID_SUFFIX_LEN = 16


def _mint(prefix: str) -> str:
    suffix = "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_SUFFIX_LEN))
    return f"{prefix}_{suffix}"


def new_event_id() -> str:
    """Mint a server event id (`event_…`)."""
    return _mint("event")


def new_item_id() -> str:
    """Mint a conversation item id (`item_…`)."""
    return _mint("item")


def new_response_id() -> str:
    """Mint a response id (`resp_…`)."""
    return _mint("resp")


def new_session_id() -> str:
    """Mint a session id (`sess_…`)."""
    return _mint("sess")


def new_conversation_id() -> str:
    """Mint a conversation id (`conv_…`)."""
    return _mint("conv")


def new_call_id() -> str:
    """Mint a tool-call linkage id (`call_…`)."""
    return _mint("call")


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------

#: `error.code` to `error.type`, as documented in the profile's rejection-code table.
ERROR_CODE_TYPES: dict[str, str] = {
    "invalid_event": "invalid_request_error",
    "invalid_value": "invalid_request_error",
    "engine_busy": "server_error",
    "engine_error": "server_error",
}


class ProtocolError(Exception):
    """A rejected client event or a runtime failure rendered as the `error` event."""

    def __init__(
        self,
        code: str,
        message: str,
        param: str | None = None,
        event_id: str | None = None,
    ) -> None:
        super().__init__(message)
        if code not in ERROR_CODE_TYPES:
            raise ValueError(f"unknown protocol error code: {code}")
        self.code = code
        self.message = message
        self.param = param
        self.event_id = event_id

    @property
    def type(self) -> str:
        """The `error.type` that goes with `code`."""
        return ERROR_CODE_TYPES[self.code]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"ProtocolError(code={self.code!r}, message={self.message!r}, "
            f"param={self.param!r}, event_id={self.event_id!r})"
        )


def error_event(exc: ProtocolError) -> dict[str, Any]:
    """Render a `ProtocolError` as the wire `error` event."""
    return ErrorEvent(
        error={
            "type": exc.type,
            "code": exc.code,
            "message": exc.message,
            "param": exc.param,
            "event_id": exc.event_id,
        }
    ).to_wire()


def _invalid_value(message: str, param: str | None, event_id: str | None) -> ProtocolError:
    return ProtocolError("invalid_value", message, param, event_id)


def _invalid_event(message: str, param: str | None, event_id: str | None) -> ProtocolError:
    return ProtocolError("invalid_event", message, param, event_id)


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------


def _as_dict(value: Any, param: str, event_id: str | None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid_value(f"Expected an object for {param!r}.", param, event_id)
    return dict(value)


def _as_str(value: Any, param: str, event_id: str | None) -> str:
    if not isinstance(value, str):
        raise _invalid_value(f"Expected a string for {param!r}.", param, event_id)
    return value


def _as_int(value: Any, param: str, event_id: str | None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _invalid_value(f"Expected an integer for {param!r}.", param, event_id)
    return value


def _as_number(value: Any, param: str, event_id: str | None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _invalid_value(f"Expected a number for {param!r}.", param, event_id)
    return float(value)


def _as_bool(value: Any, param: str, event_id: str | None) -> bool:
    if not isinstance(value, bool):
        raise _invalid_value(f"Expected a boolean for {param!r}.", param, event_id)
    return value


def _as_list(value: Any, param: str, event_id: str | None) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise _invalid_value(f"Expected an array for {param!r}.", param, event_id)
    return list(value)


def _reject_unknown(
    data: Mapping[str, Any],
    allowed: Sequence[str],
    prefix: str,
    event_id: str | None,
    rejected: Mapping[str, str] | None = None,
) -> None:
    """Reject unknown keys, and known-but-rejected keys with their own message."""
    for key in data:
        if rejected is not None and key in rejected:
            raise _invalid_value(rejected[key], f"{prefix}{key}", event_id)
        if key not in allowed:
            raise _invalid_value(f"Unknown parameter: {prefix}{key}.", f"{prefix}{key}", event_id)


# ---------------------------------------------------------------------------
# audio formats
# ---------------------------------------------------------------------------

#: GA `audio.*.format.type` to the format name understood by `talkover.realtime.audio`.
AUDIO_FORMAT_NAMES: dict[str, str] = {
    "audio/pcm": "pcm16",
    "audio/pcmu": "g711_ulaw",
    "audio/pcma": "g711_alaw",
}

_AUDIO_FORMAT_RATES: dict[str, int] = {
    "audio/pcm": 24000,
    "audio/pcmu": 8000,
    "audio/pcma": 8000,
}


def audio_format_name(spec: Mapping[str, Any] | None) -> str:
    """Map a validated GA format object to the `talkover.realtime.audio` name."""
    if spec is None:
        return "pcm16"
    return AUDIO_FORMAT_NAMES[str(spec["type"])]


def _validate_audio_format(value: Any, param: str, event_id: str | None) -> dict[str, Any]:
    data = _as_dict(value, param, event_id)
    _reject_unknown(data, ("type", "rate"), f"{param}.", event_id)
    kind = _as_str(data.get("type", "audio/pcm"), f"{param}.type", event_id)
    if kind not in AUDIO_FORMAT_NAMES:
        raise _invalid_value(
            f"Unsupported audio format {kind!r}; expected one of "
            f"{', '.join(sorted(AUDIO_FORMAT_NAMES))}.",
            f"{param}.type",
            event_id,
        )
    expected_rate = _AUDIO_FORMAT_RATES[kind]
    rate = _as_int(data.get("rate", expected_rate), f"{param}.rate", event_id)
    if rate != expected_rate:
        raise _invalid_value(
            f"Audio format {kind!r} requires rate {expected_rate}.",
            f"{param}.rate",
            event_id,
        )
    return {"type": kind, "rate": rate}


# ---------------------------------------------------------------------------
# instructions cap (profile §4, DESIGN.md 5.5)
# ---------------------------------------------------------------------------

_CJK_RANGES = (
    (0x3000, 0x303F),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0xFF00, 0xFFEF),
)


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """Approximate the token count of `text` without loading a tokenizer.

    One token per CJK character, one token per four other characters. The task
    slate cap only needs an upper-bounded estimate, and the engine tokenizer is
    not available to the protocol layer.
    """
    cjk = sum(1 for char in text if _is_cjk(char))
    other = len(text) - cjk
    return cjk + math.ceil(other / 4)


def truncate_instructions(text: str, max_tokens: int) -> str:
    """Truncate `text` so that `estimate_tokens` stays within `max_tokens`.

    Truncation is by characters; the returned value is what `session.updated`
    echoes and what is written to the task slate.
    """
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if estimate_tokens(text[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid - 1
    return text[:low]


# ---------------------------------------------------------------------------
# event base classes
# ---------------------------------------------------------------------------


def _unwrap(value: Any) -> Any:
    raw = getattr(value, "raw", None)
    if raw is not None and isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(value, tuple):
        return [_unwrap(item) for item in value]
    return value


@dataclass(frozen=True)
class ClientEvent:
    """Base class for validated client events."""

    TYPE: ClassVar[str]

    event_id: str | None = field(default=None, kw_only=True)

    def to_wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {"type": self.TYPE}
        if self.event_id is not None:
            data["event_id"] = self.event_id
        for f in dataclasses.fields(self):
            if f.name == "event_id":
                continue
            value = getattr(self, f.name)
            if value is None and f.default is None:
                continue
            data[f.name] = _unwrap(value)
        return data


@dataclass(frozen=True)
class ServerEvent:
    """Base class for server events Talkover emits."""

    TYPE: ClassVar[str]

    event_id: str = field(default_factory=new_event_id, kw_only=True)

    def to_wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {"event_id": self.event_id, "type": self.TYPE}
        for f in dataclasses.fields(self):
            if f.name == "event_id":
                continue
            data[f.name] = _unwrap(getattr(self, f.name))
        return data


def to_wire(event: ClientEvent | ServerEvent) -> dict[str, Any]:
    """Render a client or server event as a JSON-ready dict."""
    return event.to_wire()


# ---------------------------------------------------------------------------
# session object
# ---------------------------------------------------------------------------

_SESSION_FIELDS = (
    "type",
    "model",
    "instructions",
    "output_modalities",
    "audio",
    "tools",
    "tool_choice",
    "max_output_tokens",
    "truncation",
    "tracing",
    "prompt",
    "include",
)

_BETA_SESSION_FIELDS = (
    "modalities",
    "voice",
    "turn_detection",
    "input_audio_format",
    "output_audio_format",
    "input_audio_transcription",
    "temperature",
    "max_response_output_tokens",
)

_TURN_DETECTION_COMMON = ("type", "create_response", "interrupt_response", "idle_timeout_ms")
_SERVER_VAD_FIELDS = ("threshold", "prefix_padding_ms", "silence_duration_ms")
_SEMANTIC_VAD_FIELDS = ("eagerness",)
_EAGERNESS = ("low", "medium", "high", "auto")


@dataclass(frozen=True)
class SessionConfig:
    """A validated `session` object; `raw` is the normalized object as sent."""

    raw: Mapping[str, Any]

    def _audio(self, direction: str, key: str) -> Any:
        audio = self.raw.get("audio")
        if not isinstance(audio, Mapping):
            return None
        side = audio.get(direction)
        if not isinstance(side, Mapping):
            return None
        return side.get(key)

    @property
    def model(self) -> str | None:
        value = self.raw.get("model")
        return value if isinstance(value, str) else None

    @property
    def instructions(self) -> str | None:
        value = self.raw.get("instructions")
        return value if isinstance(value, str) else None

    @property
    def output_modalities(self) -> tuple[str, ...] | None:
        value = self.raw.get("output_modalities")
        return tuple(value) if isinstance(value, list) else None

    @property
    def input_audio_format(self) -> Mapping[str, Any] | None:
        return self._audio("input", "format")

    @property
    def output_audio_format(self) -> Mapping[str, Any] | None:
        return self._audio("output", "format")

    @property
    def voice(self) -> str | None:
        value = self._audio("output", "voice")
        return value if isinstance(value, str) else None

    @property
    def speed(self) -> float | None:
        value = self._audio("output", "speed")
        return float(value) if isinstance(value, (int, float)) else None

    @property
    def transcription(self) -> Mapping[str, Any] | None:
        return self._audio("input", "transcription")

    @property
    def noise_reduction(self) -> Mapping[str, Any] | None:
        return self._audio("input", "noise_reduction")

    @property
    def turn_detection(self) -> Mapping[str, Any] | None:
        return self._audio("input", "turn_detection")

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        value = self.raw.get("tools")
        return tuple(value) if isinstance(value, list) else ()

    @property
    def tool_choice(self) -> Any:
        return self.raw.get("tool_choice")

    def has(self, name: str) -> bool:
        """Whether the client sent a top-level session field in this update."""
        return name in self.raw


def _validate_turn_detection(value: Any, param: str, event_id: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _as_dict(value, param, event_id)
    kind = _as_str(data.get("type", "server_vad"), f"{param}.type", event_id)
    if kind == "server_vad":
        allowed = _TURN_DETECTION_COMMON + _SERVER_VAD_FIELDS
        foreign = _SEMANTIC_VAD_FIELDS
    elif kind == "semantic_vad":
        allowed = _TURN_DETECTION_COMMON + _SEMANTIC_VAD_FIELDS
        foreign = _SERVER_VAD_FIELDS
    else:
        raise _invalid_value(
            f"Unsupported turn detection type {kind!r}; expected 'server_vad' or 'semantic_vad'.",
            f"{param}.type",
            event_id,
        )
    for key in foreign:
        if key in data:
            raise _invalid_value(
                f"{key!r} does not apply to turn detection type {kind!r}.",
                f"{param}.{key}",
                event_id,
            )
    _reject_unknown(data, allowed, f"{param}.", event_id)
    out: dict[str, Any] = {"type": kind}
    if "threshold" in data:
        threshold = _as_number(data["threshold"], f"{param}.threshold", event_id)
        if not 0.0 <= threshold <= 1.0:
            raise _invalid_value("threshold must be in [0, 1].", f"{param}.threshold", event_id)
        out["threshold"] = threshold
    if "prefix_padding_ms" in data:
        padding = _as_int(data["prefix_padding_ms"], f"{param}.prefix_padding_ms", event_id)
        if padding < 0:
            raise _invalid_value(
                "prefix_padding_ms must be >= 0.", f"{param}.prefix_padding_ms", event_id
            )
        out["prefix_padding_ms"] = padding
    if "silence_duration_ms" in data:
        silence = _as_int(data["silence_duration_ms"], f"{param}.silence_duration_ms", event_id)
        if silence <= 0:
            raise _invalid_value(
                "silence_duration_ms must be > 0.", f"{param}.silence_duration_ms", event_id
            )
        out["silence_duration_ms"] = silence
    if "eagerness" in data:
        eagerness = _as_str(data["eagerness"], f"{param}.eagerness", event_id)
        if eagerness not in _EAGERNESS:
            raise _invalid_value(
                f"eagerness must be one of {', '.join(_EAGERNESS)}.",
                f"{param}.eagerness",
                event_id,
            )
        out["eagerness"] = eagerness
    for key in ("create_response", "interrupt_response"):
        if key in data:
            out[key] = _as_bool(data[key], f"{param}.{key}", event_id)
    if data.get("idle_timeout_ms") is not None:
        raise _invalid_value(
            "idle_timeout_ms is not supported; only null is accepted.",
            f"{param}.idle_timeout_ms",
            event_id,
        )
    if "idle_timeout_ms" in data:
        out["idle_timeout_ms"] = None
    return out


def _validate_transcription(value: Any, param: str, event_id: str | None) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _as_dict(value, param, event_id)
    _reject_unknown(data, ("model", "language", "prompt"), f"{param}.", event_id)
    out: dict[str, Any] = {}
    for key in ("model", "language", "prompt"):
        if key in data:
            out[key] = _as_str(data[key], f"{param}.{key}", event_id)
    return out


def _validate_noise_reduction(
    value: Any, param: str, event_id: str | None
) -> dict[str, Any] | None:
    if value is None:
        return None
    data = _as_dict(value, param, event_id)
    _reject_unknown(data, ("type",), f"{param}.", event_id)
    kind = _as_str(data.get("type", "near_field"), f"{param}.type", event_id)
    if kind not in ("near_field", "far_field"):
        raise _invalid_value(
            "noise_reduction type must be 'near_field' or 'far_field'.",
            f"{param}.type",
            event_id,
        )
    return {"type": kind}


def _validate_tools(value: Any, param: str, event_id: str | None) -> list[dict[str, Any]]:
    entries = _as_list(value, param, event_id)
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        base = f"{param}[{index}]"
        data = _as_dict(entry, base, event_id)
        _reject_unknown(data, ("type", "name", "description", "parameters"), f"{base}.", event_id)
        kind = data.get("type")
        if kind != "function":
            raise _invalid_value(
                "Only tools of type 'function' are supported.", f"{base}.type", event_id
            )
        if "name" not in data:
            raise _invalid_value("A tool entry requires 'name'.", f"{base}.name", event_id)
        name = _as_str(data["name"], f"{base}.name", event_id)
        if name in names:
            raise _invalid_value(f"Duplicate tool name {name!r}.", f"{base}.name", event_id)
        names.add(name)
        tool: dict[str, Any] = {"type": "function", "name": name}
        if "description" in data:
            tool["description"] = _as_str(data["description"], f"{base}.description", event_id)
        if "parameters" in data:
            tool["parameters"] = _as_dict(data["parameters"], f"{base}.parameters", event_id)
        out.append(tool)
    return out


def _validate_tool_choice(value: Any, param: str, names: set[str], event_id: str | None) -> Any:
    if isinstance(value, str):
        if value not in ("none", "auto", "required"):
            raise _invalid_value(
                "tool_choice must be 'none', 'auto', 'required' or a function object.",
                param,
                event_id,
            )
        return value
    data = _as_dict(value, param, event_id)
    _reject_unknown(data, ("type", "name"), f"{param}.", event_id)
    if data.get("type") != "function":
        raise _invalid_value(
            "tool_choice objects must have type 'function'.", f"{param}.type", event_id
        )
    name = _as_str(data.get("name", ""), f"{param}.name", event_id)
    if name not in names:
        raise _invalid_value(
            f"tool_choice names {name!r}, which is not declared in tools.",
            f"{param}.name",
            event_id,
        )
    return {"type": "function", "name": name}


def _validate_audio_section(value: Any, param: str, event_id: str | None) -> dict[str, Any]:
    data = _as_dict(value, param, event_id)
    _reject_unknown(data, ("input", "output"), f"{param}.", event_id)
    out: dict[str, Any] = {}
    if "input" in data:
        side = _as_dict(data["input"], f"{param}.input", event_id)
        _reject_unknown(
            side,
            ("format", "transcription", "noise_reduction", "turn_detection"),
            f"{param}.input.",
            event_id,
        )
        block: dict[str, Any] = {}
        if "format" in side:
            block["format"] = _validate_audio_format(
                side["format"], f"{param}.input.format", event_id
            )
        if "transcription" in side:
            block["transcription"] = _validate_transcription(
                side["transcription"], f"{param}.input.transcription", event_id
            )
        if "noise_reduction" in side:
            block["noise_reduction"] = _validate_noise_reduction(
                side["noise_reduction"], f"{param}.input.noise_reduction", event_id
            )
        if "turn_detection" in side:
            block["turn_detection"] = _validate_turn_detection(
                side["turn_detection"], f"{param}.input.turn_detection", event_id
            )
        out["input"] = block
    if "output" in data:
        side = _as_dict(data["output"], f"{param}.output", event_id)
        _reject_unknown(side, ("format", "voice", "speed"), f"{param}.output.", event_id)
        block = {}
        if "format" in side:
            block["format"] = _validate_audio_format(
                side["format"], f"{param}.output.format", event_id
            )
        if "voice" in side:
            block["voice"] = _as_str(side["voice"], f"{param}.output.voice", event_id)
        if "speed" in side:
            speed = _as_number(side["speed"], f"{param}.output.speed", event_id)
            if not 0.25 <= speed <= 1.5:
                raise _invalid_value(
                    "speed must be in [0.25, 1.5].", f"{param}.output.speed", event_id
                )
            block["speed"] = speed
        out["output"] = block
    return out


def _validate_session(value: Any, event_id: str | None) -> SessionConfig:
    data = _as_dict(value, "session", event_id)
    for name in _BETA_SESSION_FIELDS:
        if name in data:
            raise _invalid_value(
                f"Unknown parameter: session.{name} (beta shape; the GA session shape "
                "is required).",
                f"session.{name}",
                event_id,
            )
    _reject_unknown(data, _SESSION_FIELDS, "session.", event_id)
    if "type" not in data:
        raise _invalid_value(
            "session.type is required and must be 'realtime'.", "session.type", event_id
        )
    if data["type"] != "realtime":
        raise _invalid_value(
            "session.type must be 'realtime'; the beta protocol is not served.",
            "session.type",
            event_id,
        )
    out: dict[str, Any] = {"type": "realtime"}
    if "model" in data:
        out["model"] = _as_str(data["model"], "session.model", event_id)
    if "instructions" in data:
        out["instructions"] = _as_str(data["instructions"], "session.instructions", event_id)
    if "output_modalities" in data:
        modalities = _as_list(data["output_modalities"], "session.output_modalities", event_id)
        if modalities != ["audio"]:
            raise _invalid_value(
                "output_modalities must be ['audio']; Talkover has no text-only response path.",
                "session.output_modalities",
                event_id,
            )
        out["output_modalities"] = ["audio"]
    if "audio" in data:
        out["audio"] = _validate_audio_section(data["audio"], "session.audio", event_id)
    names: set[str] = set()
    if "tools" in data:
        tools = _validate_tools(data["tools"], "session.tools", event_id)
        names = {tool["name"] for tool in tools}
        out["tools"] = tools
    if "tool_choice" in data:
        out["tool_choice"] = _validate_tool_choice(
            data["tool_choice"], "session.tool_choice", names, event_id
        )
    if "max_output_tokens" in data:
        value_tokens = data["max_output_tokens"]
        if value_tokens != "inf":
            tokens = _as_int(value_tokens, "session.max_output_tokens", event_id)
            if tokens < 1:
                raise _invalid_value(
                    "max_output_tokens must be 'inf' or a positive integer.",
                    "session.max_output_tokens",
                    event_id,
                )
            value_tokens = tokens
        out["max_output_tokens"] = value_tokens
    for name, accepted in (("truncation", "auto"), ("tracing", None), ("prompt", None)):
        if name in data:
            if data[name] != accepted:
                raise _invalid_value(
                    f"session.{name} only accepts {accepted!r}.", f"session.{name}", event_id
                )
            out[name] = accepted
    if "include" in data:
        include = data["include"]
        if include not in (None, []):
            raise _invalid_value(
                "session.include only accepts null or [].", "session.include", event_id
            )
        out["include"] = include
    return SessionConfig(raw=out)


# ---------------------------------------------------------------------------
# conversation items
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MessageItem:
    """A client-created text message item."""

    raw: Mapping[str, Any]
    role: str
    text: str
    id: str | None = None


@dataclass(frozen=True)
class FunctionCallOutputItem:
    """A client-created `function_call_output` item."""

    raw: Mapping[str, Any]
    call_id: str
    output: str
    id: str | None = None


ConversationItem = MessageItem | FunctionCallOutputItem

_REJECTED_ITEM_TYPES = (
    "function_call",
    "item_reference",
    "mcp_call",
    "mcp_list_tools",
    "mcp_approval_request",
    "mcp_approval_response",
)


def _validate_item(value: Any, event_id: str | None) -> ConversationItem:
    data = _as_dict(value, "item", event_id)
    kind = data.get("type")
    if not isinstance(kind, str):
        raise _invalid_value("item.type is required.", "item.type", event_id)
    item_id = None
    if "id" in data:
        item_id = _as_str(data["id"], "item.id", event_id)
    if kind == "message":
        _reject_unknown(
            data, ("id", "type", "role", "content", "status", "object"), "item.", event_id
        )
        role = _as_str(data.get("role", ""), "item.role", event_id)
        if role != "user":
            raise _invalid_value(
                "Only user message items can be created; Talkover cannot inject system or "
                "assistant turns into the model context.",
                "item.role",
                event_id,
            )
        content = _as_list(data.get("content", []), "item.content", event_id)
        if len(content) != 1:
            raise _invalid_value(
                "Exactly one content part is supported per message item.", "item.content", event_id
            )
        part = _as_dict(content[0], "item.content[0]", event_id)
        if part.get("type") != "input_text":
            raise _invalid_value(
                "Only 'input_text' content is supported; audio enters through the input "
                "audio buffer.",
                "item.content[0].type",
                event_id,
            )
        _reject_unknown(part, ("type", "text"), "item.content[0].", event_id)
        text = _as_str(part.get("text", ""), "item.content[0].text", event_id)
        raw: dict[str, Any] = {
            "type": "message",
            "role": role,
            "content": [{"type": "input_text", "text": text}],
        }
        if item_id is not None:
            raw["id"] = item_id
        return MessageItem(raw=raw, role=role, text=text, id=item_id)
    if kind == "function_call_output":
        _reject_unknown(
            data, ("id", "type", "call_id", "output", "status", "object"), "item.", event_id
        )
        if "call_id" not in data:
            raise _invalid_value(
                "function_call_output requires 'call_id'.", "item.call_id", event_id
            )
        call_id = _as_str(data["call_id"], "item.call_id", event_id)
        if "output" not in data:
            raise _invalid_value("function_call_output requires 'output'.", "item.output", event_id)
        output = _as_str(data["output"], "item.output", event_id)
        raw = {"type": "function_call_output", "call_id": call_id, "output": output}
        if item_id is not None:
            raw["id"] = item_id
        return FunctionCallOutputItem(raw=raw, call_id=call_id, output=output, id=item_id)
    if kind in _REJECTED_ITEM_TYPES:
        raise _invalid_value(
            f"Item type {kind!r} cannot be created by a client.", "item.type", event_id
        )
    raise _invalid_value(f"Unknown item type {kind!r}.", "item.type", event_id)


# ---------------------------------------------------------------------------
# client events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionUpdate(ClientEvent):
    TYPE: ClassVar[str] = "session.update"

    session: SessionConfig


@dataclass(frozen=True)
class InputAudioBufferAppend(ClientEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.append"

    audio: str


@dataclass(frozen=True)
class InputAudioBufferCommit(ClientEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.commit"


@dataclass(frozen=True)
class InputAudioBufferClear(ClientEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.clear"


@dataclass(frozen=True)
class ResponseOptions:
    """The accepted subset of `response.create.response`."""

    raw: Mapping[str, Any]

    @property
    def metadata(self) -> Mapping[str, str] | None:
        value = self.raw.get("metadata")
        return value if isinstance(value, Mapping) else None


@dataclass(frozen=True)
class ResponseCreate(ClientEvent):
    TYPE: ClassVar[str] = "response.create"

    response: ResponseOptions | None = None


@dataclass(frozen=True)
class ResponseCancel(ClientEvent):
    TYPE: ClassVar[str] = "response.cancel"

    response_id: str | None = None


@dataclass(frozen=True)
class ConversationItemCreate(ClientEvent):
    TYPE: ClassVar[str] = "conversation.item.create"

    item: ConversationItem
    previous_item_id: str | None = None


@dataclass(frozen=True)
class ConversationItemTruncate(ClientEvent):
    TYPE: ClassVar[str] = "conversation.item.truncate"

    item_id: str
    content_index: int
    audio_end_ms: int


@dataclass(frozen=True)
class ConversationItemDelete(ClientEvent):
    TYPE: ClassVar[str] = "conversation.item.delete"

    item_id: str


_RESPONSE_CREATE_FIELDS = ("conversation", "metadata")
_RESPONSE_CREATE_REJECTED = {
    "instructions": (
        "Per-response instructions are not supported; session.instructions is written to the "
        "task slate instead."
    ),
    "output_modalities": "Per-response output_modalities are not supported.",
    "max_output_tokens": "Per-response max_output_tokens is not supported.",
    "audio": "Per-response audio overrides are not supported.",
    "input": "Out-of-band response input is not supported.",
    "tools": "Tools are session-level only.",
    "tool_choice": "Tools are session-level only.",
    "parallel_tool_calls": "Tools are session-level only.",
    "prompt": "prompt is not supported.",
    "reasoning": "reasoning is not supported.",
}


def _validate_response_options(value: Any, event_id: str | None) -> ResponseOptions:
    data = _as_dict(value, "response", event_id)
    _reject_unknown(
        data, _RESPONSE_CREATE_FIELDS, "response.", event_id, rejected=_RESPONSE_CREATE_REJECTED
    )
    out: dict[str, Any] = {}
    if "conversation" in data:
        if data["conversation"] != "auto":
            raise _invalid_value(
                "response.conversation only accepts 'auto'; out-of-band responses are not "
                "supported.",
                "response.conversation",
                event_id,
            )
        out["conversation"] = "auto"
    if "metadata" in data:
        metadata = _as_dict(data["metadata"], "response.metadata", event_id)
        if len(metadata) > 16:
            raise _invalid_value(
                "metadata accepts at most 16 entries.", "response.metadata", event_id
            )
        for key, item in metadata.items():
            _as_str(item, f"response.metadata.{key}", event_id)
        out["metadata"] = dict(metadata)
    return ResponseOptions(raw=out)


_REJECTED_CLIENT_TYPES = {
    "conversation.item.retrieve": (
        "conversation.item.retrieve is not supported; Talkover does not retain input audio."
    ),
    "output_audio_buffer.clear": (
        "output_audio_buffer.clear applies to WebRTC and SIP sessions only."
    ),
    "transcription_session.update": "Transcription sessions are not served.",
    "response.audio.delta": "Beta event names are not accepted; use the GA protocol.",
    "session.create": "session.create is not a client event.",
}


def _decode(raw: str | bytes | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, (bytes, bytearray)):
        try:
            raw = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _invalid_event("Invalid UTF-8 in the client frame.", None, None) from exc
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise _invalid_event("Invalid JSON in the client frame.", None, None) from exc
    if not isinstance(parsed, Mapping):
        raise _invalid_event("A client event must be a JSON object.", None, None)
    return dict(parsed)


def parse_client_event(raw: str | bytes | Mapping[str, Any]) -> ClientEvent:
    """Validate one client frame and return the matching `ClientEvent`.

    Raises `ProtocolError` with the code, message and `param` documented in
    `docs/protocol-profile.md`.
    """
    data = _decode(raw)

    event_id: str | None = None
    if "event_id" in data and data["event_id"] is not None:
        if not isinstance(data["event_id"], str):
            raise _invalid_value("event_id must be a string.", "event_id", None)
        if len(data["event_id"]) > 512:
            raise _invalid_value("event_id exceeds 512 characters.", "event_id", None)
        event_id = data["event_id"]

    if "type" not in data:
        raise _invalid_event("The 'type' field is missing.", None, event_id)
    kind = data["type"]
    if not isinstance(kind, str):
        raise _invalid_value("The 'type' field must be a string.", "type", event_id)
    if kind in _REJECTED_CLIENT_TYPES:
        raise _invalid_event(_REJECTED_CLIENT_TYPES[kind], "type", event_id)

    if kind == "session.update":
        if "session" not in data:
            raise _invalid_value("session.update requires 'session'.", "session", event_id)
        _reject_unknown(data, ("type", "event_id", "session"), "", event_id)
        return SessionUpdate(
            session=_validate_session(data["session"], event_id), event_id=event_id
        )

    if kind == "input_audio_buffer.append":
        _reject_unknown(data, ("type", "event_id", "audio"), "", event_id)
        if "audio" not in data:
            raise _invalid_value("input_audio_buffer.append requires 'audio'.", "audio", event_id)
        return InputAudioBufferAppend(
            audio=_as_str(data["audio"], "audio", event_id), event_id=event_id
        )

    if kind == "input_audio_buffer.commit":
        _reject_unknown(data, ("type", "event_id"), "", event_id)
        return InputAudioBufferCommit(event_id=event_id)

    if kind == "input_audio_buffer.clear":
        _reject_unknown(data, ("type", "event_id"), "", event_id)
        return InputAudioBufferClear(event_id=event_id)

    if kind == "response.create":
        _reject_unknown(data, ("type", "event_id", "response"), "", event_id)
        options = None
        if data.get("response") is not None:
            options = _validate_response_options(data["response"], event_id)
        return ResponseCreate(response=options, event_id=event_id)

    if kind == "response.cancel":
        _reject_unknown(data, ("type", "event_id", "response_id"), "", event_id)
        response_id = None
        if data.get("response_id") is not None:
            response_id = _as_str(data["response_id"], "response_id", event_id)
        return ResponseCancel(response_id=response_id, event_id=event_id)

    if kind == "conversation.item.create":
        _reject_unknown(data, ("type", "event_id", "item", "previous_item_id"), "", event_id)
        if "item" not in data:
            raise _invalid_value("conversation.item.create requires 'item'.", "item", event_id)
        previous = None
        if data.get("previous_item_id") is not None:
            previous = _as_str(data["previous_item_id"], "previous_item_id", event_id)
        return ConversationItemCreate(
            item=_validate_item(data["item"], event_id),
            previous_item_id=previous,
            event_id=event_id,
        )

    if kind == "conversation.item.truncate":
        _reject_unknown(
            data, ("type", "event_id", "item_id", "content_index", "audio_end_ms"), "", event_id
        )
        for name in ("item_id", "content_index", "audio_end_ms"):
            if name not in data:
                raise _invalid_value(
                    f"conversation.item.truncate requires {name!r}.", name, event_id
                )
        content_index = _as_int(data["content_index"], "content_index", event_id)
        if content_index != 0:
            raise _invalid_value("content_index must be 0.", "content_index", event_id)
        audio_end_ms = _as_int(data["audio_end_ms"], "audio_end_ms", event_id)
        if audio_end_ms < 0:
            raise _invalid_value("audio_end_ms must be >= 0.", "audio_end_ms", event_id)
        return ConversationItemTruncate(
            item_id=_as_str(data["item_id"], "item_id", event_id),
            content_index=content_index,
            audio_end_ms=audio_end_ms,
            event_id=event_id,
        )

    if kind == "conversation.item.delete":
        _reject_unknown(data, ("type", "event_id", "item_id"), "", event_id)
        if "item_id" not in data:
            raise _invalid_value(
                "conversation.item.delete requires 'item_id'.", "item_id", event_id
            )
        return ConversationItemDelete(
            item_id=_as_str(data["item_id"], "item_id", event_id), event_id=event_id
        )

    raise _invalid_value(f"Unknown client event type {kind!r}.", "type", event_id)


# ---------------------------------------------------------------------------
# server events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionCreated(ServerEvent):
    TYPE: ClassVar[str] = "session.created"

    session: Mapping[str, Any]


@dataclass(frozen=True)
class SessionUpdated(ServerEvent):
    TYPE: ClassVar[str] = "session.updated"

    session: Mapping[str, Any]


@dataclass(frozen=True)
class ConversationCreated(ServerEvent):
    TYPE: ClassVar[str] = "conversation.created"

    conversation: Mapping[str, Any]


@dataclass(frozen=True)
class InputAudioBufferCommitted(ServerEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.committed"

    item_id: str
    previous_item_id: str | None = None


@dataclass(frozen=True)
class InputAudioBufferCleared(ServerEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.cleared"


@dataclass(frozen=True)
class InputAudioBufferSpeechStarted(ServerEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.speech_started"

    audio_start_ms: int
    item_id: str


@dataclass(frozen=True)
class InputAudioBufferSpeechStopped(ServerEvent):
    TYPE: ClassVar[str] = "input_audio_buffer.speech_stopped"

    audio_end_ms: int
    item_id: str


@dataclass(frozen=True)
class ConversationItemCreated(ServerEvent):
    TYPE: ClassVar[str] = "conversation.item.created"

    item: Mapping[str, Any]
    previous_item_id: str | None = None


@dataclass(frozen=True)
class ConversationItemTruncated(ServerEvent):
    TYPE: ClassVar[str] = "conversation.item.truncated"

    item_id: str
    content_index: int
    audio_end_ms: int


@dataclass(frozen=True)
class ConversationItemDeleted(ServerEvent):
    TYPE: ClassVar[str] = "conversation.item.deleted"

    item_id: str


@dataclass(frozen=True)
class ConversationItemInputAudioTranscriptionCompleted(ServerEvent):
    TYPE: ClassVar[str] = "conversation.item.input_audio_transcription.completed"

    item_id: str
    content_index: int
    transcript: str
    usage: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseCreated(ServerEvent):
    TYPE: ClassVar[str] = "response.created"

    response: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseOutputItemAdded(ServerEvent):
    TYPE: ClassVar[str] = "response.output_item.added"

    response_id: str
    output_index: int
    item: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseOutputItemDone(ServerEvent):
    TYPE: ClassVar[str] = "response.output_item.done"

    response_id: str
    output_index: int
    item: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseContentPartAdded(ServerEvent):
    TYPE: ClassVar[str] = "response.content_part.added"

    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseContentPartDone(ServerEvent):
    TYPE: ClassVar[str] = "response.content_part.done"

    response_id: str
    item_id: str
    output_index: int
    content_index: int
    part: Mapping[str, Any]


@dataclass(frozen=True)
class ResponseOutputAudioDelta(ServerEvent):
    TYPE: ClassVar[str] = "response.output_audio.delta"

    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass(frozen=True)
class ResponseOutputAudioDone(ServerEvent):
    TYPE: ClassVar[str] = "response.output_audio.done"

    response_id: str
    item_id: str
    output_index: int
    content_index: int


@dataclass(frozen=True)
class ResponseOutputAudioTranscriptDelta(ServerEvent):
    TYPE: ClassVar[str] = "response.output_audio_transcript.delta"

    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str


@dataclass(frozen=True)
class ResponseOutputAudioTranscriptDone(ServerEvent):
    TYPE: ClassVar[str] = "response.output_audio_transcript.done"

    response_id: str
    item_id: str
    output_index: int
    content_index: int
    transcript: str


@dataclass(frozen=True)
class ResponseFunctionCallArgumentsDelta(ServerEvent):
    TYPE: ClassVar[str] = "response.function_call_arguments.delta"

    response_id: str
    item_id: str
    output_index: int
    call_id: str
    delta: str


@dataclass(frozen=True)
class ResponseFunctionCallArgumentsDone(ServerEvent):
    TYPE: ClassVar[str] = "response.function_call_arguments.done"

    response_id: str
    item_id: str
    output_index: int
    call_id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ResponseDone(ServerEvent):
    TYPE: ClassVar[str] = "response.done"

    response: Mapping[str, Any]


@dataclass(frozen=True)
class ErrorEvent(ServerEvent):
    TYPE: ClassVar[str] = "error"

    error: Mapping[str, Any]
