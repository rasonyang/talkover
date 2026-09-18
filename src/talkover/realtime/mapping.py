"""Mapping from `EngineStepEvent` (upstream `DuplexStepEvent`) to Realtime server events.

This is the right column of the event mapping table in DESIGN.md 5.3, and the response
half of `docs/protocol-profile.md` section 9: the model produces one causal unit per
second and this module turns that stream into the `response.*` lifecycle a Realtime
client expects.

The rules, in the order a turn exercises them:

- The first unit with `is_listen: false` opens a response: `response.created`,
  `response.output_item.added`, `conversation.item.created` (the assistant message item)
  and `response.content_part.added` with the empty audio part.
- `EngineStepEvent.text` becomes `response.output_audio_transcript.delta`.
- `EngineStepEvent.audio_waveform` becomes `response.output_audio.delta`, base64 in the
  session's negotiated output format (pcm16 at 24 kHz, or g711 at 8 kHz), encoded by
  `talkover.realtime.audio.encode_output_audio` through one resampler per response so
  the chunk boundaries stay seamless.
- `end_of_turn: true` closes the response: `response.output_audio.done`,
  `response.output_audio_transcript.done`, `response.content_part.done`,
  `response.output_item.done` and `response.done` with `status: "completed"`.
- `interrupted: true` closes the same chain with `status: "cancelled"`, an `incomplete`
  item, and `conversation.item.truncated` at the audio already sent.
- An ASR segment becomes `conversation.item.input_audio_transcription.completed` against
  the last committed input item (`RealtimeSession.input_item_id`).
- `ResponseMapper.emit_function_call` is the T3.7 seam: the Brain's `transfer_to_human`
  becomes one complete `response.function_call_arguments.delta` and its `.done`.

`ResponseMapper` owns the response bookkeeping that `RealtimeSession` only reads
(`active_response_id`, `pending_response_metadata`, `register_item`), and writes every
event through `RealtimeSession.emit`, so it never touches the WebSocket.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from talkover.engine.protocol import OUTPUT_SAMPLE_RATE, EngineStepEvent
from talkover.realtime import events as ev
from talkover.realtime.audio import (
    CLIENT_SAMPLE_RATE,
    G711_SAMPLE_RATE,
    StreamResampler,
    encode_output_audio,
    float_to_pcm16,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from talkover.engine.asr import Transcript
    from talkover.realtime.session import RealtimeSession

__all__ = [
    "CONTENT_INDEX",
    "MESSAGE_OUTPUT_INDEX",
    "ActiveResponse",
    "ResponseMapper",
]

#: `content_index` is always 0: one audio part per assistant message (profile §9).
CONTENT_INDEX = 0
#: `output_index` of the assistant message item; a `function_call` follows it (profile §9).
MESSAGE_OUTPUT_INDEX = 0

#: Output formats that are g711 and therefore need the 24 kHz -> 8 kHz resampler.
_G711_FORMATS = frozenset({"g711_ulaw", "g711_alaw", "audio/pcmu", "audio/pcma"})


@dataclass
class ActiveResponse:
    """The response currently streaming to the client."""

    #: `resp_…`, mirrored in `RealtimeSession.active_response_id`.
    id: str
    #: `item_id` of the assistant message item, or `None` for a tool-call-only response.
    item_id: str | None
    #: `response.create.response.metadata`, echoed on the response object.
    metadata: Mapping[str, str] | None = None
    #: Everything `response.output_audio_transcript.delta` has carried so far.
    transcript: str = ""
    #: Assistant audio already sent, in milliseconds at the client rate.
    audio_ms: float = 0.0
    #: Finished `function_call` items, in `response.done.output` order.
    tool_items: list[dict[str, Any]] = field(default_factory=list)
    #: 24 kHz -> 8 kHz, held for the whole response so g711 chunks stay seamless.
    g711_resampler: StreamResampler | None = None
    #: Engine rate -> 24 kHz, only built when the engine does not emit at 24 kHz.
    client_resampler: StreamResampler | None = None

    def next_output_index(self) -> int:
        """`output_index` for the next `function_call` item of this response."""
        base = 1 if self.item_id is not None else 0
        return base + len(self.tool_items)


class ResponseMapper:
    """Turn the engine's unit stream into the `response.*` events of profile §9."""

    def __init__(self, session: RealtimeSession) -> None:
        self.session = session
        #: The streaming response, or `None` while the model is listening.
        self.active: ActiveResponse | None = None

    # -- engine units ------------------------------------------------------

    async def on_engine_event(self, step: EngineStepEvent) -> None:
        """Map one model unit. This is what the session's engine-event pump calls."""
        if step.is_audio_chunk:
            # A detached-Talker audio chunk carries nothing but the waveform of a unit
            # already reported, so it may only extend the response that unit opened.
            if self.active is not None and step.has_audio:
                await self._emit_audio(self.active, step)
            return
        if step.is_tool_call and self.active is None:
            # A native tool-call unit is the Cerebellum talking to the runtime, not to the
            # customer: it carries no speech, so it opens no response. What the Brain hands
            # back becomes a `function_call` through `emit_function_call` (T3.7). A unit
            # that arrives while a response is streaming still closes it below.
            return
        if not step.is_listen and self.active is None:
            await self.open_response()
        active = self.active
        if active is None:
            # A listening unit outside a response carries nothing a client can see;
            # `end_of_turn` / `interrupted` without an open response are likewise moot.
            return
        if step.text:
            active.transcript += step.text
            await self.session.emit(
                ev.ResponseOutputAudioTranscriptDelta(
                    response_id=active.id,
                    item_id=self._message_item_id(active),
                    output_index=MESSAGE_OUTPUT_INDEX,
                    content_index=CONTENT_INDEX,
                    delta=step.text,
                )
            )
        if step.has_audio:
            await self._emit_audio(active, step)
        if step.interrupted:
            await self.close_response("cancelled")
        elif step.end_of_turn:
            await self.close_response("completed")

    # -- response lifecycle ------------------------------------------------

    async def open_response(self, *, message_item: bool = True) -> ActiveResponse:
        """Open a response and announce it (profile §9, DESIGN.md 5.3).

        `message_item=False` opens a response that carries only `function_call` items,
        which is what a Brain transfer intercepted before the model speaks needs.
        """
        active = ActiveResponse(
            id=ev.new_response_id(),
            item_id=ev.new_item_id() if message_item else None,
            metadata=self.session.pending_response_metadata,
        )
        self.active = active
        self.session.active_response_id = active.id
        self.session.pending_response_metadata = None
        self.session.client_cancel_requested = False

        await self.session.emit(
            ev.ResponseCreated(
                response=self._response_object(
                    active,
                    status="in_progress",
                    status_details=None,
                    output=[],
                    usage=None,
                )
            )
        )
        if active.item_id is None:
            return active

        previous_item_id = self.session.last_item_id
        item = self.session.register_item(
            {
                "id": active.item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            }
        )
        await self.session.emit(
            ev.ResponseOutputItemAdded(
                response_id=active.id, output_index=MESSAGE_OUTPUT_INDEX, item=item
            )
        )
        await self.session.emit(
            ev.ConversationItemCreated(item=item, previous_item_id=previous_item_id)
        )
        await self.session.emit(
            ev.ResponseContentPartAdded(
                response_id=active.id,
                item_id=active.item_id,
                output_index=MESSAGE_OUTPUT_INDEX,
                content_index=CONTENT_INDEX,
                part={"type": "audio", "transcript": ""},
            )
        )
        return active

    async def close_response(self, status: str = "completed") -> None:
        """Close the streaming response with `completed` or `cancelled` (profile §9)."""
        active = self.active
        if active is None:
            return
        self.active = None
        self.session.active_response_id = None
        cancelled = status == "cancelled"
        reason = "client_cancelled" if self.session.client_cancel_requested else "turn_detected"
        self.session.client_cancel_requested = False

        output: list[dict[str, Any]] = []
        if active.item_id is not None:
            await self.session.emit(
                ev.ResponseOutputAudioDone(
                    response_id=active.id,
                    item_id=active.item_id,
                    output_index=MESSAGE_OUTPUT_INDEX,
                    content_index=CONTENT_INDEX,
                )
            )
            await self.session.emit(
                ev.ResponseOutputAudioTranscriptDone(
                    response_id=active.id,
                    item_id=active.item_id,
                    output_index=MESSAGE_OUTPUT_INDEX,
                    content_index=CONTENT_INDEX,
                    transcript=active.transcript,
                )
            )
            await self.session.emit(
                ev.ResponseContentPartDone(
                    response_id=active.id,
                    item_id=active.item_id,
                    output_index=MESSAGE_OUTPUT_INDEX,
                    content_index=CONTENT_INDEX,
                    part={"type": "audio", "transcript": active.transcript},
                )
            )
            item = self.session.register_item(
                {
                    "id": active.item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "assistant",
                    "status": "incomplete" if cancelled else "completed",
                    "content": [{"type": "output_audio", "transcript": active.transcript}],
                }
            )
            output.append(item)
            await self.session.emit(
                ev.ResponseOutputItemDone(
                    response_id=active.id, output_index=MESSAGE_OUTPUT_INDEX, item=item
                )
            )
            if cancelled:
                await self.session.emit(
                    ev.ConversationItemTruncated(
                        item_id=active.item_id,
                        content_index=CONTENT_INDEX,
                        audio_end_ms=round(active.audio_ms),
                    )
                )
        output.extend(active.tool_items)

        status_details = {"type": "cancelled", "reason": reason} if cancelled else None
        await self.session.emit(
            ev.ResponseDone(
                response=self._response_object(
                    active,
                    status=status,
                    status_details=status_details,
                    output=output,
                    usage=_usage(active.transcript),
                )
            )
        )

    # -- ASR side channel --------------------------------------------------

    async def on_asr_transcript(
        self, transcript: Transcript | str, *, item_id: str | None = None
    ) -> bool:
        """Emit `conversation.item.input_audio_transcription.completed` for one segment.

        The segment is attached to the last committed input item unless `item_id` names
        another one. Returns `False` when nothing has been committed yet, in which case
        there is no item the transcript could belong to and no event is emitted.
        """
        target = item_id or self.session.input_item_id
        if target is None:
            return False
        text = transcript if isinstance(transcript, str) else str(transcript.text)
        start_ms = 0 if isinstance(transcript, str) else int(getattr(transcript, "start_ms", 0))
        end_ms = 0 if isinstance(transcript, str) else int(getattr(transcript, "end_ms", 0))

        stored = self.session.items.get(target)
        if stored is not None:
            content = stored.get("content")
            if isinstance(content, list) and content:
                content[0]["transcript"] = text

        await self.session.emit(
            ev.ConversationItemInputAudioTranscriptionCompleted(
                item_id=target,
                content_index=CONTENT_INDEX,
                transcript=text,
                usage={"type": "duration", "seconds": round(max(end_ms - start_ms, 0) / 1000.0, 3)},
            )
        )
        return True

    # -- the Brain seam (T3.7) ---------------------------------------------

    async def emit_function_call(self, call_id: str, name: str, arguments: str) -> str:
        """Emit the `function_call` pair for one Brain tool call; return its item id.

        This is the method the T3.7 wiring calls when the Brain hands over
        `transfer_to_human`. The arguments arrive complete, so exactly one
        `response.function_call_arguments.delta` is emitted, immediately followed by
        `.done` (profile §9). A call that arrives while the model is speaking joins the
        streaming response as the next output item; a call intercepted before the model
        speaks opens and closes a response of its own.
        """
        standalone = self.active is None
        active = self.active or await self.open_response(message_item=False)
        output_index = active.next_output_index()
        previous_item_id = self.session.last_item_id
        item = self.session.register_item(
            {
                "id": ev.new_item_id(),
                "object": "realtime.item",
                "type": "function_call",
                "name": name,
                "call_id": call_id,
                "arguments": arguments,
                "status": "in_progress",
            }
        )
        item_id = str(item["id"])
        await self.session.emit(
            ev.ResponseOutputItemAdded(response_id=active.id, output_index=output_index, item=item)
        )
        await self.session.emit(
            ev.ConversationItemCreated(item=item, previous_item_id=previous_item_id)
        )
        await self.session.emit(
            ev.ResponseFunctionCallArgumentsDelta(
                response_id=active.id,
                item_id=item_id,
                output_index=output_index,
                call_id=call_id,
                delta=arguments,
            )
        )
        await self.session.emit(
            ev.ResponseFunctionCallArgumentsDone(
                response_id=active.id,
                item_id=item_id,
                output_index=output_index,
                call_id=call_id,
                name=name,
                arguments=arguments,
            )
        )
        done = self.session.register_item({**item, "status": "completed"})
        active.tool_items.append(done)
        await self.session.emit(
            ev.ResponseOutputItemDone(response_id=active.id, output_index=output_index, item=done)
        )
        if standalone:
            await self.close_response("completed")
        return item_id

    # -- helpers -----------------------------------------------------------

    def _message_item_id(self, active: ActiveResponse) -> str:
        """The message item id, or the response id when the response carries no message."""
        return active.item_id if active.item_id is not None else active.id

    async def _emit_audio(self, active: ActiveResponse, step: EngineStepEvent) -> None:
        """Encode one unit of model audio into `response.output_audio.delta`."""
        item_id = active.item_id
        if item_id is None:  # a tool-call-only response carries no audio part
            return
        samples = _to_pcm16(step.audio_waveform)
        rate = step.sample_rate or OUTPUT_SAMPLE_RATE
        if rate != CLIENT_SAMPLE_RATE:
            if active.client_resampler is None:
                active.client_resampler = StreamResampler(rate, CLIENT_SAMPLE_RATE)
            samples = active.client_resampler.process_pcm16(samples)
        audio_format = self.session.output_audio_format
        if audio_format in _G711_FORMATS and active.g711_resampler is None:
            active.g711_resampler = StreamResampler(CLIENT_SAMPLE_RATE, G711_SAMPLE_RATE)
        delta = encode_output_audio(samples, audio_format, active.g711_resampler)
        if not delta:
            return
        active.audio_ms += 1000.0 * int(samples.size) / CLIENT_SAMPLE_RATE
        await self.session.emit(
            ev.ResponseOutputAudioDelta(
                response_id=active.id,
                item_id=item_id,
                output_index=MESSAGE_OUTPUT_INDEX,
                content_index=CONTENT_INDEX,
                delta=delta,
            )
        )

    def _response_object(
        self,
        active: ActiveResponse,
        *,
        status: str,
        status_details: Mapping[str, Any] | None,
        output: list[dict[str, Any]],
        usage: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """The `response` object of profile §9, in the shape both `response.*` carry."""
        audio_output = self.session.session["audio"]["output"]
        return {
            "id": active.id,
            "object": "realtime.response",
            "status": status,
            "status_details": dict(status_details) if status_details is not None else None,
            "output": output,
            "usage": dict(usage) if usage is not None else None,
            "conversation_id": self.session.conversation_id,
            "output_modalities": list(self.session.session["output_modalities"]),
            "audio": {
                "output": {
                    "format": dict(audio_output["format"]),
                    "voice": audio_output["voice"],
                }
            },
            "max_output_tokens": self.session.session["max_output_tokens"],
            "metadata": dict(active.metadata) if active.metadata else None,
        }


def _to_pcm16(waveform: np.ndarray | None) -> np.ndarray:
    """Model audio as int16, whether the engine hands over float32 or int16."""
    data = np.asarray(waveform)
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.int16, copy=False)
    return float_to_pcm16(data)


def _usage(transcript: str) -> dict[str, Any]:
    """Best-effort `response.usage`: audio tokens are always 0 (profile §11)."""
    text_tokens = ev.estimate_tokens(transcript)
    return {
        "total_tokens": text_tokens,
        "input_tokens": 0,
        "output_tokens": text_tokens,
        "input_token_details": {"text_tokens": 0, "audio_tokens": 0, "cached_tokens": 0},
        "output_token_details": {"text_tokens": text_tokens, "audio_tokens": 0},
    }
