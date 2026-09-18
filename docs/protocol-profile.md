# Talkover — OpenAI Realtime GA compatibility profile

Status: draft, tracks `docs/DESIGN.md` section 5. Field paths, rejection codes and error
event formats are ported from `cascade-realtime-gateway`'s `docs/protocol-profile.md`
(documentation and test cases only; no code is reused). Section 11 lists every point where
Talkover differs from Cascade.

Sections 8 and 9 are machine-read by `tests/protocol/test_events.py`: every code in §8.1 and
every case id in §8.2 must have a test, so the document and the suite cannot drift. §8.3 is
machine-read the same way by `tests/protocol/test_session.py`.

## 0. Scope

Only the GA protocol is served: `session.type = "realtime"`, `output_modalities`,
`session.audio.input/output`, `response.output_audio.*`, `response.output_audio_transcript.*`.
Beta shapes — `modalities`, top-level `voice` / `turn_detection` / `input_audio_format`,
`response.audio.*`, `response.text.*` — are rejected, never translated. No beta alias is
accepted anywhere.

Markers:

- **Supported** — accepted and implemented as documented.
- **Rejected** — an `error` event is emitted (§8) and the whole client event is discarded;
  the session stays open.
- **Ignored** — accepted without effect; the echo column says what `session.updated` reports.

`src/talkover/realtime/events.py` implements §2 to §5 and §8. State-dependent rejections
(unknown `item_id`, duplicate `call_id`, "a response is already in progress") are raised by
the session state machine, `src/talkover/realtime/session.py`, with the same codes; they are
listed in §8.3.

## 1. Transport

| Item | Profile |
|---|---|
| Endpoint | `GET /v1/realtime` upgraded to WebSocket. |
| Auth | `Authorization: Bearer <server.api_key>`; a single key. Missing or wrong → HTTP 401 before the upgrade. |
| `?model=` | Accepted and echoed as `session.model`; not used for routing. Absent → echo `"talkover"`. |
| `OpenAI-Beta` header | Ignored; the GA shape is the only one served. |
| Message framing | One JSON object per text frame. Binary frames → Rejected (`invalid_event`). |
| First server events | `session.created` then `conversation.created`, before any client event is processed. |
| Concurrency | One session per process. A second connection gets `error` with `code: "engine_busy"` and is closed; `/health` returns 503 while busy (§10). |

## 2. Client events

| Event | Status | Notes |
|---|---|---|
| `session.update` | Supported | Body validated against §3; any rejected or unknown field rejects the whole event. Success → `session.updated` with the full effective session. |
| `input_audio_buffer.append` | Supported | `audio` is base64 of the negotiated input format, decoded, resampled to 16 kHz and fed to the engine one 1 s unit at a time. No acknowledgement event. An undecodable payload → Rejected (`invalid_value`, `param: "audio"`). |
| `input_audio_buffer.commit` | Supported | The buffered incomplete unit is zero-padded and fed first, then `flush_pending` on the engine. Success → `input_audio_buffer.committed` → `conversation.item.created` (user `input_audio` item whose `transcript` is `null` until the ASR segment completes). Empty buffer → Rejected (`invalid_event`). |
| `input_audio_buffer.clear` | Supported | Drops the incomplete 1 s unit; whole units already fed to the engine cannot be recalled. → `input_audio_buffer.cleared`. |
| `conversation.item.create` | Supported (subset, §5) | → `conversation.item.created`. |
| `conversation.item.truncate` | Supported | Acknowledged only; the model context is not rewound (§11). `content_index` must be `0`. Success → `conversation.item.truncated`. |
| `conversation.item.delete` | Supported | Acknowledged only. Unknown id → Rejected (`invalid_value`, `param: "item_id"`). Success → `conversation.item.deleted`. |
| `response.create` | Supported (subset, §4) | Behaves as `flush_pending` (the buffered incomplete unit is fed first, as on commit); the model decides on its own whether to speak. There is no immediate acknowledgement: `response.created` follows only when the model stops listening (§9). A second `response.create` while a response is streaming → Rejected (`invalid_event`). |
| `response.cancel` | Supported | `interrupt_output` on the engine; there is no immediate acknowledgement, `response.done` with `status: "cancelled"` follows the model's `interrupted` unit. Nothing to cancel → Rejected (`invalid_event`). A `response_id` that is not the active one → Rejected (`invalid_value`, `param: "response_id"`). |
| `conversation.item.retrieve` | Rejected | Talkover does not retain input audio; `invalid_event`, `param: "type"`. |
| `output_audio_buffer.clear` | Rejected | WebRTC and SIP sessions only; `invalid_event`, `param: "type"`. |
| `transcription_session.update` | Rejected | Transcription sessions are not served; `invalid_event`, `param: "type"`. |
| beta event names (`response.audio.delta`, …) | Rejected | `invalid_event`, `param: "type"`. |
| any other `type` | Rejected | `invalid_value`, `param: "type"`. |
| missing `type` | Rejected | `invalid_event`, message `"The 'type' field is missing."` |
| `event_id` on client events | Supported | Echoed as `error.event_id` on any error the event causes. Maximum length 512. |

## 3. Session object

`session.update.session`, echoed by `session.created` and `session.updated`. The echo is the
full effective session in GA request shape plus `id` (`sess_…`) and
`object: "realtime.session"`.

| Field | Status | Accepted values | Echo |
|---|---|---|---|
| `type` | Supported | required; must be `"realtime"` | `"realtime"` |
| `model` | Ignored | any string | last value sent, else `?model=`, else `"talkover"` |
| `instructions` | Supported | any string; written into the task slate, truncated to `realtime.memory_slate_max_tokens` (§6) | the **truncated** value |
| `output_modalities` | Supported | exactly `["audio"]`; `["text"]` → Rejected | `["audio"]` |
| `audio.input.format` | Supported | `{type, rate}`: `audio/pcm` 24000, `audio/pcmu` 8000, `audio/pcma` 8000; both keys optional | effective object |
| `audio.input.transcription` | Supported (subset) | `null` disables; object with `model`, `language`, `prompt`. Any other key → Rejected | effective object or `null` |
| `audio.input.noise_reduction` | Ignored | `null`, or `{type: "near_field" \| "far_field"}` | effective object or `null` |
| `audio.input.turn_detection` | Supported (approximate, §7) | `null`; `server_vad` with `threshold` [0,1], `prefix_padding_ms` ≥ 0, `silence_duration_ms` > 0, `create_response`, `interrupt_response`; `semantic_vad` with `eagerness` low/medium/high/auto, `create_response`, `interrupt_response`. Fields of the other mode → Rejected. `idle_timeout_ms` non-null → Rejected | effective object or `null` |
| `audio.output.format` | Supported | `audio/pcm` 24000, `audio/pcmu` 8000, `audio/pcma` 8000 | effective object |
| `audio.output.voice` | Ignored | any string | effective string |
| `audio.output.speed` | Ignored | number in [0.25, 1.5] | effective number |
| `max_output_tokens` | Ignored | `"inf"` or an integer ≥ 1 | effective value |
| `tools` | Supported (subset, §11) | array of `{type:"function", name, description?, parameters?}`. `type` and `name` required; `parameters` an object; duplicate `name` → Rejected | effective array; `[]` when never set |
| `tool_choice` | Ignored | `"none"` \| `"auto"` \| `"required"`, or `{type:"function", name}` naming a declared tool | effective value |
| `truncation` | Ignored if `"auto"`, else Rejected | | `"auto"` |
| `tracing` | Ignored if `null`, else Rejected | | `null` |
| `prompt` | Ignored if `null`, else Rejected | | `null` |
| `include` | Ignored if `[]` or `null`, else Rejected | | not echoed |
| unknown field | Rejected | `invalid_value`, `param` = dotted path, message "Unknown parameter" | |
| beta top-level fields (`modalities`, `voice`, `turn_detection`, `input_audio_format`, `output_audio_format`, `input_audio_transcription`, `temperature`, `max_response_output_tokens`) | Rejected | `invalid_value`, `param` = the field, message "Unknown parameter … (beta shape; the GA session shape is required)" | |

Merge semantics: only fields present are updated; nested objects merge field-wise; `null`
clears the nullable fields (`transcription`, `turn_detection`, `noise_reduction`, `prompt`,
`tracing`).

Effective values before the first `session.update`, as echoed by `session.created`:

```json
{"type": "realtime", "instructions": "", "output_modalities": ["audio"],
 "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": null,
                     "noise_reduction": null,
                     "turn_detection": {"type": "server_vad", "threshold": 0.5,
                                        "prefix_padding_ms": 300, "silence_duration_ms": 200}},
           "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": "gander",
                      "speed": 1.0}},
 "tools": [], "tool_choice": "auto", "max_output_tokens": "inf", "truncation": "auto",
 "tracing": null, "prompt": null}
```

`turn_detection` defaults to the GA `server_vad` values so that a client gets the approximate
`speech_started` / `speech_stopped` events without asking (§7); `voice` defaults to `gander`
because the Talker voice is fixed by the checkpoint and the field is echo-only.

Changing `audio.input.format` restarts the decoder, the resampler and the framer, so the
buffered incomplete unit is dropped; whole units already fed to the engine are unaffected.

## 4. `response.create.response`

| Field | Status | Notes |
|---|---|---|
| `conversation` | Ignored if `"auto"`, else Rejected | out-of-band responses are out of scope |
| `metadata` | Ignored | at most 16 string pairs; echoed on the response object |
| `instructions` | Rejected | `session.instructions` is the only path into the task slate (§6) |
| `output_modalities`, `max_output_tokens`, `audio` | Rejected | per-response overrides do not exist; the model owns the turn |
| `input` | Rejected | out-of-band context is out of scope |
| `tools`, `tool_choice`, `parallel_tool_calls`, `prompt`, `reasoning` | Rejected | tools are session-level only |
| unknown field | Rejected | `invalid_value`, `param` = dotted path |

A bare `response.create` with no `response` object is the expected form.

## 5. Items accepted by `conversation.item.create`

| Item | Status |
|---|---|
| `{type:"message", role:"user", content:[{type:"input_text", text}]}` | Supported — fed to the runtime as a text user turn |
| `{type:"function_call_output", call_id, output}` | Supported — acknowledged with `conversation.item.created` and then routed to the Brain (§9.1); `call_id` and `output` are required, a `call_id` may be answered only once (§8.3), and it must name a `function_call` this session emitted (§8.4) |
| `role:"system"` or `role:"assistant"` message | Rejected (`invalid_value`, `param: "item.role"`) |
| `input_audio` / `input_image` / `output_text` / `output_audio` content | Rejected (`invalid_value`, `param: "item.content[0].type"`) |
| multiple content parts | Rejected (`invalid_value`, `param: "item.content"`) |
| `function_call`, `item_reference`, `mcp_*` | Rejected (`invalid_value`, `param: "item.type"`) |
| client-supplied `id` | Supported; must be unique in the session (duplicate → `invalid_value`, `param: "item.id"`) |
| `status`, `object` | Ignored |
| `previous_item_id` | Accepted; ordering only |

Item shape on the wire: `{id, object:"realtime.item", type:"message", role, status,
content:[part]}` where `part` is `{type:"input_text", text}`, `{type:"input_audio",
transcript}`, or `{type:"output_audio", transcript}` (audio bytes omitted). The two tool items
carry neither `role` nor `content`:

```text
function_call         {id, object:"realtime.item", type:"function_call",
                       name, call_id, arguments, status}
function_call_output  {id, object:"realtime.item", type:"function_call_output",
                       call_id, output, status}
```

`arguments` and `output` are JSON-encoded strings; Talkover parses neither.

## 6. `instructions` and the task slate

The Cerebellum's system prefix is fixed upstream, so `session.instructions` is written to
`set_task_slate` and nothing else (DESIGN.md 5.5). The value is truncated to
`realtime.memory_slate_max_tokens` (256 by default) and the **truncated** value is echoed in
`session.updated`, so a client can see what actually took effect. Over-length instructions are
never an error.

The protocol layer has no tokenizer, so `events.estimate_tokens` approximates: one token per
CJK character, one token per four other characters. The cap is a safety bound on the slate,
not an exact budget. Business phrasing belongs in the Brain system prompt.

## 7. Turn detection

Gander has no explicit VAD boundary, so `server_vad` and `semantic_vad` map to the same
approximate implementation (DESIGN.md 5.4):

- `input_audio_buffer.speech_started` — from the ASR backend's VAD signal, or emitted
  retroactively when the model reports `interrupted: true`.
- `input_audio_buffer.speech_stopped` — at the end of an ASR segment.
- `create_response` and `interrupt_response` are accepted and do not change behavior; the
  model decides when to speak and when to stop.
- `threshold`, `prefix_padding_ms`, `silence_duration_ms` and `eagerness` are validated and
  echoed; only the ASR VAD acts on them, so they are approximate.
- `idle_timeout_ms` is validated and echoed but never acts:
  `input_audio_buffer.timeout_triggered` is in the never-emitted list of §9.
- `turn_detection: null` synthesizes neither event.

How `talkover.realtime.turn_detection` makes the two events concrete:

- A speech span opens once and closes once. A second `speech_started` inside an open span,
  and a `speech_stopped` with no span open, are both dropped, so the events always alternate.
- `audio_start_ms` from the ASR path is the VAD's own position on the input stream. On the
  retroactive path it is the start of the interrupted unit, `(unit_index - 1) * 1000`, since
  a barge-in is only visible once the unit that was cut short has been produced. The two
  clocks are separate approximations of the same audio, so `audio_end_ms` is clamped up to
  `audio_start_ms` rather than allowed to precede it.
- `item_id` names the user item the audio ends up in. Talkover creates that item at
  `input_audio_buffer.commit`, which comes after the span, so the id is minted with
  `speech_started`, repeated on `speech_stopped` and then used for the committed item. A span
  that is never committed leaves an unused id behind; when two spans precede one commit the
  committed item carries the id of the later span.
- Events that carry only late Talker audio (`is_audio_chunk`) belong to an earlier unit and
  never open a span.

## 8. Errors

Shape: `{type:"error", event_id, error:{type, code, message, param, event_id}}`, where
`error.event_id` is the client `event_id` that caused it, or `null` for server-originated
errors. `event_id` on the envelope is a fresh server id.

### 8.1 Rejection codes

| `error.type` | `error.code` | When | `param` |
|---|---|---|---|
| `invalid_request_error` | `invalid_event` | unparsable JSON, binary frame, missing `type`, a rejected or unknown event `type`, or an event that is not valid in the current state (commit on an empty buffer, `response.cancel` with nothing to cancel) | `"type"` for type problems, otherwise `null` |
| `invalid_request_error` | `invalid_value` | a field is missing, has the wrong type, an out-of-profile value, or is unknown ("Unknown parameter"); references an unknown item or a foreign `response_id` | dotted path of the field |
| `server_error` | `engine_busy` | a second connection while a session is running (§10) | `null` |
| `server_error` | `engine_error` | the inference engine, ASR, or the Brain failed; fatal | `null` |

A fatal error is followed by WebSocket close code `1011` with the code as the close reason.

### 8.2 Rejection cases

Each case id below has a test in `tests/protocol/test_events.py`.

| Case | Cause | `code` | `param` |
|---|---|---|---|
| `frame_not_json` | frame is not valid JSON | `invalid_event` | |
| `frame_not_object` | frame is valid JSON but not an object | `invalid_event` | |
| `frame_bad_utf8` | binary frame that is not UTF-8 | `invalid_event` | |
| `missing_type` | no `type` field | `invalid_event` | |
| `type_not_string` | `type` is not a string | `invalid_value` | `type` |
| `unknown_type` | unknown event `type` | `invalid_value` | `type` |
| `rejected_type_retrieve` | `conversation.item.retrieve` | `invalid_event` | `type` |
| `rejected_type_output_audio_buffer_clear` | `output_audio_buffer.clear` | `invalid_event` | `type` |
| `rejected_type_transcription_session` | `transcription_session.update` | `invalid_event` | `type` |
| `rejected_type_beta_alias` | beta event name `response.audio.delta` | `invalid_event` | `type` |
| `event_id_not_string` | `event_id` is not a string | `invalid_value` | `event_id` |
| `event_id_too_long` | `event_id` longer than 512 characters | `invalid_value` | `event_id` |
| `unknown_top_level_field` | unknown field on a client event | `invalid_value` | `garbage` |
| `session_missing` | `session.update` without `session` | `invalid_value` | `session` |
| `session_not_object` | `session` is not an object | `invalid_value` | `session` |
| `session_type_missing` | `session.type` absent | `invalid_value` | `session.type` |
| `session_type_beta` | `session.type` is not `"realtime"` | `invalid_value` | `session.type` |
| `session_unknown_field` | unknown session field | `invalid_value` | `session.nope` |
| `session_beta_field` | beta top-level session field | `invalid_value` | `session.voice` |
| `session_instructions_type` | `instructions` is not a string | `invalid_value` | `session.instructions` |
| `session_modalities_text` | `output_modalities` is `["text"]` | `invalid_value` | `session.output_modalities` |
| `session_input_format_type` | unsupported `audio.input.format.type` | `invalid_value` | `session.audio.input.format.type` |
| `session_input_format_rate` | wrong rate for the format | `invalid_value` | `session.audio.input.format.rate` |
| `session_output_format_type` | unsupported `audio.output.format.type` | `invalid_value` | `session.audio.output.format.type` |
| `session_transcription_field` | unsupported transcription field | `invalid_value` | `session.audio.input.transcription.delay` |
| `session_noise_reduction_type` | unsupported noise reduction type | `invalid_value` | `session.audio.input.noise_reduction.type` |
| `session_speed_range` | `speed` outside [0.25, 1.5] | `invalid_value` | `session.audio.output.speed` |
| `session_voice_type` | `voice` is not a string | `invalid_value` | `session.audio.output.voice` |
| `turn_detection_type` | unsupported turn detection type | `invalid_value` | `session.audio.input.turn_detection.type` |
| `turn_detection_threshold` | `threshold` outside [0, 1] | `invalid_value` | `session.audio.input.turn_detection.threshold` |
| `turn_detection_prefix_padding` | negative `prefix_padding_ms` | `invalid_value` | `session.audio.input.turn_detection.prefix_padding_ms` |
| `turn_detection_silence_duration` | `silence_duration_ms` not > 0 | `invalid_value` | `session.audio.input.turn_detection.silence_duration_ms` |
| `turn_detection_eagerness` | `eagerness` outside the enum | `invalid_value` | `session.audio.input.turn_detection.eagerness` |
| `turn_detection_foreign_field` | `server_vad` field under `semantic_vad` | `invalid_value` | `session.audio.input.turn_detection.threshold` |
| `turn_detection_idle_timeout` | non-null `idle_timeout_ms` | `invalid_value` | `session.audio.input.turn_detection.idle_timeout_ms` |
| `tool_type` | tool entry is not `type: "function"` | `invalid_value` | `session.tools[0].type` |
| `tool_name_missing` | tool entry without `name` | `invalid_value` | `session.tools[0].name` |
| `tool_name_duplicate` | duplicate tool name | `invalid_value` | `session.tools[1].name` |
| `tool_parameters_type` | `parameters` is not an object | `invalid_value` | `session.tools[0].parameters` |
| `tool_choice_value` | `tool_choice` outside the four forms | `invalid_value` | `session.tool_choice` |
| `tool_choice_unknown_name` | forced function not declared in `tools` | `invalid_value` | `session.tool_choice.name` |
| `session_max_output_tokens` | `max_output_tokens` neither `"inf"` nor ≥ 1 | `invalid_value` | `session.max_output_tokens` |
| `session_truncation_value` | `truncation` other than `"auto"` | `invalid_value` | `session.truncation` |
| `session_include_value` | `include` other than `[]` or `null` | `invalid_value` | `session.include` |
| `append_audio_missing` | `input_audio_buffer.append` without `audio` | `invalid_value` | `audio` |
| `append_audio_type` | `audio` is not a string | `invalid_value` | `audio` |
| `response_create_instructions` | per-response `instructions` | `invalid_value` | `response.instructions` |
| `response_create_audio` | per-response `audio` override | `invalid_value` | `response.audio` |
| `response_create_tools` | per-response `tools` | `invalid_value` | `response.tools` |
| `response_create_conversation` | `conversation` other than `"auto"` | `invalid_value` | `response.conversation` |
| `response_create_metadata_size` | more than 16 metadata entries | `invalid_value` | `response.metadata` |
| `response_create_unknown_field` | unknown `response` field | `invalid_value` | `response.nope` |
| `response_cancel_id_type` | `response_id` is not a string | `invalid_value` | `response_id` |
| `item_missing` | `conversation.item.create` without `item` | `invalid_value` | `item` |
| `item_type_missing` | item without `type` | `invalid_value` | `item.type` |
| `item_type_unknown` | unknown item type | `invalid_value` | `item.type` |
| `item_type_function_call` | client-created `function_call` item | `invalid_value` | `item.type` |
| `item_role_assistant` | assistant message item | `invalid_value` | `item.role` |
| `item_role_system` | system message item | `invalid_value` | `item.role` |
| `item_content_multiple` | more than one content part | `invalid_value` | `item.content` |
| `item_content_input_audio` | `input_audio` content part | `invalid_value` | `item.content[0].type` |
| `item_content_text_type` | content `text` is not a string | `invalid_value` | `item.content[0].text` |
| `item_call_id_missing` | `function_call_output` without `call_id` | `invalid_value` | `item.call_id` |
| `item_output_missing` | `function_call_output` without `output` | `invalid_value` | `item.output` |
| `truncate_missing_field` | `conversation.item.truncate` without `audio_end_ms` | `invalid_value` | `audio_end_ms` |
| `truncate_content_index` | `content_index` other than `0` | `invalid_value` | `content_index` |
| `truncate_audio_end_ms` | negative `audio_end_ms` | `invalid_value` | `audio_end_ms` |
| `delete_item_id_missing` | `conversation.item.delete` without `item_id` | `invalid_value` | `item_id` |
| `engine_busy` | a second concurrent connection | `engine_busy` | |
| `engine_error` | engine, ASR or Brain failure | `engine_error` | |

### 8.3 State-dependent rejections

`src/talkover/realtime/session.py` raises these; they cannot be decided by
`parse_client_event`, which is stateless. Each case id below has a test in
`tests/protocol/test_session.py`, which reads this table the same way
`tests/protocol/test_events.py` reads §8.2.

| Case | Cause | `code` | `param` |
|---|---|---|---|
| `append_audio_payload` | `audio` is not decodable in the negotiated input format | `invalid_value` | `audio` |
| `commit_empty_buffer` | `input_audio_buffer.commit` with nothing appended since the last commit or clear | `invalid_event` | |
| `response_create_in_progress` | `response.create` while a response is streaming | `invalid_event` | |
| `response_cancel_nothing` | `response.cancel` with no response in progress | `invalid_event` | |
| `response_cancel_foreign_id` | `response_id` is not the response in progress | `invalid_value` | `response_id` |
| `item_id_duplicate` | client-supplied item `id` already used in this session | `invalid_value` | `item.id` |
| `item_call_id_duplicate` | `function_call_output` repeats a `call_id` already answered | `invalid_value` | `item.call_id` |
| `truncate_unknown_item` | `conversation.item.truncate` for an unknown `item_id` | `invalid_value` | `item_id` |
| `delete_unknown_item` | `conversation.item.delete` for an unknown `item_id` | `invalid_value` | `item_id` |
| `engine_call_failed` | an `EngineProtocol` call raised `EngineError`; fatal | `engine_error` | |

A binary frame is rejected with `invalid_event` and `param: null` by the session, not by
`parse_client_event` (§1); `frame_bad_utf8` in §8.2 covers the pure-decoding half of that rule.

### 8.4 Brain bridge rejections

`src/talkover/realtime/brain_bridge.py` raises these: they depend on what the Brain has
asked the client to do, which neither `parse_client_event` nor the session state machine
knows. `tests/protocol/test_function_call.py` machine-reads this table the way
`tests/protocol/test_session.py` reads §8.3.

| Case | Cause | `code` | `param` |
|---|---|---|---|
| `function_call_output_unknown_call_id` | `function_call_output` whose `call_id` names no `function_call` this session emitted, or one the Brain has already resolved | `invalid_value` | `item.call_id` |

The item is created and acknowledged before the routing is attempted, so this rejection
arrives **after** its `conversation.item.created`, and the `call_id` counts as answered
(§8.3) even though nothing consumed it.

## 9. Server events

| Event | Emitted when |
|---|---|
| `session.created` | first event after the upgrade |
| `conversation.created` | immediately after `session.created` |
| `session.updated` | after a successful `session.update` |
| `error` | any rejected event or runtime failure (§8) |
| `input_audio_buffer.committed` | after `input_audio_buffer.commit`; `item_id`, `previous_item_id` |
| `input_audio_buffer.cleared` | after `input_audio_buffer.clear` |
| `input_audio_buffer.speech_started` | approximate turn detection (§7); `audio_start_ms`, `item_id` |
| `input_audio_buffer.speech_stopped` | approximate turn detection (§7); `audio_end_ms`, `item_id` |
| `conversation.item.created` | a client-created item, a committed audio item, or a response output item |
| `conversation.item.truncated` | after `conversation.item.truncate` |
| `conversation.item.deleted` | after `conversation.item.delete` |
| `conversation.item.input_audio_transcription.completed` | an ASR segment finished; `transcript`, `usage` |
| `response.created` | first unit with `is_listen: false` |
| `response.output_item.added` | with `response.created` |
| `response.content_part.added` | right after `response.output_item.added`; `part: {type:"audio", transcript:""}` |
| `response.output_audio_transcript.delta` | `DuplexStepEvent.text` |
| `response.output_audio.delta` | `DuplexStepEvent.audio_waveform`, base64 in the negotiated output format |
| `response.output_audio.done` / `response.output_audio_transcript.done` | `end_of_turn: true`, in that order |
| `response.content_part.done` | before `response.output_item.done`; `part` carries the final transcript |
| `response.output_item.done` | terminal, with the full item (audio bytes omitted) |
| `response.function_call_arguments.delta` | Brain emitted `transfer_to_human`; carries the **complete** arguments |
| `response.function_call_arguments.done` | immediately after, with the final arguments |
| `response.done` | always, with `status` ∈ `completed` / `cancelled` / `failed`, `status_details`, `usage`, `output` |

Never emitted: `conversation.item.added` / `conversation.item.done`, `rate_limits.updated`,
`response.output_text.*`, `conversation.item.input_audio_transcription.delta` / `.failed` /
`.segment`, `input_audio_buffer.timeout_triggered`, `input_audio_buffer.dtmf_event_received`,
`output_audio_buffer.*`, `conversation.item.retrieved`, `response.mcp_call*`,
`mcp_list_tools.*`, `transcription_session.updated`.

Field details:

- Every server event carries a unique `event_id` (`event_…`).
- `response` object: `{id, object:"realtime.response", status, status_details, output, usage,
  conversation_id, output_modalities, audio:{output:{format, voice}}, max_output_tokens,
  metadata}`. In `response.created`: `status:"in_progress"`, `status_details:null`,
  `output:[]`, `usage:null`.
- `status_details`: `{type:"cancelled", reason:"turn_detected"|"client_cancelled"}`,
  `{type:"failed", error:{type, code}}`, or `null` when completed.
- `content_index` is always `0`; it does not apply to `function_call` items. `output_index` is
  `0` for the message item and `1` for a `function_call` that follows it.
- `response.function_call_arguments.delta` is emitted once, with the complete arguments; the
  Brain hands over a finished tool call, so there are no fragments to accumulate.
- An interrupted response emits `response.done` with `status: "cancelled"` and
  `status_details.reason: "turn_detected"`, plus `conversation.item.truncated`.

The response half of the table is `src/talkover/realtime/mapping.py`; these are the rules it
settles, all covered by `tests/protocol/test_mapping.py`:

- A closing response always emits the whole chain, in this order: `response.output_audio.done`,
  `response.output_audio_transcript.done`, `response.content_part.done`,
  `response.output_item.done`, then `response.done`. A cancelled response inserts
  `conversation.item.truncated` between `response.output_item.done` and `response.done`, with
  `audio_end_ms` equal to the assistant audio already sent, and its output item carries
  `status: "incomplete"` where a completed one carries `"completed"`.
- `status_details.reason` is `client_cancelled` when a `response.cancel` preceded the model's
  `interrupted` unit and `turn_detected` otherwise; the flag is consumed by the response it
  cancels, so a later barge-in is `turn_detected` again.
- The assistant message item is announced once, as `response.output_item.added` plus
  `conversation.item.created`, and is retained in the conversation in its final form, so
  `conversation.item.truncate` and `.delete` can reference it.
- `response.usage` is best-effort: `output_tokens` and `total_tokens` are
  `events.estimate_tokens` of the transcript, `input_tokens` is `0`, and every audio token
  count is `0`. `usage` is `null` on `response.created` and present on `response.done`.
- `response.output_audio.delta` carries the negotiated output format: pcm16 at 24 kHz, or
  g711 at 8 kHz, in which case one resampler is kept for the whole response so the chunk
  boundaries stay seamless. A unit with no waveform emits no delta.
- `conversation.item.input_audio_transcription.completed` targets the last item created by
  `input_audio_buffer.commit` and also fills that item's `content[0].transcript`. Its `usage`
  is `{type: "duration", seconds}` over the ASR segment's span. A segment that arrives before
  anything was committed has no item to name and is dropped.
- A `function_call` joins the streaming response as its next output item and appears in
  `response.done.output` after the message item. When the Brain intercepts a transfer before
  the model speaks there is no response to join, so one is opened for the call alone: it
  carries no message item, the `function_call` takes `output_index: 0`, and it is closed with
  `status: "completed"` right after `.done`.
- A unit carrying a native tool call (`task_start`, the Cerebellum talking to the runtime)
  opens no response: it is not speech. If one is already streaming, the unit may still close
  it, but it never starts one and never produces a transcript delta.

### 9.1 The Brain function call

`transfer_to_human` is the only function call Talkover ever emits, and
`src/talkover/realtime/brain_bridge.py` is what emits it (DESIGN.md 5.3, 6.3). The rules it
settles, all covered by `tests/protocol/test_function_call.py`:

- The `call_id` is the Brain's own: the client must echo it back verbatim in the
  `function_call_output`. `name` is `transfer_to_human`; `arguments` is the JSON object the
  Brain produced (`department`, and `reason` when it gave one), serialized without ASCII
  escaping so a Chinese reason stays readable.
- The call is emitted **whether or not** the client declared `transfer_to_human` in
  `session.tools`. The decision to hand the customer to a person is the Brain's, and an
  incomplete client tool list must not strand a caller who asked for a human; the omission
  is logged at `WARNING` once per session. `session.tools` therefore only needs to declare
  `transfer_to_human`: `query_ticket` and `query_order` run inside the Brain and are never
  exposed (§11).
- The `function_call_output` is routed to the Brain run's `respond()` as a
  `TaskInteractionReply`, which reaches `BrainSession.resolve()`. It is never upstream
  `task_resolve`, whose action set is an authorization decision (DESIGN.md 6.3). The
  `output` string is handed over unparsed.
- A pre-intercepted transfer (the keyword rule of DESIGN.md 6.4) produces exactly the same
  events with no LLM round; its `function_call_output` completes the task instead of
  continuing a loop.
- Brain `share`, `question` and the final result never become client events. They belong to
  the Cerebellum, which speaks them in its own words.
- A Brain failure is logged and the task ends with the provider's spoken fallback; it is not
  reported as `engine_error`, which would close the connection over a recoverable error.

## 10. Session lifecycle

- One session per process. A second connection is accepted, receives `error` with
  `code: "engine_busy"` and is then closed with code `1011`; the running session is
  unaffected. `GET /health` returns 503 while a session is running.
- After a disconnect the engine session is kept for `realtime.trailing_silence_sec`; a new
  connection presenting the same `session_id` resumes it, otherwise it is released (§10.1).

`GET /health` needs no authentication and answers with JSON:

```json
{"status": "idle", "engine": true, "asr": true, "brain": null, "session_active": false}
```

`status` is `idle` (HTTP 200), `busy` (503, a session holds the slot) or `not_ready` (503,
the engine is not loaded yet); `not_ready` wins over `busy`. `engine`, `asr` and `brain`
report each component's readiness, and `null` means the component is not configured, which
does not by itself make the service unhealthy. A session inside its reconnect window
(§10.1) still holds the slot, so it still reads `busy`.

### 10.1 Resuming a session after a disconnect

A dropped WebSocket does not end the session. The engine session and the whole protocol
state — session id, conversation id, the effective session object, conversation items, the
response in progress and the turn detector — are kept for `realtime.trailing_silence_sec`
(default 8 s, the same value upstream uses for trailing silence; `0` disables resuming).

**Presenting the id.** A reconnection names the session it wants in either place, and the
id is the `session.id` of the `session.created` the dropped connection received:

| Where | Form |
|---|---|
| Query parameter | `GET /v1/realtime?session_id=sess_…` (also with `?model=`) |
| Header | `X-Talkover-Session-Id: sess_…`, for a client whose URL is fixed by its dialler |

The query parameter wins when both are present. Nothing else identifies a session: there
is no resume token, because the Bearer key already authenticates the connection and the
process serves one session at a time.

**What a resumed connection reads first.** The same opening pair as a fresh one (§1), but
carrying what the dropped connection left behind:

1. `session.created` with the *same* `session.id` and the effective session as it stood —
   the truncated `instructions`, the negotiated formats, `turn_detection`. That the id
   comes back unchanged is what tells the client the resume worked.
2. `conversation.created` with the same `conversation.id`.
3. `session.updated`, only when a `session.update` had changed something before the
   disconnect, so a client can see its configuration survived without diffing.

Then the events the engine produced while nobody was connected, in order, and from there
the session behaves exactly as before the gap.

**Events produced while detached.** The model keeps running: it is mid-turn, and
`trailing_silence_sec` is what ends its turn. Those server events are buffered in order,
up to 256 of them; past that the *oldest* is dropped, counted and logged at `WARNING`,
because the end of what was missed is worth more to a resumed call than its beginning.
Nothing is discarded silently, but a resumed client can be missing a slice of one
response's audio, so it must not treat the replayed stream as gapless.

**Who is refused.** While a session is inside its window the slot is still taken, so any
connection that does not present its id — a different `session_id`, or none — gets the
same `error` with `code: "engine_busy"` and close `1011` that a second live connection
gets, and the waiting session is untouched. A `session_id` naming the session of a *live*
connection is refused the same way: an attached session cannot be taken over.

**When the window closes.** The session is released, the process resets its engine and
`/health` reports `idle` again. A connection presenting an expired — or simply unknown —
`session_id` is not an error: the id is ignored and a fresh session is created, with a new
id in `session.created`. A session that ended with a fatal error (`engine_error`, close
`1011`) is never kept: it is released at once, with no window.

## 11. Differences from Cascade

| Area | Cascade | Talkover |
|---|---|---|
| Auth | `REALTIME_API_KEY` | a single `server.api_key`; no per-tenant keys |
| Concurrency | many sessions per process | one session per process; a second connection gets `engine_busy` and `/health` returns 503 |
| Item lifecycle | `conversation.item.added` → `conversation.item.done`, never `.created` | `conversation.item.created` only; `.added` / `.done` are never emitted (DESIGN.md 5.3) |
| Turn detection | real VAD with exact boundaries | approximate (§7); Gander has no explicit VAD, so `speech_started` can be retroactive and the VAD parameters only reach the ASR side channel |
| `instructions` | a full system prompt for the LLM | written to the task slate only, truncated to `realtime.memory_slate_max_tokens`, truncated value echoed (§6) |
| Audio pre-processing | `noise_reduction` rejected unless `null` | `noise_reduction` accepted and echoed, with no effect; the same applies to `voice`, `speed` and `max_output_tokens`, which the Gander Talker does not honour |
| Audio formats | `audio/pcm` 24 kHz only | `audio/pcm` 24 kHz plus `audio/pcmu` / `audio/pcma` at 8 kHz on both directions, for direct telephony integration |
| Output modalities | `["audio"]` or `["text"]` | `["audio"]` only; there is no text-only response path |
| Tools | tools are effective and the gateway forwards every call | `tools` is validated and echoed, but only `transfer_to_human` is ever emitted to the client; `query_ticket` and `query_order` run inside the Brain and are never exposed. `tool_choice` is echoed and ignored — the Brain owns tool selection |
| `response.create` overrides | `instructions`, `output_modalities`, `voice`, `max_output_tokens`, `metadata` supported | only `conversation: "auto"` and `metadata` accepted; everything else rejected, because the model owns the turn and `response.create` is only a `flush_pending` |
| `conversation.item.create` | user, system and assistant text items | user text items and `function_call_output` only |
| `conversation.item.truncate` | rewinds the assistant audio and text | acknowledged only; the model context is not rewound |
| `conversation.item.retrieve` | rejected (no retained audio) | same rejection, same reason |
| Server error codes | `provider_error`, `transcript_timeout`, `input_audio_buffer_overflow`, `input_queue_overflow` | `engine_busy`, `engine_error` |
| `usage` | LLM text tokens | best-effort; audio tokens are `0` |

Cascade behaviours deliberately not adopted:

- `conversation.item.added` / `conversation.item.done` (Talkover emits `conversation.item.created`).
- `conversation.item.input_audio_transcription.delta`: Talkover emits only `.completed`, one
  per ASR segment.
- `response.output_text.*` and the text-response flow.
- Cascade's `semantic_vad` transcript-gated `response.created` ordering: Talkover starts a
  response when the model stops listening, not when a transcript is final.
- Per-response `voice` and the "no voice change after the first audio" rule: the Talker voice
  is fixed by the checkpoint.
- `parallel_tool_calls`, MCP tools, `conversation.item.retrieve`, `rate_limits.updated`.

### 11.1 Skipped Cascade cases

These Cascade client cases have no Talkover behaviour to assert: they exercise features
Talkover does not implement, or behaviours the table above deliberately does not adopt.
`tests/protocol/test_cascade_cases.py` carries one `pytest.skip` per row, with the same
reason, and machine-reads this table so the two cannot drift.

| Case | Why it cannot be ported |
|---|---|
| `cascade_server_error_codes` | provider_error, transcript_timeout, input_audio_buffer_overflow and input_queue_overflow do not exist in Talkover; the codes are engine_busy and engine_error (§8.1, §11). |
| `conversation_item_added_then_done` | Cascade's item lifecycle is .added then .done; Talkover emits conversation.item.created only (§11). |
| `conversation_item_retrieved` | conversation.item.retrieve is rejected because Talkover retains no input audio (§2), so the .retrieved event has no case. |
| `idle_timeout_triggered` | input_audio_buffer.timeout_triggered is in the never-emitted list of §9; idle_timeout_ms is rejected outright (§7). |
| `input_audio_transcription_delta` | Cascade streams transcription deltas; Talkover emits .completed only, one per ASR segment (§11). |
| `mcp_tool_events` | response.mcp_call* and mcp_list_tools.* are never emitted (§9, §11). |
| `multi_session_concurrency` | Cascade serves many sessions per process; Talkover is one session per process and a second connection gets engine_busy (§1, §10, §11). |
| `output_audio_buffer_events` | output_audio_buffer.* is WebRTC and SIP only; the WebSocket profile rejects the client event and never emits the server ones (§2, §9). |
| `parallel_tool_calls` | Not adopted: tools are session-level and the Brain owns selection (§11). |
| `per_response_voice_override` | Cascade supports a per-response voice and the 'no voice change after the first audio' rule; the Talker voice is fixed by the checkpoint (§11). |
| `per_tenant_api_keys` | Cascade authenticates per tenant; Talkover has a single server.api_key (§1, §11). |
| `rate_limits_updated` | rate_limits.updated is in the never-emitted list of §9. |
| `response_output_text_flow` | Cascade serves output_modalities ['text']; Talkover has no text-only response path (§3, §11). |
| `semantic_vad_transcript_gated_response_created` | Cascade orders response.created after a final transcript; Talkover starts a response when the model stops listening (§11, deliberately not adopted). |
| `transcription_session_updated` | Talkover serves no transcription sessions: the client event is rejected and transcription_session.updated is never emitted (§2, §9). |
| `truncate_rewinds_the_model_context` | Cascade rewinds assistant audio and text on conversation.item.truncate; Talkover acknowledges only (§2, §11). |
| `vad_prefix_padding_trims_the_committed_audio` | Cascade's VAD rewinds the committed buffer by prefix_padding_ms; Talkover validates and echoes the field without acting on it (§7). |
| `vad_threshold_boundary` | Cascade tunes a real VAD; Talkover's turn detection is approximate (§7, §11) and threshold only reaches the ASR side channel. |
