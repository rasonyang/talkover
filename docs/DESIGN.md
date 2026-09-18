# Talkover Design Document

> Version 0.2 · 2026-09-18 · Status: draft

## 1. Positioning and Naming

**Talkover** is an inference gateway that wraps the Gander full-duplex interaction model in an OpenAI Realtime compatible service, together with a Brain provider aimed at customer-service workloads. It runs on a single Apple Silicon machine.

One-line pitch: **Talk over it while it works.** The user can interrupt, ask follow-up questions, or change the request at any time, while background queries and actions keep running.

Naming notes:

| Item | Value |
| --- | --- |
| Project name | Talkover |
| PyPI package name | `talkover` (checked as available on 2026-09-16) |
| CLI | `talkover serve` / `talkover check` |
| Repository | `github.com/rasonyang/talkover` |
| Alternative names | Interject, Backchannel (also available on PyPI) |

## 2. Goals and Non-Goals

### Goals

1. **Dual-backend inference engine**: the same engine code runs on Apple M-series GPUs (MPS) and NVIDIA GPUs (CUDA). The development machine is an M3 Max with 128 GB; the test machine is a single A10 with 24 GB. MPS comes first, CUDA is validated afterwards.
2. **OpenAI Realtime compatible interface**: `ws://host/v1/realtime`, with clients using the GA protocol (`session.type = "realtime"`). The existing aicc FreeSWITCH → Realtime path needs no changes.
3. **Business Brain provider**: replaces Codex and provides three capabilities — ticket lookup, order lookup, and transfer to a human agent — integrated through Gander's `task_start / task_send / task_resolve` task lifecycle.

### Non-Goals (out of scope for phase one of this repository)

- Multi-session concurrency. One process serves one session; concurrency comes from multiple processes and an upstream scheduler.
- Training and data processing. The `minicpm_ft` training path stays on CUDA.
- Camera and screen sharing. The telephony scenario is audio only, `media_mode` is fixed to `voice`. The video channel keeps an interface but is not implemented.
- Admin API and multiple profiles. Configuration is read once at startup.
- MLX port. Reserved for phase two; trigger conditions are in section 10.

## 3. Overall Architecture

```
 SIP/WebRTC clients (aicc / FreeSWITCH / browser)
        │  OpenAI Realtime events (WebSocket, pcm16 24k)
        ▼
 ┌─────────────────────────────────────────────┐
 │  talkover.realtime  protocol translation    │
 │ session FSM · validation · resample · tools │
 └─────────────┬──────────────────▲────────────┘
               │ feed_pcm16       │ DuplexStepEvent
               ▼                  │
 ┌─────────────────────────────────────────────┐
 │  talkover.engine    Gander inference layer  │
 │  DuplexLiveSession (mcpmft) · backend · ASR │
 │   Thinker ─ Talker ─ token2wav   on MPS     │
 └─────────────┬──────────────────▲────────────┘
               │ task_start/send/resolve  │ share / question / result
               ▼                  │
 ┌─────────────────────────────────────────────┐
 │  talkover.brain     business Brain provider │
 │   LLM function-calling loop                 │
 │   query_ticket · query_order → business API │
 │   transfer_to_human → client function_call  │
 └─────────────────────────────────────────────┘
```

Process structure: a single process with three asyncio task groups. Model inference runs on a dedicated thread (`GanderDuplexSession` already has a `_model_lock`); the protocol layer and the Brain run on the event loop. ASR runs as a subprocess or a thread, depending on the backend.

## 4. Module One: Dual-Backend Inference Engine `talkover.engine`

### 4.0 Backend Matrix

| | MPS (development) | CUDA (test and deployment) |
| --- | --- | --- |
| Machine | M3 Max, 128 GB unified memory | A10, 24 GB VRAM, single card |
| Precision | bf16 | bf16; int8 Thinker when VRAM is insufficient (see 4.5) |
| Talker | Thread in the same process (opt-in this phase, see 4.2) | Thread in the same process; with multiple cards, the upstream detached mode can place it on a second card |
| ASR | mlx-whisper | faster-whisper CUDA float16, or CPU int8 to save VRAM |
| attention | sdpa | sdpa, optionally flash-attn |
| Concurrency | One session per process | One session per process; multiple processes across multiple cards |

The backend is selected by `engine.device`: `mps`, `cuda`, `cuda:<n>`, `cpu`. `cpu` is only used for CI smoke tests.

### 4.1 CUDA Bindings in the Upstream Code

| File | Binding | Handling |
| --- | --- | --- |
| `gander_runtime/cli.py` | Forces `detached_talker_device` to `cuda:<n>` | Talkover does not reuse that CLI; it builds settings itself |
| `mcpmft/infer/detached_talker.py` | `torch.cuda.device`, `get_rng_state`, `synchronize` | Implement a device backend abstraction inside Talkover; by default the Talker runs on a thread in the same process. On multi-card CUDA, keep the upstream detached mode as an option |
| `mcpmft/infer/common.py` | `torch.cuda.is_available()` check | Patch upstream, or monkeypatch inside Talkover before importing |
| `mcpmft/infer/asr.py` | faster-whisper (CTranslate2) has no Metal backend | Replace the ASR backend, see 4.3 |
| `serve.example.yaml` | Three-GPU mapping | Unified to `device: mps` |

Principle: **do not fork upstream**. Upstream is brought in as a path dependency, and CUDA bindings are bypassed through an adapter layer inside Talkover. Anything that cannot be bypassed is submitted upstream as a minimal patch.

### 4.2 Device Backend Abstraction

```python
class DeviceBackend(Protocol):
    name: str  # "mps" | "cuda" | "cpu"

    def is_available(self) -> bool: ...
    def device(self, index: int | None = None) -> torch.device: ...
    def synchronize(self) -> None: ...
    def rng_state(self) -> Any: ...
    def set_rng_state(self, state: Any) -> None: ...
    def autocast_dtype(self) -> torch.dtype: ...  # mps: bf16
```

- Two backend implementations: `MpsBackend` and `CudaBackend`; `CpuBackend` is for CI only. All device-related calls go through this interface, and the engine code contains no direct references to `torch.cuda` or `torch.mps`. `device(index=None)` returns the device the backend was constructed for, which is how a `cuda:<n>` selection is carried around; an explicit index selects another card of the same kind. `is_available()` reports whether the device is usable in the current process, and `talkover check` calls it — `get_backend()` itself does not check availability.
- By default the Thinker, Talker, and token2wav live on the same device. On MPS, unified memory means no copies; on a single CUDA card they share VRAM.
- The Talker is meant to run on a separate thread in the same process, separated by the backend's `synchronize`: `TalkerThread` drives a `TalkerRuntime` (upstream's `DetachedTalkerRuntime`, rebuilt on the configured device by `build_talker_runtime` because the upstream classmethod refuses a non-CUDA device), and `EngineSession` publishes its waveforms as `is_audio_chunk` events (5.3). It is **opt-in this phase**: `EngineSession(talker_factory=None)` is the default and keeps T1.4's in-step Talker, where the Talker and token2wav run inside the upstream step on the inference thread and the waveform rides the unit's own event. `talker_factory_from_config` becomes the default once T1.7 has validated the hand-over against real weights — the AR half of the Talker has never run with a checkpoint. On multi-card CUDA, `engine.talker_device` can point at another card; that branch is the M4 one and currently raises `NotImplementedError`.
- `attn_implementation` defaults to `sdpa`; on CUDA it can be set to `flash_attention_2`.
- Precision is bf16. On an M3 Max with 128 GB, Thinker + Talker + KV cache (128 units) is estimated to stay under 30 GB, so no quantization is needed. VRAM allocation on the A10 is covered in 4.5.

### 4.3 ASR Backend

In Gander, ASR is a side path: it produces the browser transcript and the trusted text instructions for the Brain, while the Cerebellum consumes raw audio directly. The ASR backend can therefore be swapped freely.

| Backend | Use |
| --- | --- |
| `mlx-whisper` large-v3-turbo | MPS default, Metal accelerated |
| `faster-whisper` CUDA float16 | CUDA default, matches upstream |
| `faster-whisper` CPU int8 | Option when A10 VRAM is tight; also used as a reference |
| SenseVoice ONNX | Candidate for Chinese telephony; evaluated in phase two |

The interface matches the upstream `asr_process`: it receives pcm16 16k segments and returns text with timestamps.

### 4.4 Real-Time Budget

The model processes one causal unit per second; the hard constraint is **end-to-end processing time per unit < 1.0 s**.

| Stage | Target |
| --- | --- |
| One Thinker step (audio encoding, control prediction, ≤ 11 speak tokens) | ≤ 0.45 s |
| Talker 50 S3 tokens + token2wav | ≤ 0.35 s |
| ASR and protocol layer | ≤ 0.10 s |
| Headroom | 0.10 s |

The first milestone only requires "it runs and real numbers are measured". If a single Thinker step on MPS consistently exceeds 0.6 s, the phase-two MLX + quantization path is triggered; see section 10. The A10's bf16 throughput is roughly one third of an A100, which is expected to meet real-time requirements, but needs to be measured at M4.

**token2wav on MPS (T1.5, M3 Max)**: the vocoder alone costs 0.49 s per unit at upstream's `n_timesteps=10`, 0.24 s at 5 and 0.14 s at 2, all float32 — the upstream default already exceeds the 0.35 s budgeted for Talker plus token2wav, with the AR stage excluded. `float16=True` aborts the process on MPS (MPSGraph dtype mismatch), so the vocoder stays float32 there. The flow-matching step count is the obvious knob: `engine.token2wav_timesteps` is **planned, not implemented** (section 9); the value is fixed at upstream's 10 in `build_talker_runtime`, and T1.8/T1.9 decides the setting after a quality check.

### 4.5 A10 Single-Card VRAM Budget

The upstream released configuration uses three cards; the A10 has only 24 GB, so the budget must be worked out first:

The table below is what `src/talkover/engine/memory.py` computes; it is a static estimate from the typed configuration, with no weights read and no model instantiated.

| Component | bf16 estimate | Source |
| --- | --- | --- |
| Thinker backbone + audio encoder | 16.75 GiB | `THINKER_BF16_BYTES`; `quantize_thinker: int8` scales it by 0.55 |
| Vision encoder | 1.00 GiB | `VISION_BF16_BYTES`; zero when `init_vision: false` |
| Talker + token2wav | 1.75 GiB | `TALKER_BF16_BYTES` |
| KV cache, 128 units + 1500 tokens of prior context | about 2 GiB (1.96 GiB) | 2 × 36 layers × 8 KV heads × 128 head_dim × 2 bytes per token, 100 tokens per unit assumed; to be pinned by M1 measurement |
| Activations and device runtime context | 1.25 GiB | `ACTIVATION_BYTES` |
| ASR (whisper large-v3-turbo, fp16) | 1.62 GiB weights + 0.90 GiB runtime | estimates, to be calibrated in M1; counted only when ASR shares the model's memory pool |

Verdict rule: `over_budget` when the total exceeds the budget, `borderline` when the total is at or above 85 % of the budget (`BORDERLINE_RATIO`), `ok` otherwise. The budget is 24 GiB on CUDA (single A10), the machine's physical memory on MPS and CPU.

On MPS the memory pool is unified, so setting `asr.device: cpu` does not reduce the budgeted total; on CUDA the same setting moves the ASR bytes out of VRAM entirely.

Conclusion: with `init_vision: false`, the model side comes to 21.71 GiB — `borderline` on a 24 GiB A10 — and adding fp16 ASR on the same card gives 24.23 GiB, which is `over_budget`. That verdict has only a 0.23 GiB margin and depends on the two ASR estimates above, so it must be re-checked once M1 measures them. Order of handling:

1. Put ASR on CPU int8 first, leaving only the Thinker and Talker on the card.
2. If it still overflows, quantize the Thinker's linear layers to int8 (`torchao` or `bitsandbytes`), and skip loading the vision encoder under `media_mode: voice` (`init_vision: false`).
3. If two A10s are available, run the Talker in detached mode on the second card and move ASR back to CUDA.

Before startup on CUDA, `talkover check` estimates VRAM from the configuration and emits a warning, so loading does not OOM halfway through.

### 4.6 The Cerebellum Channel

The Cerebellum (the model) drives the Brain, not the other way round: it opens a task with the native tool call `task_start` and then waits, and it phrases every Brain update in its own words. Two `EngineProtocol` methods carry that traffic back into the model, and `realtime/brain_bridge.py` is their only caller — nothing they carry is ever sent to the Realtime client.

| `EngineProtocol` | Upstream call | Payload | When |
| --- | --- | --- | --- |
| `feed_tool_response(response)` | `DuplexLiveSession.feed_tool_response` | the bounded front-brain result `gander_runtime.lean_realtime.control_tool_response` builds: `status`, `task_ids`, optional `reason` / `content` | exactly once per unit with `tool_response_expected: true`. The model produces no further unit until it arrives |
| `feed_worker_delivery(delivery)` | `DuplexLiveSession.feed_runtime_event` | the `worker_delivery` envelope `gander_runtime.lean_realtime.worker_delivery_response` builds: `type`, `task_name`, `topic`, `content`, plus `status` on a `final` | whenever the Brain emits a `share` or a `result` |

Both are prefilled inside `<tool_response>` markers together with the next unit's microphone audio, but they use different slots: upstream refuses a runtime event while a tool response is still pending, and refuses a tool response when no tool call is pending. Both run on the inference thread like every other command.

- `task_name` is the semantic name the model itself passed to `task_start`, never Talkover's internal `task_id` — upstream's training data validates a delivery's `task_name` against the task names the model has seen, so an id it never uttered reads as an unknown task.
- `topic` is `DeliveryRecord.topic` (`gander_runtime/coordination.py`), the closed set `milestone | interaction | final | risk | aggregate`. Talkover maps a Brain `share` to `milestone` and a Brain `result` to the terminal `final`; `final` is the only topic that may carry a `status`, which must be one of `completed | partial | failed | cancelled`.
- **Refusals are not terminal.** Neither call runs the model: upstream only validates the payload, tokenizes it and queues it, so a refusal (the wrong slot, or an envelope longer than `max_tool_response_tokens`) says nothing about the model's health. It is reported to the caller as `EngineError` and logged; the session keeps running. Losing one spoken update is a smaller failure than dropping a live call. This is the one exception to the rule that any exception on the inference thread is terminal.

Both calls are **unvalidated against a real model run**, for the same reason `submit_text_turn` is (5.3): no checkpoint has been loaded yet. They land with the `mps` acceptance of T1.4.

## 5. Module Two: Realtime Compatible Interface `talkover.realtime`

Protocol decisions (T2.1). `docs/protocol-profile.md` is the detailed reference — per-field status, rejection codes and the full server-event list. The decisions worth recording here:

- `response.create` rejects every per-response override except `conversation: "auto"` and `metadata`; `instructions`, `output_modalities`, `max_output_tokens`, `audio`, `input`, `tools` and `tool_choice` are rejected, because the model owns the turn and `response.create` is only a `flush_pending`. To be confirmed against the aicc client's actual payload.
- `output_modalities: ["text"]` is rejected; there is no text-only response path.
- `input_audio_noise_reduction` is accepted and echoed but has no effect.
- `voice`, `speed` and `max_output_tokens` are echoed and ignored; the Gander Talker does not honour them.
- Only `conversation.item.created` is emitted; `conversation.item.added` and `conversation.item.done` never are.
- The effective session echoed in `session.created` starts from the GA defaults: `audio.output.voice` is `"gander"`, echo-only because the Talker voice is fixed by the checkpoint, and `audio.input.turn_detection` is `server_vad` **on** (`threshold: 0.5`, `prefix_padding_ms: 300`, `silence_duration_ms: 200`). An explicit `null` in `session.update` clears it (5.4).
- The `instructions` cap is enforced with a character heuristic (one token per CJK character, one per four other characters) because the protocol layer has no tokenizer. It is a safety bound on the slate, not an exact token budget.

### 5.1 Endpoint and Authentication

- `GET /v1/realtime?model=<any>` upgrades to a WebSocket. The `model` parameter accepts any value and is echoed back.
- `Authorization: Bearer $TALKOVER_API_KEY` is the only accepted credential. A single key only; there is no query-parameter key, no subprotocol key and no per-tenant key. A missing, malformed or wrong header is refused with HTTP `401` **before** the WebSocket handshake is accepted, so a rejected client never sees an `error` event.
- Plain `ws://`; TLS is terminated by an upstream reverse proxy.
- `GET /health` needs no authentication and answers with JSON (`docs/protocol-profile.md` §10):

```json
{"status": "idle", "engine": true, "asr": true, "brain": null, "session_active": false}
```

  `status` is `idle` (HTTP `200`), `busy` (`503`, a session holds the only slot) or `not_ready` (`503`, the engine is not loaded yet); `not_ready` wins over `busy`. `engine`, `asr` and `brain` report each component's readiness, and `null` means the component is not configured, which does not by itself make the service unhealthy.
- A second WebSocket connection is accepted, receives an `error` event with `code: "engine_busy"` and is then closed with WebSocket code `1011`; the running session is unaffected.

### 5.2 Audio

- `pcm16` externally, 24 kHz mono for both input and output. Input is resampled to 16 kHz for the model; output comes straight from the model at 24 kHz.
- `g711_ulaw` / `g711_alaw` are supported for input and output, which makes direct telephony integration easier.
- `input_audio_noise_reduction` is accepted and echoed in `session.updated` but has no effect; no audio pre-processing is performed.

### 5.3 Event Mapping

| Client → Server | Internal behavior |
| --- | --- |
| `session.update` | Update tools, voice, and audio formats; `instructions` is written into the task slate (see 5.5) |
| `input_audio_buffer.append` | Decode, resample, and `feed_pcm16` in 1 s units |
| `input_audio_buffer.commit` | The incomplete unit held in the protocol layer's framer is zero-padded and sent with `feed_pcm16` first, then `flush_pending`; the committed user item takes the turn detector's pending `item_id` when a speech span preceded it (5.4) |
| `input_audio_buffer.clear` | Drop the buffer for the incomplete unit |
| `response.create` | Accepted; the same zero-padded `feed_pcm16` followed by `flush_pending` as a commit; the model decides on its own whether to speak |
| `response.cancel` | `interrupt_output` |
| `conversation.item.create` (`function_call_output`) | Routed to the Brain as an interaction reply: the realtime layer (T3.7) builds a `TaskInteractionReply` and calls the run's `respond()`, which reaches `BrainSession.resolve()`. Never upstream `task_resolve`, whose `TaskResolveAction` is an authorization decision (`cancel | allow_once | allow_session | deny`), not a tool output (see 6.3) |
| `conversation.item.create` (`message`, text) | Fed into the runtime as a text user turn, not into the model's audio channel: `EngineSession.submit_text_turn` calls upstream `DuplexLiveSession.feed_runtime_event({"type": "user_text", "content": ...})`, which is prefilled inside `<tool_response>` markers alongside the next unit's microphone audio |
| `conversation.item.truncate` / `delete` | Accepted and acknowledged; does not affect model context |

| Internal event | Server → Client |
| --- | --- |
| First occurrence of a unit with `is_listen: false` | `response.created`, `response.output_item.added`, `conversation.item.created` |
| A unit with `is_tool_call: true` | Nothing. The Cerebellum is talking to the runtime, not to the customer: the unit carries no speech, so it opens no response. It goes to `brain_bridge.on_engine_event` instead, and is answered with `feed_tool_response` (4.6). A tool-call unit arriving while a response is already streaming still closes that response |
| `DuplexStepEvent.text` | `response.output_audio_transcript.delta` |
| `DuplexStepEvent.audio_waveform` | `response.output_audio.delta` |
| `EngineStepEvent(is_audio_chunk=True)` | `response.output_audio.delta` for the unit its `unit_id` / `generation_id` name. With the Talker thread (4.2) the waveform is vocoded after the Thinker step that produced it has already been reported, so it arrives on its own event; it may only extend a response that is still open |
| `end_of_turn: true` | `response.output_audio.done`, `transcript.done`, `content_part.done`, `output_item.done`, `response.done` with `status: "completed"` |
| `interrupted: true` | The same close chain with `status: "cancelled"`: the assistant item is marked `incomplete`, `conversation.item.truncated` reports the audio already sent, and `response.done.status_details` is `{"type": "cancelled", "reason": ...}` — `client_cancelled` after a `response.cancel`, `turn_detected` for a model barge-in. `response.usage` is best-effort on every close: a character heuristic for the text tokens, audio tokens always 0 |
| ASR segment completed | `conversation.item.input_audio_transcription.completed` |
| Brain emits `transfer_to_human` | `response.function_call_arguments.delta` + `.done` (a single complete delta) |
| Brain `share` / `result` | Not exposed directly: handed to the Cerebellum with `EngineProtocol.feed_worker_delivery` (4.6) as a `milestone` / `final` worker delivery, and turned into speech by the model in its own words |

Upstream has no API for a text user turn: `DuplexLiveSession` exposes `feed_pcm16`, `flush_pending`, `interrupt_output`, `set_task_slate`, `feed_tool_response` and the generic `feed_runtime_event`. The `{"type": "user_text"}` envelope above is the only available carrier and is **unvalidated against a real model run** — no checkpoint has yet confirmed that the Thinker treats it as a user turn rather than as a tool response. Section 11 records this as an open risk.

### 5.4 Turn Detection

Gander has no explicit VAD boundary. `turn_detection` accepts `server_vad` and `semantic_vad`, both mapped to the same implementation (`turn_detection.py`, profile §7). `TurnDetector` is pure: it returns the server events it wants emitted, and `RealtimeSession` owns the sink.

- `speech_started`: the ASR backend's `EnergyVad` opens a speech span. A unit with `interrupted: true` opens one retroactively — a barge-in is the model saying, after the fact, that the user had started talking — anchored on the engine's unit clock at `(unit_index - 1) * 1000` ms. A late Talker audio chunk (`is_audio_chunk`) never opens a span.
- `speech_stopped`: the end of the ASR segment. `audio_end_ms` is clamped up to `audio_start_ms`, because the ASR timeline and the unit clock are two approximations of the same instant.
- `item_id`: both events must name the user item the audio ends up in, but that item is only created at `input_audio_buffer.commit`, after the span. The detector mints the id at `speech_started` and the next commit takes it; an uncommitted span leaves an unused id behind, and two spans before one commit leave the last span's id on the item.
- The tuning fields (`threshold`, `prefix_padding_ms`, `silence_duration_ms`, `eagerness`) are validated and echoed but change nothing here; only the ASR side channel could act on them. `create_response`, `interrupt_response` and `idle_timeout_ms` are accepted and ignored — the model decides on its own, and `input_audio_buffer.timeout_triggered` is never emitted.

The documentation states explicitly that these are approximate semantics. When `turn_detection: null`, neither event is synthesized.

### 5.5 Instructions

The Cerebellum's system prefix is assembled in `mcpmft/prompts.py` and does not accept an arbitrary runtime system prompt. The first version writes `session.instructions` into `set_task_slate`, with a length cap of `memory_slate_max_tokens` (256 tokens by default). Over-long values are truncated, and the truncated value is echoed in `session.updated`. Business phrasing is carried mainly by the Brain's system prompt rather than by the Cerebellum.

### 5.6 Protocol Validation

Field paths, rejection codes, and error event formats follow the rules in Cascade's `docs/protocol-profile.md` directly. Cascade's Realtime client test cases serve as Talkover's protocol regression suite.

### 5.7 Session Lifecycle

- Only one session at a time is allowed per process. A second connection receives an `error` event with `code: "engine_busy"` and is then closed with WebSocket code `1011`; `GET /health` answers `503` with `status: "busy"` so a scheduler can probe it (5.1).
- After a disconnect the session is kept for `realtime.trailing_silence_sec` (profile §10.1). A reconnection names it with `?session_id=sess_…` or the `X-Talkover-Session-Id` header, the query parameter winning when both are present; the Bearer key already authenticates the connection, so there is no resume token. The slot stays busy for the whole window: any other connection — a different id, or none — still gets `engine_busy`, and an id naming a session that is *live* is refused the same way.
- A resumed connection reads the same opening pair carrying what the dropped one left behind (the same `session.id` and `conversation.id`, the effective session, and `session.updated` only when a `session.update` had changed something), then the events produced while nobody was attached. That backlog is bounded at `DETACHED_EVENT_LIMIT` (256) events; past it the *oldest* is dropped, counted on the sink and logged at `WARNING`, so a resumed stream may have a gap but never a silent one.
- When the window closes, `SessionSlot` closes the runner and awaits the `on_release` hook of `create_app`; it never calls `engine.stop()` itself, and wiring that hook to an engine reset is T2.9. An expired or unknown `session_id` is not an error: it is ignored and a fresh session is created. A session that ended with a fatal error is released at once, with no window.
- `DuplexLiveConfig.stop_on_turn_end` is forced to `false`. A Realtime session is driven by the client's audio stream and by `/v1/realtime` closing, so the engine must not end itself when the model finishes a turn. The value is not configurable.

## 6. Module Three: Business Brain Provider `talkover.brain`

### 6.1 Integration Point

The Cerebellum only emits `task_start(name)`, `task_send(main|fork)`, and `task_resolve`. The Brain receives a `TaskRequest` (trusted user turn text + task name + bounded context) and must produce a stream of `ProviderEvent`s. Talkover implements the `WorkerProvider` protocol from `gander_runtime` and registers it through `ProviderFactoryRegistry` under the key `business`.

**Who builds the `TaskRequest` (T3.7).** In a Gander deployment the Gateway compiles a `WorkerRequest` from its own ledger and the provider is reached through `ProviderFactoryRegistry`. Talkover runs the Brain in the same process with no Gateway, so `realtime/brain_bridge.py` builds the `TaskRequest` itself and calls `BusinessProject.open_run(request, autostart=True, intercept=True)` — the entry point T3.4 documents for exactly this caller, and also where the 6.4 keyword pre-interception runs. There is no `WorkerRequest` to compile and no `WorkerRunChannel` to unwrap.

| `TaskRequest` field | Filled from |
| --- | --- |
| `instruction` | the latest **trusted ASR text** (4.3), because that is what the customer actually said and what the 6.4 keyword rule matches. With no transcript yet — ASR is a side channel and may lag or be disabled — the semantic task name is used instead, which still carries the intent in the model's own words |
| `context` | the last 8 trusted transcripts as `ContextEvent`s (`kind="audio_transcript"`, with their `start_ms` / `end_ms`) |
| `metadata` | `{"task_name": <the name the model passed to task_start>, "realtime_session_id": <session id>}` — the task name is metadata, not the instruction, and it is also what the `worker_delivery` envelope addresses the model with (4.6) |
| `task_id` | a fresh `new_id("task")`; `session_id` is the Realtime session id and `generation` is 1 |

The Cerebellum's `task_start` arrives at the bridge as an `EngineStepEvent` with `is_tool_call` set (5.3), and is answered with `feed_tool_response` (4.6). `task_send` / `task_resolve` are recognised and logged but not yet acted on.

### 6.2 Internal Structure

```
TaskRequest ──▶ BrainSession
                 ├── system prompt (business phrasing + tool descriptions)
                 ├── LLM function-calling loop (at most max_rounds rounds)
                 │     ├── query_ticket(ticket_id | phone)   → business HTTP API
                 │     ├── query_order(order_id | phone)     → business HTTP API
                 │     └── transfer_to_human(department, reason)
                 │           → not executed, emits ProviderEvent(kind="interaction")
                 ├── share(text)     → intermediate result fed back to Cerebellum
                 └── TaskResult      → final result
TaskUpdate (task_send main) ──▶ appended to the conversation of the same BrainSession
TaskQuery  (task_send fork) ──▶ read-only copy, does not modify main task state
```

**Three levels (T3.4).** Upstream `gateway.py:188` defines `WorkerProvider` as `name`, `capabilities`, `open_project(ProjectRecord) -> WorkerProject`, `close()`; `WorkerProject` (`gateway.py:178`) has `start(WorkerRequest, WorkerControl) -> WorkerRun`; `WorkerRun` (`gateway.py:162`) has `events()`, `send()`, `cancel()`, `close()`. The Codex provider is the reference shape, with the per-task object `_CodexRun` (`providers/codex.py:836`). Talkover mirrors it:

| Level | Talkover object | Holds |
| --- | --- | --- |
| Provider | `BusinessProvider`, registered under the key `business` | the config, the shared `httpx` client, `BackendCapabilities` |
| Project | one per `ProjectRecord` | nothing device-bound; it only constructs runs |
| Run | one per task, owns one `BrainSession` | the conversation, the pending interaction id, the `ProviderEvent` queue |

**Event mapping.** `BrainSession` yields neutral events (`Share`, `Question`, `ClientTool`, `Result`) and never imports `gander_runtime`; the run translates them:

| Brain event | `ProviderEvent` |
| --- | --- |
| `Share` on the main lane | `kind="share"`, `ShareEvent(kind="important")` |
| Any event on the fork lane (a `TaskQuery` answer), `Result` included | `kind="share"`, `ShareEvent(kind="answer")` — the kind the Codex provider uses for a side-query answer (`providers/codex.py:1565`) — carrying `state_patch={"_gander": {"lane": "query", "request_id": …}}`. The fork lane never emits `kind="result"`: upstream `WorkerRunChannel` would read it as a `DonePayload` and terminate the run. A `ClientTool` raised by a fork is dropped with a warning |
| `Question` | `kind="share"`, `ShareEvent(kind="need_input")` |
| `ClientTool` (`transfer_to_human`) | `kind="interaction"`, `TaskInteraction(kind="user_input")` — see 6.3 |
| `Result(ok=True)` | `kind="result"`, `TaskResult(status="completed")` |
| `Result(ok=False)` | `kind="result"`, `TaskResult(status="failed")` |

`ShareEvent` and `TaskResult` also carry `task_id` / `session_id` / `generation`, which live on the `TaskRequest`; `BrainSession` never sees them, so the run fills them in.

**Run lifecycle.** `WorkerProject.start` returns a `WorkerRunChannel` wrapping a `BusinessRun`; the duck type the channel calls is `events`, `steer`, `query`, `respond`, `cancel`, `close`. A `TaskUpdate` with `mode="replace"` is treated as additive — a spoken turn cannot retract what the customer already heard, so the text is appended like any other update; `mode="cancel"` closes the run. `BackendCapabilities.steering` is `next_turn`, because `BrainSession.update` appends and runs a fresh bounded loop and cannot interrupt an LLM call in flight; `side_queries` is `isolated_fork`. An exception escaping the loop does not kill the provider: it becomes `ProviderEvent(kind="error")` and the run stays usable.

**Registration.** `ProviderFactoryRegistry` takes `BusinessProviderSettings` (`config_path`, `max_rounds`, `spoken_numbers`) as the settings type for the key `business`: the Brain is configured by Talkover's own `brain` section, so `worker.settings` only says which file to read plus the two knobs a deployment is likely to override. The provider owns the shared `BrainLLM` and `ToolExecutor` — one `httpx` client each — and hands them to every run's `BrainSession`.

**Loop bounds.** `brain.max_rounds` (default 6) caps the LLM calls *per invocation* — one `start`, one `update`, one `query` or one `resolve` each get their own budget. Six covers the two lookups a customer-service turn needs plus a retry. Exhausting the budget ends the invocation with `Result(ok=False)` carrying `fallback_texts.round_cap`, never with another round. A `LLMError` marked retryable is retried exactly once inside the same round; a second failure, or a non-retryable error, ends the invocation with `Result(ok=False)` carrying `fallback_texts.failure`.

**Config mapping.** The `brain` section (section 9) feeds the `BrainSession` constructor; a `fallback_texts` key left null keeps the built-in English constant in `src/talkover/brain/session.py`:

| Config key | `BrainSession` argument | Default |
| --- | --- | --- |
| `brain.max_rounds` | `max_rounds` | `DEFAULT_MAX_ROUNDS` (6) |
| `brain.spoken_numbers` | `spoken_numbers` | `False` |
| `brain.fallback_texts.round_cap` | `round_cap_text` | `ROUND_CAP_RESULT_TEXT` |
| `brain.fallback_texts.failure` | `failure_text` | `FAILURE_RESULT_TEXT` |
| `brain.fallback_texts.no_answer` | `no_answer_text` | `NO_ANSWER_RESULT_TEXT` |

### 6.3 Where the Three Tools Execute

| Tool | Execution location | Rationale |
| --- | --- | --- |
| `query_ticket` | Brain calls the business API internally | Data lookup; the Brain turns the result into spoken form and then calls `share`, which the bridge hands to the Cerebellum as a `milestone` worker delivery (4.6) |
| `query_order` | Same as above | Same as above |
| `transfer_to_human` | Sent back to the Realtime client as a `function_call`; the client executes it and returns a `function_call_output` | Transfer is a SIP action and belongs to aicc / FreeSWITCH |

Externally, `session.tools` only needs to declare `transfer_to_human`. The lookup tools are not exposed.

**How `transfer_to_human` is carried (decision, T3.4).** Upstream `ProviderEvent.kind` (`gander_runtime/contracts.py:268`) is the closed `Literal` `share | interaction | interaction_resolved | result | error`. There is no `client_tool` member, and upstream is never forked, so the call travels as a `TaskInteraction` (`contracts.py:211`) emitted as `ProviderEvent(kind="interaction", generation=..., interaction=...)`, exactly as the Codex provider raises a permission request (`providers/codex.py:1362`):

| `TaskInteraction` field | Value for `transfer_to_human` |
| --- | --- |
| `task_id`, `session_id`, `generation` | copied from the `TaskRequest` the run was started with |
| `interaction_id` | `new_id("interaction")`; the run keeps the mapping to the Brain's `call_id` |
| `kind` | `"user_input"`. `InteractionKind` (`contracts.py:34`) is `approval \| user_input`; this is not an authorization prompt, it is an action the client performs and reports back |
| `prompt` | the spoken reason, so a client without tool support can still render something |
| `choices`, `questions` | empty: the client executes a function call, it does not pick an option |
| `metadata` | `{"tool": "transfer_to_human", "call_id": <Brain call id>, "arguments": {"department": ..., "reason": ...}}` — the realtime layer reads `arguments` to build `response.function_call_arguments.delta` / `.done` |

The reply path is the mirror image. The client's `conversation.item.create` with a `function_call_output` item becomes a `TaskInteractionReply` (`contracts.py:227`) carrying the same `task_id` / `session_id` / `generation` / `interaction_id` and the output text in `text`; the realtime layer (T3.7) hands it to the run's `respond(reply)` (`providers/codex.py:983`), which validates the ids and the generation, then calls `BrainSession.resolve(output, call_id=...)` and emits `ProviderEvent(kind="interaction_resolved", interaction_id=..., interaction_resolution="response")` before the loop continues. The `TaskResult` the resumed loop ends with does not go to the client either: it reaches the Cerebellum as the terminal `final` worker delivery (4.6), and the model speaks it.

Upstream `task_resolve` is **not** that path. Its `TaskResolveAction` (`gander_runtime/coordination.py:66`) is `cancel | allow_once | allow_session | deny`: an authorization decision taken by the front brain about a pending interaction, not a place to deliver a tool result.

### 6.4 Pre-Interception of Transfer to Human

Running the full Brain loop adds two to three seconds of latency. `TransferInterceptor` (`intercept.py`) matches the trusted text of a `task_start` against `brain.transfer_keywords` (such as "转人工", "人工客服", "找个人") before the loop starts; on a hit `BusinessProject.open_run` emits the `transfer_to_human` interaction straight away instead of calling `run.start()`, and the LLM is never reached. This is a pure rule: no I/O, no session state, no `gander_runtime` import.

Matching is substring matching on a normalised form, applied to the keywords as well as to the text: NFKC first (full-width forms folded onto ASCII), then case folding, then every whitespace, control, punctuation and symbol character dropped (Unicode general categories `Z`, `C`, `P`, `S`). So "转人工！", "转人工, 谢谢" and "转 人 工" all hit, and a keyword that normalises to nothing is dropped rather than allowed to match every utterance. The emitted call carries only `department: "general"`: the tool schema is closed (`additionalProperties: false`), and a `reason` would become the interaction's `prompt`, which a client without tool support speaks back to the customer — an English sentence mid-call is worse than the provider's own fallback. The matched keyword is reported on `TransferMatch` for logging instead. Measured cost from `task_start` to the emitted interaction: median 0.03 ms, max 0.15 ms over 50 in-process runs (T3.5).

### 6.5 LLM Choice

The task is simple, the context is short, and the requirements are low latency and stable parameter extraction.

| Option | Scenario |
| --- | --- |
| DeepSeek 4.1 Flash via its Anthropic-compatible API | Default, low latency |
| Claude Sonnet 5 via Anthropic API | When phrasing quality matters |
| Self-hosted 7B–14B with function calling (MLX locally or vLLM) | On-premises deployment; GPU contention must be evaluated when co-located with Gander |

The LLM client is abstracted as a `BrainLLM` protocol, with Anthropic and OpenAI-compatible implementations in the first version.

- The system prompt is a separate `system` argument of `BrainLLM.complete`, not a message in the conversation; the OpenAI-compatible client is the one that folds it into a leading `system` message.
- Both clients are implemented directly on `httpx`. No provider SDK is used.
- `timeout_sec` (30) and `max_tokens` (1024) are constructor defaults and are not yet exposed in the YAML configuration.

### 6.6 Business API Interface

`query_ticket` and `query_order` call user-configured endpoints over HTTP. Requests and responses are JSON, the timeout is 3 s (`brain.business_api.timeout_sec`), and on failure the Brain explains the situation to the user and suggests a transfer to a human agent. The first version ships a mock server for development and demos.

Endpoint contract (`src/talkover/brain/tools.py`):

| Tool | Request | Response |
| --- | --- | --- |
| `query_ticket` | `GET {base_url}/tickets` with `ticket_id` and/or `phone` as query parameters; at least one is required | any 2xx JSON object, passed through to the LLM unchanged |
| `query_order` | `GET {base_url}/orders` with `order_id` and/or `phone` as query parameters; at least one is required | any 2xx JSON object, passed through to the LLM unchanged |
| `transfer_to_human` | no HTTP call | a successful result marked `client_tool`, carrying the validated arguments for the protocol layer to emit as a `function_call` |

A 2xx body that is not a JSON object is wrapped as `{"result": <body>}`. Execution never raises; every failure is a result with one of these `error_kind` values:

| `error_kind` | Cause |
| --- | --- |
| `timeout` | the business API did not respond within `timeout_sec` |
| `http_error` | a non-2xx status |
| `unavailable` | connection failure, or a 2xx body that is not JSON |
| `invalid_arguments` | a non-string argument, a missing required argument, or neither identifier nor phone given |
| `unknown_tool` | a tool name outside the three above |

**Mock server (T3.6).** `uv run talkover mock-api` serves the same contract on `127.0.0.1:9100` by default (`--host` / `--port`, or `-c <config>` to take the address from `brain.business_api.base_url`), which is also `brain.business_api.base_url` in both example configs. Shapes:

- A lookup by id returns the single matching record as a JSON object.
- A lookup by phone returns an object, never a bare list, so the body stays a JSON object the LLM receives unchanged: `{"phone": "13800138000", "count": 2, "tickets": [...]}` for `/tickets` and the same shape with `"orders"` for `/orders`. `count` is `0` and the list empty when the phone is unknown; that is a `200`, not a `404`.
- An unknown id is `404` with the body `{"error": "not_found", "detail": ...}`.
- A request with neither parameter is `422` with `{"error": "invalid_request", ...}`.
- `GET /healthz` reports the server is up.
- The fixture records are runtime data, so their text fields are Chinese. An artificial delay (`--delay-sec`, the `X-Mock-Delay` header or the `_delay` query parameter) exercises the 3 s timeout path.

## 7. Repository Layout

```
talkover/
├── pyproject.toml
├── README.md
├── docs/
│   ├── DESIGN.md                 # this document
│   ├── protocol-profile.md       # ported from Cascade, with differences noted
│   └── mps-porting-notes.md      # how each CUDA binding point was handled
├── configs/
│   ├── serve.example.yaml
│   └── brain.business.example.yaml
├── src/talkover/
│   ├── __init__.py
│   ├── cli.py                    # talkover serve / check / bench
│   ├── config.py
│   ├── app.py                    # process composition: engine <-> realtime <-> brain
│   ├── engine/
│   │   ├── backend/              # DeviceBackend protocol; mps.py, cuda.py, cpu.py
│   │   ├── memory.py             # pre-startup VRAM/memory estimation
│   │   ├── session.py            # wraps DuplexLiveSession
│   │   ├── talker.py             # in-process Talker thread
│   │   ├── asr/                  # mlx_whisper / faster_whisper / sensevoice
│   │   └── patches.py            # minimal upstream monkeypatches, kept in one place
│   ├── realtime/
│   │   ├── server.py             # FastAPI + WebSocket
│   │   ├── session.py            # Realtime session state machine
│   │   ├── events.py             # event dataclasses and validation
│   │   ├── mapping.py            # DuplexStepEvent → Realtime events
│   │   ├── audio.py              # resampling, g711 codec
│   │   ├── brain_bridge.py       # Realtime <-> Brain, and the 4.6 channel back
│   │   └── turn_detection.py
│   └── brain/
│       ├── provider.py           # WorkerProvider implementation and registration
│       ├── session.py            # function-calling loop
│       ├── llm/                  # anthropic / openai_compat
│       ├── tools.py              # schemas and execution for the three tools
│       ├── intercept.py          # transfer-to-human pre-interception
│       └── mock_api.py           # business API for demos
├── tests/
│   ├── protocol/                 # protocol cases ported from Cascade
│   ├── engine/                   # MPS smoke tests and real-time benchmarks
│   └── brain/
└── scripts/
    ├── bench_rtf.py              # per-unit processing time measurement
    └── download_models.sh
```

## 8. Environment and Dependencies

| Item | Value |
| --- | --- |
| Python | 3.12 (local `~/.local/bin/python3.12`) |
| Package manager | `uv` |
| torch | `>=2.6,<2.7` (constraint from upstream `mcpmft`; MPS support is sufficient) |
| transformers | `==4.51.0` (upstream constraint) |
| Upstream dependencies | `mcpmft[talker]` and `gander-runtime` brought in as path dependencies; the `asr` extra is not installed |
| ASR | `mlx-whisper` (extra `mps`), `faster-whisper` (extra `cuda`) |
| Quantization | `torchao` (extra `cuda`, for the A10) |
| Other | `fastapi`, `uvicorn`, `websockets`, `httpx`, `numpy<2` (both LLM clients speak HTTP directly; no provider SDK is installed) |
| torchaudio | `>=2.6,<2.7`, needed by `stepaudio2`: 2.11 resolves against torch 2.6 but its C++ extension fails to load (`_aoti_torch_abi_version`) |
| setuptools | `<81`: `librosa` 0.9, pinned through the upstream extras, imports `pkg_resources`, which setuptools removed in 81 |
| onnx | `<1.18`: 1.19, pulled in by `s3tokenizer`, needs `ml_dtypes>=0.5.1` and through it `numpy>=2`, which upstream forbids |
| deepspeed | excluded on macOS via `[tool.uv] override-dependencies` (`deepspeed; sys_platform == 'linux'`): 0.19.x cannot be imported under torch 2.6, and transformers imports it whenever it is installed. Upstream needs it only for `mcpmft.train.*`, which Talkover never imports. See `mps-porting-notes.md`, 2026-09-18 |

`talkover check -c <config>` reports the whole environment statically: nothing is loaded, no socket is opened and no API is called, so it is usable while the weights are still downloading. Every check prints one `ok` / `warn` / `fail` line (`src/talkover/check.py`), and the exit code is:

| Exit code | Meaning |
| --- | --- |
| 0 | no check failed; warnings alone do not change it |
| 1 | at least one check failed |
| 2 | the config file is missing, unreadable or invalid, so no check was run |

Note: upstream's `environment.yml` uses conda with Python 3.10; Talkover does not use conda. Any upstream code that is incompatible with 3.12 is recorded in `mps-porting-notes.md` and submitted as a patch.

### 8.1 Continuous Integration

`.github/workflows/ci.yml` runs on push and pull request to `main`, on `ubuntu-latest`: `uv sync --extra dev`, `ruff check .`, `ruff format --check .`, `uv run pytest -m cpu`, and `uv run pytest tests/protocol tests/brain`. Neither the `cuda` nor the `mps` extra is installed; both only add an ASR backend that the GPU-free suites do not exercise.

CI satisfies the `mcpmft` / `gander-runtime` path dependencies by checking out `Omni-Interaction-Agent` at the pinned commit `cf43838` rather than vendoring a stub. The repository is public, so no token is required. A stub was rejected because it would have to track the upstream `WorkerProvider` and `DuplexLiveSession` signatures by hand, and a drift between the stub and the real package would make CI green while the real environment fails.

`actions/checkout` writes into `$GITHUB_WORKSPACE`, so the workflow checks Talkover out into `talkover/` and upstream into `Omni-Interaction-Agent/`, making them siblings; every subsequent step uses `working-directory: talkover` so that the relative path `../Omni-Interaction-Agent` in `pyproject.toml` resolves.

`pytest -m cpu` exits with code 5 when nothing is collected. The workflow does not mask that exit code: at least one `cpu`-marked test must exist, and an empty selection is treated as a CI failure.

## 9. Configuration Example

```yaml
server:
  listen: 127.0.0.1:8000
  api_key: ${TALKOVER_API_KEY}

engine:
  device: mps                # or cuda:0
  talker_device: null        # can be cuda:1 on multi-card CUDA
  dtype: bfloat16
  quantize_thinker: null     # set to int8 when A10 VRAM is insufficient
  init_vision: false         # voice mode does not load the vision encoder
  base_model: /Users/rasonyang/models/MiniCPM-o-4_5
  thinker_checkpoint: /Users/rasonyang/models/Gander/thinker
  talker_checkpoint: /Users/rasonyang/models/Gander/talker
  context_max_units: 128
  sliding_window_mode: context_slate
  media_mode: voice
  attn_implementation: sdpa  # flash_attention_2 is accepted on CUDA
  token2wav_dir: null        # null resolves to <base_model>/assets/token2wav
  ref_audio_path: null       # null keeps the Talker checkpoint's own voice

realtime:
  memory_slate_max_tokens: 256   # cap on session.instructions written into the task slate
  trailing_silence_sec: 8.0      # upstream default; also the reconnect window (5.7)

asr:
  backend: auto              # mps→mlx_whisper, cuda→faster_whisper
  model: large-v3-turbo
  device: auto               # set to cpu when A10 VRAM is tight
  compute_type: auto

brain:
  provider: business
  llm:
    kind: anthropic
    base_url: https://api.deepseek.com/anthropic
    api_key: ${DEEPSEEK_API_KEY}
    model: deepseek-flash
  business_api:
    base_url: http://127.0.0.1:9100
    timeout_sec: 3
  # Chinese phrases the customer says, e.g. "transfer to a human", "human agent", "get me a person"
  transfer_keywords: ["转人工", "人工客服", "找个人"]
  spoken_numbers: false      # rewrite digits into segmented spoken form before speaking (11)
  max_rounds: 6              # LLM rounds per Brain invocation (6.2)
  # Spoken to the customer when the loop cannot answer; null keeps the built-in English text
  fallback_texts:
    round_cap: "抱歉，这个问题我这边暂时查不到，帮您转接人工客服。"
    failure: "抱歉，系统暂时没有响应，帮您转接人工客服。"
    no_answer: "抱歉，我没有听清，您可以再说一遍吗？"
```

`engine.ref_audio_path` is optional only while the in-step Talker is the default (4.2): upstream keeps the checkpoint's own voice when it is null, which is what `serve.example.yaml` ships. The in-process Talker thread has no such fallback — upstream's runtime takes the reference wav as a constructor argument — so `talker_factory_from_config` raises `ValueError` when the key is unset. There is no built-in default path; turning the thread on means setting the key.

`engine.token2wav_timesteps` is **planned, not implemented** (4.4): the vocoder's flow-matching step count, fixed at upstream's 10 today, to be exposed once T1.8/T1.9 has checked the quality at 5 and 2.

`transfer_keywords` and `fallback_texts` are the only runtime-data values in the file: the customer says the first and hears the second, so both are written in the caller's language. Everything else is English.

## 10. Milestones and Acceptance

| Stage | Content | Acceptance criteria |
| --- | --- | --- |
| M0 | Repository skeleton, uv environment, upstream path dependencies importable | `talkover check` exits 0 |
| M1 | MPS smoke test: Thinker + Talker complete inference on an offline audio clip on MPS | Audible speech output; `bench_rtf.py` reports per-unit time |
| M2 | Realtime interface, bidirectional pcm16 audio, transcript, cancel | Cascade protocol test cases pass; a browser or aicc can hold a call |
| M3 | Business Brain: three tools + transfer interception + mock API | An order lookup completes during a call and the order number is read back; transfer fires within 1 s |
| M4 | CUDA backend runs on the A10, with only `device` changed in the same configuration file | All M1–M3 test cases pass on the A10; VRAM stays within 24 GB; per-unit time < 1.0 s, stable for 10 minutes |
| M5 | MPS meets the real-time target, or the MLX project is started | Per-unit time on MPS < 1.0 s, stable for 10 minutes; otherwise phase two begins |

**Scope of the current phase (2026-09-16)**: M0–M3 and M5 on MPS. M4 (CUDA on the A10) is deferred; the engine keeps the CUDA seams (`CudaBackend`, `faster-whisper` ASR, the memory model's CUDA branch, `talker_device`) but nothing is validated on CUDA hardware. Task-level breakdown is in `docs/tasks.md`.

**Phase-two trigger condition**: M1 measures a p95 single Thinker step on MPS > 0.6 s. Phase two affects only the MPS backend; the CUDA backend is unchanged.

**Test matrix**: `tests/engine` uses pytest markers to distinguish `mps`, `cuda`, and `cpu`. The protocol layer and Brain tests do not depend on a GPU and run in CI with a fake engine. At that point the Thinker is ported to MLX with 4-bit/8-bit quantization, while the Talker and token2wav stay on PyTorch MPS for now.

## 11. Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Operators in the MiniCPM-o 4.5 remote code that MPS does not support fall back to CPU | Real-time target not met | Per-operator profiling at M1, recorded in the porting notes; move to MLX earlier if necessary |
| Cerebellum misreads digit strings (order numbers, amounts) | Unusable for customer service | Listening tests with realistically formatted data at M3; if the target is not met, have the Brain rewrite numbers into segmented spoken form before calling `share` |
| `instructions` can only go into the task slate, capped at 256 tokens | Limited phrasing customization | Put the bulk of the phrasing in the Brain system prompt; state this explicitly in the documentation |
| Model behavior on Chinese telephony channels is unknown | Unusable in the domestic market | Evaluate with 8 kHz g711 recordings after M2; if the target is not met, move to data SFT, which is not in this repository |
| A10 24 GB cannot hold full bf16 | CUDA path cannot be validated | The three-step fallback in section 4.5; `talkover check` estimates ahead of time |
| The two backends diverge in behavior (the same input gives different output on MPS and CUDA) | Hard to diagnose | `bench_rtf.py` emits a fixed-seed token sequence for comparison across both; differences are recorded in the porting notes |
| Upstream is still changing frequently (multiple commits in the past week) | Path dependency drift | Pin the upstream commit hash; upgrades follow an explicit process |
| With the Talker thread (4.2), `end_of_turn` closes the response before the Talker has drained that turn; `mapping.py` drops an `is_audio_chunk` event that arrives after the response closed | The tail of an assistant turn can be missing from the client's audio | Open. The fix is to hold the `end_of_turn` close chain until the Talker reports the turn drained (`TalkerThread.wait_until_drained`); it is settled together with the hand-over validation in T1.7. The in-step Talker, still the default, is unaffected |
| Upstream has no API for a text user turn; `conversation.item.create` (`message`) rides the `feed_runtime_event({"type": "user_text"})` envelope, unvalidated against a real model run | A text turn may be read as a tool response, or ignored | Confirm against a loaded checkpoint at M1/M2 (5.3); if the model does not accept it, reject text items in the protocol layer and record the decision in `protocol-profile.md` |
| The 4.6 channel back (`feed_tool_response`, `feed_worker_delivery`) is written against upstream's payload shapes but has never run against a checkpoint | The model stalls on an unanswered `task_start`, or speaks a delivery it does not understand | Confirm with the `mps` acceptance of T1.4; the shapes are copied from `gander_runtime.lean_realtime` rather than invented, and `topic` is asserted against `DeliveryRecord.topic` by a test |

## 12. Deployment Shapes

| Shape | Hardware | Scenario |
| --- | --- | --- |
| Single machine, single line | Mac mini / Mac Studio | Shops, clinics, small front desks; data stays local |
| Single card, single line | 24 GB cards such as A10 / L4 | The smallest unit for per-line billing in the cloud |
| Multiple cards, multiple lines | One machine with several cards, one process per card | Small call centers; the scheduler allocates idle processes based on `/health` |

True multi-session batched inference is out of scope for this repository.

## 13. Relationship to Existing Projects

- **cascade-realtime-gateway**: only the protocol documentation and test cases are reused; no code is imported and that repository is not modified.
- **aicc**: the first client of Talkover. The FreeSWITCH side is unchanged; only the Realtime endpoint is pointed at Talkover.
- **Omni-Interaction-Agent (Gander)**: a path dependency with a pinned commit. Changes to upstream are contributed back as patches; `patches.py` is only a stopgap.
