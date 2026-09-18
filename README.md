# Talkover

An inference gateway that wraps the Gander full-duplex interaction model (built on
MiniCPM-o 4.5) as an OpenAI Realtime compatible WebSocket service, plus a customer-service
Brain provider. It targets single-machine operation on Apple Silicon (MPS) first, with
NVIDIA A10 (CUDA) as the second backend.

See [docs/DESIGN.md](docs/DESIGN.md) for the design document, and
[docs/protocol-profile.md](docs/protocol-profile.md) for the exact protocol behaviour the
service implements.

> **Status.** The service builds, serves and passes its full test suite against a fake
> engine. The live call on the M3 Max — audible replies, transcripts on both sides, a
> working `response.cancel` — is **pending**: the checkpoints in `~/models` are still
> downloading (T2.9 in [docs/tasks.md](docs/tasks.md)).

## 1. Environment

Python is pinned to 3.12 and the package manager is `uv` only.

```bash
uv sync --extra dev --extra mps          # macOS / Apple Silicon
uv sync --extra dev --extra cuda         # NVIDIA
```

`mcpmft` and `gander-runtime` are path dependencies: the upstream checkout
(`Omni-Interaction-Agent`, pinned at commit `cf43838`) must sit next to this repository, or
`uv sync` fails.

## 2. Weights

```bash
scripts/download_models.sh               # into ~/models; override with MODELS_DIR=…
```

It fetches `openbmb/MiniCPM-o-4_5` (the base model, which also carries the token2wav
assets), `Gander-Omni/Gander` (the Thinker and Talker checkpoints) and the MLX
`whisper-large-v3-turbo` used by the ASR side channel on MPS. Point `engine.base_model`,
`engine.thinker_checkpoint` and `engine.talker_checkpoint` at the result.

## 3. Configuration

Copy `configs/serve.example.yaml` and edit it; every key is documented in DESIGN.md
section 9. `${ENV_VAR}` references are expanded when the file is read, so credentials stay
out of it:

```bash
export TALKOVER_API_KEY=…                # the single Bearer key of /v1/realtime
export DEEPSEEK_API_KEY=…                # the Brain's LLM
```

## 4. Check before serving

```bash
uv run talkover check -c configs/serve.example.yaml
```

Static only — it loads no weights, opens no socket and calls no API. It reports the
interpreter, the torch version, the device, the upstream packages and their pinned commit,
the four checkpoint paths, the resolved ASR backend, the LLM key and the memory estimate.
Exit code 0 means nothing failed, 1 that something did, 2 that the config itself is bad.

## 5. Serve

```bash
uv run talkover serve -c configs/serve.example.yaml
uv run talkover serve -c configs/serve.example.yaml --check-only   # build it, do not listen
```

`serve` loads the config, applies the upstream patches for the configured device, builds
the engine, the ASR side channel and the Brain, logs a startup summary (device, checkpoint
paths, ASR backend, memory estimate) and runs uvicorn on `server.listen`. The model is
loaded when the service starts and released on shutdown; `--check-only` returns before that
and is the way to validate the wiring while the weights are still downloading.

Two endpoints:

| Endpoint | Purpose |
| --- | --- |
| `GET /v1/realtime` | the WebSocket, `Authorization: Bearer <server.api_key>` |
| `GET /health` | `200` with `status: "idle"` when free, `503` while `busy` or `not_ready` |

One session per process: a second connection is told `code: "engine_busy"` and closed. A
dropped connection is *not* the end of the session — it is kept for
`realtime.trailing_silence_sec`, and a reconnection that presents the same id resumes it:

```
wss://host/v1/realtime?session_id=sess_…       # or the X-Talkover-Session-Id header
```

## 6. Connecting a Realtime client

Any OpenAI Realtime client works; `?model=` is echoed, never routed on. The audio formats
are negotiated with `session.update` — 24 kHz pcm16 for a browser or an SDK client, and
8 kHz G.711 for a telephony gateway such as aicc or FreeSWITCH:

```jsonc
// pcm16, the default
{"type": "session.update",
 "session": {"type": "realtime",
             "audio": {"input":  {"format": {"type": "audio/pcm",  "rate": 24000}},
                       "output": {"format": {"type": "audio/pcm",  "rate": 24000}}}}}

// G.711 μ-law, for a telephony leg
{"type": "session.update",
 "session": {"type": "realtime",
             "audio": {"input":  {"format": {"type": "audio/pcmu", "rate": 8000}},
                       "output": {"format": {"type": "audio/pcmu", "rate": 8000}}}}}
```

Then the usual loop: `input_audio_buffer.append` (base64 audio) → `input_audio_buffer.commit`
→ `response.output_audio.delta` / `response.output_audio_transcript.delta` back,
`response.cancel` to cut the model off. The GA event names are the only ones accepted; a
beta name such as `response.audio.delta` is rejected.

`turn_detection` is approximate: Gander has no explicit VAD, so
`input_audio_buffer.speech_started` / `speech_stopped` are derived from the ASR side
channel (DESIGN.md 5.4). `session.instructions` only goes into the task slate and is
truncated to `realtime.memory_slate_max_tokens`; business prompts belong in the Brain.

`transfer_to_human` is not executed by the service: it is emitted to the client as a
`function_call`, and the client is expected to answer with a `conversation.item.create`
carrying the matching `function_call_output`.

## 7. The mock business API

`query_ticket` and `query_order` call the business HTTP API from inside the Brain. For a
demo or a test run, serve the bundled mock instead of the real one:

```bash
uv run talkover mock-api -c configs/serve.example.yaml     # host and port from brain.business_api
uv run talkover mock-api --port 9100 --delay-sec 2         # exercise the 3 s timeout path
```

## 8. Development

```bash
uv run pytest                                   # the whole suite
uv run pytest tests/protocol tests/brain -q     # GPU-free: fake engine, fake LLM
uv run pytest -m cpu                            # engine tests that need no accelerator
uv run ruff check . && uv run ruff format .
```

`tests/engine` carries the `mps` / `cuda` / `cpu` markers; everything else runs anywhere.
