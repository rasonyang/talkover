# Talkover Task Breakdown

> Derived from `docs/DESIGN.md` v0.1 (2026-09-16). DESIGN.md stays the source of truth; when a task changes the design, update DESIGN.md first and then this file.

## How to read this file

- Task IDs are `T<milestone>.<n>`. Milestones follow DESIGN.md section 10.
- **Status**: `todo` · `doing` · `done` · `blocked` · `deferred` (not in this phase).
- **Depends on** lists the task IDs that must be `done` first. Tasks with no shared dependency can run in parallel.
- **Acceptance** is the concrete check that closes the task. A task is not `done` until the check has been run and its result recorded.
- Every task that touches the engine on MPS appends its findings to `docs/mps-porting-notes.md`.
- Every task that touches the protocol reflects its rules in `docs/protocol-profile.md`.
- Tests: `tests/protocol` and `tests/brain` must run without a GPU (fake engine, fake LLM). New `tests/engine` cases carry a `cpu` / `mps` / `cuda` marker.

## Dependency overview

```
M0  T0.1 ──▶ T0.2 (config schema) ──▶ T0.3 (talkover check) ◀── T1.2 (memory estimate)
                    │
M1                  ├──▶ T1.1 (DeviceBackend) ──▶ T1.3 (patches) ──▶ T1.4 (engine session) ──▶ T1.5 (talker thread)
                    │                                                        │
                    └──▶ T1.6 (ASR backends) ────────────────────────────────┤
                                                                             ▼
                                                        T1.7 (MPS offline smoke) ──▶ T1.8 (bench_rtf) ──▶ T1.9 (operator profiling)
M2  T0.2 ──▶ T2.1 (events + profile) ──▶ T2.4 (session FSM) ──▶ T2.5 (mapping) ──▶ T2.7 (protocol tests)
             T2.2 (audio)  ──────────────┘        │
             T2.3 (server) ───────────────────────┴──▶ T2.6 (turn detection) ──▶ T2.8 (reconnect) ──▶ T2.9 (serve + live call)
M3  T3.1 (tools) + T3.2 (BrainLLM) ──▶ T3.3 (brain session) ──▶ T3.4 (provider) ──▶ T3.7 (realtime↔brain) ──▶ T3.8 (E2E)
     T3.5 (intercept) ──────────────────────────────────────────┘        T3.6 (mock api) ──┘
M5  T1.9 + T3.8 ──▶ T5.1 ──▶ (T5.2 only if the phase-two trigger fires)
M4  (deferred) T4.1 ──▶ T4.2 ──▶ T4.3 ──▶ T4.4 — interfaces kept by T1.1, T1.2, T1.5, T1.6, T1.8
```

---

## M0 · Repository skeleton

Acceptance for the milestone: `uv run talkover check -c configs/serve.example.yaml` exits 0 on the M3 Max.

### T0.1 Repository skeleton and uv environment — `done`

- Package layout, `pyproject.toml`, path dependencies on upstream, `eva-decord` override, smoke tests, example configs.
- Verified: `uv sync --extra dev --extra mps` succeeds; `mcpmft` and `gander_runtime` import as editable installs (see `docs/mps-porting-notes.md`).

### T0.2 Typed configuration schema — `done`

- Verified 2026-09-17: `uv run pytest tests/protocol/test_config.py` passes; both example configs load; `asr` `auto` resolves to `mlx_whisper` / `mps` / `float16` on `device: mps`. A `realtime:` section (`memory_slate_max_tokens`, `trailing_silence_sec`) and `engine.attn_implementation` were added to DESIGN.md section 9 and the schema.

- **Files**: `src/talkover/config.py`, `tests/protocol/test_config.py` (marker: none, GPU-free).
- **Scope**: replace the bare `dict` with dataclasses for `server`, `engine`, `asr`, `brain` following DESIGN.md section 9. Expand `${ENV_VAR}` references. Validate `engine.device` against `mps | cuda | cuda:<n> | cpu`, `asr.backend` against `auto | mlx_whisper | faster_whisper | sensevoice`, `brain.llm.kind` against `anthropic | openai_compat`. Resolve `auto` values (`asr.backend`, `asr.device`, `asr.compute_type`) from `engine.device`. Unknown keys are an error.
- **Depends on**: T0.1.
- **Acceptance**: `configs/serve.example.yaml` and `configs/brain.business.example.yaml` load without error; tests cover env expansion, `auto` resolution, and rejection of an invalid device string.

### T0.3 `talkover check` — `doing`

- 2026-09-17: implemented; `uv run pytest tests/protocol/test_check.py` passes. `uv run talkover check -c configs/serve.example.yaml` exits 1 with a readable failure list (Gander checkpoints missing, `DEEPSEEK_API_KEY` unset), which covers the non-zero half of the acceptance. Open: the exit-0 run on the prepared machine, blocked on the model download and the API key. Exit codes: 0 no failure, 1 at least one failure, 2 unusable config.

- **Files**: `src/talkover/cli.py`, new `src/talkover/check.py`.
- **Scope**: report Python version, torch version, device availability through the backend (T1.1), importability of `mcpmft` / `gander_runtime` and the pinned upstream commit, presence of `base_model` / `thinker_checkpoint` / `talker_checkpoint` paths, ASR backend importability for the resolved backend, Brain LLM key presence, and the memory estimate from T1.2 with a warning when it exceeds the device budget. Never load model weights.
- **Depends on**: T0.2, T1.1, T1.2.
- **Acceptance**: exit 0 on a correctly prepared machine; non-zero with a readable list of failures when a checkpoint path is missing or the API key is unset.

### T0.4 CI for GPU-free tests — `done`

- 2026-09-18: green run on `main`: https://github.com/rasonyang/talkover/actions/runs/35300453646 (commit `964ae13`, 51 s). Neither open risk materialised: `uv sync --extra dev` on `ubuntu-latest` resolves the CPU torch wheel and builds both upstream packages on Linux + Python 3.12 without a problem, and `deepspeed` (installed on Linux, where the `override-dependencies` marker does not exclude it) does not break the install or the GPU-free suites, because none of them imports `transformers`. The first run (https://github.com/rasonyang/talkover/actions/runs/35300257892) failed only in the last step: four `tests/protocol/test_check.py` cases (`test_all_ok`, `test_missing_token2wav_assets_warn`, `test_unset_api_key_is_warning_for_openai_compat`, `test_cli_exit_code_zero`) reported the `memory` check as `over_budget`. Cause: the MPS budget is the host's physical RAM (`engine/memory.py` `device_budget_bytes`), so the 24.23 GiB estimate fits the 128 GiB dev machine but not the 16 GiB runner — a host dependency the `healthy_env` fixture had left unpatched although its docstring promises that no result depends on the machine. Fix: pin `_host_memory_bytes` to 128 GiB inside that fixture. The CUDA budget is a constant and the `over_budget` / `borderline` cases use `device: cuda:0`, so their coverage is unchanged; the workflow and the dependency config needed no change.

- **Files**: `.github/workflows/ci.yml`.
- **Scope**: `ruff check`, `ruff format --check`, `uv run pytest -m cpu` plus `tests/protocol` and `tests/brain`. The upstream path dependency must be available in CI: check out `Omni-Interaction-Agent` at commit `cf43838` next to the repo (or vendor a stub for import-only tests; decide and record in DESIGN.md section 8).
- **Depends on**: T0.1.
- **Acceptance**: a green run on `main`.

---

## M1 · MPS smoke test

Acceptance for the milestone: Thinker + Talker produce audible speech from an offline audio clip on MPS, and `scripts/bench_rtf.py` reports per-unit timings.

### T1.1 `DeviceBackend` protocol and three implementations — `done`

- Verified 2026-09-17 on the M3 Max: `uv run pytest tests/engine/test_backend.py` gives 25 passed, 1 skipped (`cuda`); the `mps` test ran; lint guard passes. `device(index=None)` deviation recorded in DESIGN.md section 4.2.

- **Files**: `src/talkover/engine/backend/__init__.py`, `mps.py`, `cuda.py`, `cpu.py`, `tests/engine/test_backend.py`.
- **Scope**: implement the protocol from DESIGN.md section 4.2 (`name`, `device()`, `synchronize()`, `rng_state()`, `set_rng_state()`, `autocast_dtype()`), plus `get_backend(device: str) -> DeviceBackend` that parses `cuda:<n>` and `is_available()`. `torch.cuda` / `torch.mps` are referenced only inside `cuda.py` / `mps.py`. `CudaBackend` is implemented fully (it is small) but only exercised by tests that skip without a CUDA device. Add a lint guard (a `cpu`-marked test that greps `src/talkover/engine` outside `backend/` for `torch.cuda` and `torch.mps`).
- **Depends on**: T0.1.
- **Acceptance**: `cpu` tests pass for `CpuBackend` and for `get_backend` parsing; `mps` test passes on the M3 Max; lint guard passes.

### T1.2 Memory estimation — `done`

- Verified 2026-09-17: `uv run pytest tests/engine/test_memory.py` passes (18 tests): `cuda` + ASR on CUDA is `over_budget` (24.23 / 24 GiB), `cuda` + `asr.device: cpu` is `borderline` (21.71 GiB), `mps` with 128 GiB is `ok`. `borderline` is total >= 85 % of budget (DESIGN.md 4.5). ASR footprint numbers are estimates to calibrate in M1.

- **Files**: `src/talkover/engine/memory.py`, `tests/engine/test_memory.py` (marker `cpu`).
- **Scope**: estimate bytes from the config using the table in DESIGN.md section 4.5: Thinker (with / without vision encoder), Talker + token2wav, KV cache from `context_max_units`, activations, ASR on the same device or not, int8 Thinker option. Return a structured result with per-component numbers and a verdict against the device's budget (24 GB for A10, unified memory size on MPS). The CUDA branch stays in the model so M4 can pick it up unchanged, but only the MPS numbers are validated this phase.
- **Depends on**: T0.2.
- **Acceptance**: the example config with `device: cuda` and ASR on CUDA is flagged as over budget; with `asr.device: cpu` it is flagged as borderline; MPS with 128 GB is fine.

### T1.3 Upstream monkeypatches — `done`

- Verified 2026-09-17: in a fresh interpreter `import talkover.engine.session` succeeds without CUDA and `mcpmft.infer.common` / `mcpmft.infer.realtime` import afterwards with `load_for_infer` wrapped; `tests/engine/test_patches.py` passes (12 tests). One patch; unpatched runtime binding points for T1.4 / T1.5 are listed in the porting notes.

- **Files**: `src/talkover/engine/patches.py`, `docs/mps-porting-notes.md`.
- **Scope**: `apply_patches(backend)` must be called before any `mcpmft.infer` import. First known target: the `torch.cuda.is_available()` gate in `mcpmft/infer/common.py`. Each patch is a separate function with a docstring naming the upstream file, line, and the upstream patch it corresponds to. Import-order enforcement: `talkover.engine.session` calls `apply_patches` at module import time.
- **Depends on**: T1.1.
- **Acceptance**: `import talkover.engine.session` succeeds on a machine without CUDA; every patch has a matching entry in the porting notes.

### T1.4 Engine session wrapping `DuplexLiveSession` — `doing`

- 2026-09-17: `engine/protocol.py` (`EngineProtocol`, `EngineStepEvent`) and `EngineSession` implemented; `cpu` tests pass. Open: the `mps` acceptance (real model, 5 s of silence) has not run because the Gander weights are still downloading; it is the first real model load on MPS. `submit_text_turn` uses `feed_runtime_event({"type": "user_text"})`, unvalidated until then (DESIGN.md section 11). `EngineProtocol` is stable, so T2.4 is not blocked.

- 2026-09-18: `EngineProtocol` grew the channel back to the Cerebellum that T3.7 found missing (DESIGN.md 4.6): `feed_tool_response(response)` wrapping upstream `DuplexLiveSession.feed_tool_response` (the bounded `control_tool_response` shape — `status`, `task_ids`, optional `reason` — that the model blocks on until it arrives) and `feed_worker_delivery(delivery)` wrapping `feed_runtime_event` with the `worker_delivery` envelope `gander_runtime.lean_realtime.worker_delivery_response` produces (`type`, `task_name`, `topic`, `content`, plus `status` on a `final`). Both are ordinary commands on the inference thread, marshalled exactly like `submit_text_turn`, with one deliberate exception to the "every exception is terminal" rule: neither runs the model, so an upstream refusal (wrong slot, or an envelope past `max_tool_response_tokens`) is reported to the caller as `EngineError` and logged instead of ending the session — one unspoken Brain update is a smaller loss than a dropped call. `realtime/brain_bridge.py` is their only caller and now holds the engine directly; the stub `CerebellumChannel` protocol is deleted and `app.py` passes the session's engine to every `BrainBridge`. The delivery envelope was corrected while wiring it: `topic` is `DeliveryRecord.topic`'s closed set (`milestone | interaction | final | risk | aggregate`), so a Brain `share` is a `milestone` and a `result` is the terminal `final` carrying a `status`, and `task_name` is the semantic name the model passed to `task_start`, not Talkover's internal `task_id` which the model has never uttered. Open: **the real-model validation of these two calls is pending with the `mps` acceptance** (the weights are still downloading), exactly like `submit_text_turn`'s `user_text` envelope — until a checkpoint has answered a `task_start` and spoken a delivery, 4.6 is unverified against the model (DESIGN.md section 11).
- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean; `uv run pytest -q` gives 867 passed, 22 skipped (the T3.7 baseline of 859 plus 8 new cases). New `cpu` cases in `tests/engine/test_session.py`: both calls reach the stub `DuplexLiveSession` on the inference thread with the upstream payloads, a `type` other than `worker_delivery` is a `ValueError` before submission, a refused payload raises `EngineError` and leaves the session able to run the next unit, and `WORKER_DELIVERY_TOPICS` is pinned. New bridge cases in `tests/protocol/test_function_call.py`: the whole round trip goes back through the engine (`feed_tool_response` on the `task_start` unit, then `feed_worker_delivery` with the `final` result the client's `function_call_output` unblocked), `WORKER_DELIVERY_TOPICS` is asserted against `get_type_hints(DeliveryRecord)["topic"]`, a refusing engine costs the delivery and not the session, and a bridge without an engine still logs and drops. `tests/protocol/fake_engine.py` gained both methods plus `tool_responses` / `deliveries`, backward-compatibly.

- **Files**: `src/talkover/engine/session.py`, `tests/engine/test_session.py`.
- **Scope**: build upstream settings from the typed config without going through `gander_runtime/cli.py`; construct `DuplexLiveSession` on the backend device with `attn_implementation` from config, bf16, `init_vision: false` under `media_mode: voice`. Run inference on a dedicated thread; expose an asyncio-facing API: `start()`, `feed_pcm16(pcm16_16k_1s)`, `flush_pending()`, `interrupt_output()`, `set_task_slate(text)`, `submit_text_turn(text)`, `events()` async iterator of `DuplexStepEvent`, `stop()`. Define an `EngineProtocol` so the realtime layer can run against a fake engine.
- **Depends on**: T1.1, T1.3.
- **Acceptance**: `cpu` test drives a fake engine through the protocol; `mps` test loads the real model, feeds 5 s of silence, and receives `is_listen: true` units without error.

### T1.5 In-process Talker thread — `doing`

- 2026-09-18: implemented. `TalkerThread` runs a Talker runtime on its own thread with the
  `backend.synchronize()` separation (hand-over on the Thinker thread, completion on the Talker thread), keeping
  upstream's `AsyncTalkerWorker` interface so `DuplexLiveSession` drives it; `talker_device_shim`
  (`engine/backend/shims.py`) redirects the CUDA-only call sites of `detached_talker.py` **and** of the third-party
  `stepaudio2` vocoder onto the `DeviceBackend`; `build_talker_runtime` builds the upstream
  `DetachedTalkerRuntime` on the backend device without its CUDA-only classmethod; `patches.speech_worker_override`
  puts the thread in place of upstream's worker for one `DuplexLiveSession(...)` call. `EngineSession` takes
  `talker_factory` (the T1.4 seam, renamed) and publishes Talker audio as `EngineStepEvent(is_audio_chunk=True)`;
  it stays **opt-in**, so the default path is still T1.4's in-step Talker.
- Verified 2026-09-18: `uv run pytest -m cpu -q` 128 passed; `uv run pytest tests/engine/test_talker.py -m mps`
  passed on the M3 Max — a fixed 25-token S3 sequence went through the thread into one second of 24 kHz float32 audio
  (peak 0.21), hand-over cost 0.02 ms against a 0.01 ms `synchronize`, warm unit 0.68 s. That covers the vocoder half
  of the acceptance, which needs only `Gander/talker/assets`.
- Open (awaiting weights): the AR Talker half — `build_talker_runtime`, `warm_token2wav` (the RNG bypass) and
  `DetachedTalkerRuntime.synthesize` — has never run, because `MiniCPM-o-4_5` and `Gander/thinker` are still
  downloading and `Gander/talker/model.safetensors` is missing. The `EngineSession` hand-over (upstream
  `take_talker_condition` feeding `talker_factory`) is likewise unvalidated against a model; it is exercised only with
  a fake runtime. Turning the factory on by default, and the timing decision below, belong to that run (T1.7).
- Open (budget): token2wav alone costs 0.49 s per unit at upstream's `n_timesteps=10` on MPS, against the 0.35 s
  DESIGN.md 4.4 allows for Talker plus token2wav; 5 steps costs 0.24 s and 2 steps 0.14 s, quality unmeasured.
  `float16=True` aborts the process on MPS (MPSGraph dtype mismatch), so the vocoder stays float32 there.
  Three dependency pins were needed before `stepaudio2` would import at all (`torchaudio>=2.6,<2.7`, `setuptools<81`,
  `onnx<1.18`); all of it is in `docs/mps-porting-notes.md`, 2026-09-18.
- 2026-09-18 (drain marker): `EngineStepEvent` gained the additive `talker_done` flag and `EngineSession` gained
  `drained_event(...)`. `_publish_talker_audio` now publishes a marker event — `is_audio_chunk=True`,
  `talker_done=True`, no waveform, `generation_id` / `unit_id` naming the turn — for a `TalkerDone` whose request
  carried `end_of_turn`, and for every `TalkerInterrupted` (with `cancelled_generation_id`, since a cancelled
  generation can produce no more audio either). It is the signal `realtime/mapping.py` holds the `end_of_turn` close
  chain on, which closes the DESIGN.md section 11 risk row; nothing else in the engine reads it, and the in-step
  Talker never emits one.
- 2026-09-18 (`engine.token2wav_timesteps`): the knob the budget note above asked for is now a config field —
  `int`, default upstream's 10, validated against `config.TOKEN2WAV_TIMESTEPS_RANGE` (`1..50`), rejected with
  `ConfigError` outside it and for a non-integer (booleans included). `talker_factory_from_config` forwards it to
  `build_talker_runtime(n_timesteps=...)` and on into `build_token2wav` -> `stepaudio2.Token2wav`;
  `talkover check` echoes it on the `engine.token2wav_dir` line
  (`... (<engine.base_model>/assets/token2wav), 10 flow-matching steps`); `configs/serve.example.yaml` and
  DESIGN.md 4.4 / 9 document it as implemented. **The value still ships at 10**: only the latency of 5 (0.24 s) and
  2 (0.14 s) was measured, never the quality, so choosing a lower default stays with T1.9.
- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean for this task's files;
  `uv run pytest tests/engine -q -rs` gives 141 passed, 3 skipped (2 CUDA, 1 awaiting the Gander weights) and
  `uv run pytest -m cpu -q` 137 passed. New `cpu` cases in `tests/engine/test_talker.py`:
  `build_token2wav` passes `n_timesteps` through to a fake `stepaudio2.Token2wav` (default 10, explicit 2), and the
  factory built by `talker_factory_from_config` forwards it to `build_talker_runtime`. `tests/protocol/test_config.py`
  pins the default, both range bounds, both out-of-range values, a string and a bool.

- **Files**: `src/talkover/engine/talker.py`, `src/talkover/engine/backend/shims.py`,
  `src/talkover/engine/patches.py`, `src/talkover/engine/session.py`, `src/talkover/engine/protocol.py`,
  `src/talkover/config.py`, `src/talkover/check.py`, `configs/serve.example.yaml`,
  `tests/engine/test_talker.py`, `tests/protocol/test_config.py`, `pyproject.toml`,
  `docs/mps-porting-notes.md`.
- **Scope**: run Talker + token2wav on its own thread on the same device, handing over speak tokens through a queue and separating Thinker and Talker with `backend.synchronize()`. Keep upstream detached mode reachable when `engine.talker_device` names a different CUDA card; this phase it may raise `NotImplementedError` pointing at M4. Record how `detached_talker.py`'s `torch.cuda.device` / RNG / `synchronize` calls were bypassed.
- **Depends on**: T1.4.
- **Acceptance**: `mps` test produces a 24 kHz waveform from a fixed speak-token sequence; the Talker thread never blocks the Thinker step for longer than `synchronize`.

### T1.6 ASR backends — `done`

- Verified 2026-09-17: `uv run pytest tests/engine/test_asr.py -rs` gives 19 passed, 1 skipped (`cuda` extra not installed); the `mps` test transcribed `tests/fixtures/zh_order_16k.wav` with mlx-whisper. The timing assertion covers a warm transcribe only.

- **Files**: `src/talkover/engine/asr/__init__.py`, `mlx_whisper.py`, `faster_whisper.py`, `sensevoice.py`, `tests/engine/test_asr.py`.
- **Scope**: `AsrBackend` protocol matching upstream `asr_process` (pcm16 16 kHz segments in, text with timestamps out) plus a VAD-style `speech_started` / `speech_stopped` signal for T2.6. `mlx_whisper` (MPS default) implemented and tested; `faster_whisper` (CUDA default, CPU int8 option) implemented against the same protocol but not validated on CUDA this phase; `sensevoice` as a stub raising `NotImplementedError`. `create_asr(config, backend)` handles `auto`.
- **Depends on**: T0.2, T1.1.
- **Acceptance**: `mps` test transcribes a bundled Chinese clip with mlx-whisper; `cuda` test for faster-whisper exists and skips without a device; `cpu` test covers `auto` resolution with a fake backend, including the `cuda` → `faster_whisper` mapping.

### T1.7 MPS offline smoke test — `todo`

- **Files**: `tests/engine/test_mps_smoke.py` (marker `mps`), `scripts/download_models.sh`, `tests/fixtures/` (short 16 kHz clip).
- **Scope**: verify `download_models.sh` fetches `MiniCPM-o-4_5` and the Gander Thinker / Talker checkpoints into `MODELS_DIR`. Feed the clip through T1.4 + T1.5 and write the output waveform to a file for manual listening.
- **Depends on**: T1.4, T1.5, T1.6.
- **Acceptance**: an output wav with audible speech exists; the test asserts non-silent output and at least one `is_listen: false` unit.

### T1.8 `bench_rtf.py` and `talkover bench` — `doing`

- 2026-09-18: the harness is implemented; the numbers are not, because `MiniCPM-o-4_5` and
  `Gander/thinker` are still downloading. `scripts/bench_rtf.py` loads the config, builds the backend and the
  engine, pins the RNG from `--seed` through `backend.set_rng_state`, feeds a fixed clip in 1 s units
  (`tests/fixtures/zh_order_16k.wav` by default, a deterministic synthetic tone with `--clip synthetic` or when
  the fixture is absent, looped when `--units` asks for more), prints the per-stage mean / p50 / p95 / max table
  against the DESIGN.md 4.4 budget, and writes the JSON dump (`schema: talkover.bench/1`) whose `token_sequence`
  is the M4 cross-backend diff. The per-stage seam is `EngineSession(on_unit_timing=...)` (new, optional,
  keyword-only): one `UnitTiming` per stepped unit and per piece of Talker-thread work, with `synchronize()` on
  both sides of the upstream call and the stage costs read from upstream's `cost_llm` / `cost_tts_prep` /
  `cost_tts` / `cost_token2wav`. `EngineProtocol` is unchanged. `talkover bench -c <config>` loads the script by
  path and delegates to its `main`.
- Verified 2026-09-18: `uv run ruff check` and `ruff format --check` clean on this task's files;
  `uv run pytest -m cpu -q` 157 passed; `uv run pytest tests/engine -q -rs` 161 passed, 4 skipped (the two
  weight-dependent `mps` cases among them). `tests/engine/test_bench.py` (20 `cpu` cases) covers the statistics
  helpers, the clip loader, `stage_times` against upstream-shaped metrics, the hook against `EngineSession` with
  a stub upstream session (a raising callback does not end the session), `run_bench` with `FakeBenchEngine`,
  the pinned JSON key set, the seed-stable `token_sequence`, and `talkover bench --engine fake` writing the dump
  into `tmp_path`. A fake run deliberately uses the `cpu` backend whatever `engine.device` says, so it runs in CI.
- Open (awaiting weights): every real number. The recorded stages differ from the 4.4 table, which now says so:
  upstream has no separate audio-encoding cost (it is inside `cost_llm`) and speak-token **ids** never reach
  Talkover — the dump carries `n_tts_tokens` instead. The `mps` acceptance case
  (`tests/engine/test_bench.py::test_real_bench_run_on_mps`) skips on incomplete checkpoints; once they land, run
  `uv run talkover bench -c configs/serve.example.yaml --units 30 --warmup 2 --asr` and put the table, the p95
  Thinker step (the 0.6 s phase-two trigger) and the `token2wav_timesteps` comparison into
  `docs/mps-porting-notes.md`.

- **Files**: `scripts/bench_rtf.py`, `src/talkover/cli.py`, `src/talkover/engine/session.py`,
  `tests/engine/test_bench.py`.
- **Scope**: run a fixed-seed clip, report per-unit wall time and per-stage time (audio encoding, Thinker step, Talker, token2wav, ASR) as mean / p50 / p95, and dump the token sequence to JSON so a later CUDA run (M4) can be diffed against it. Set RNG through `backend.set_rng_state`.
- **Depends on**: T1.7.
- **Acceptance**: `uv run talkover bench -c configs/serve.example.yaml` prints the table and writes the JSON; numbers are recorded in `docs/mps-porting-notes.md`.

### T1.9 MPS operator profiling and phase-two decision — `todo`

- **Files**: `docs/mps-porting-notes.md`, `docs/DESIGN.md` section 10 if the decision changes anything.
- **Scope**: profile one Thinker step on MPS, list every operator that falls back to CPU, and record whether p95 exceeds 0.6 s. If it does, open the phase-two decision (T5.2) but continue M2–M3 on PyTorch MPS.
- **Depends on**: T1.8.
- **Acceptance**: a dated entry in the porting notes with the p95 number and the operator list.

---

## M2 · Realtime interface

Acceptance for the milestone: Cascade protocol cases pass against the fake engine; a browser or aicc holds a call with bidirectional pcm16 audio, transcript, and cancel.

### T2.1 Event models and protocol profile — `done`

- Verified 2026-09-17: `uv run pytest tests/protocol` passes (199 tests); the rejection-code and rejection-case tables in `docs/protocol-profile.md` are parsed by the tests and must match them. Protocol decisions recorded in DESIGN.md section 5; the `response.create` override rejection is to be confirmed against the aicc client.

- **Files**: `src/talkover/realtime/events.py`, `docs/protocol-profile.md`, `tests/protocol/test_events.py`.
- **Scope**: port field paths, rejection codes, and error event formats from Cascade's `docs/protocol-profile.md` into Talkover's profile with a differences section (single API key, `engine_busy`, approximate turn detection, instructions cap, ignored audio pre-processing fields). Implement dataclasses and validation for every client event in DESIGN.md section 5.3 and every server event Talkover emits. Docs only from Cascade; no code is copied.
- **Depends on**: T0.2.
- **Acceptance**: each rejection code in the profile has a test that produces the documented `error` event.

### T2.2 Audio utilities — `done`

- Verified 2026-09-17: `uv run pytest tests/protocol/test_audio.py` passes (29 tests). `scipy>=1.11` declared in `pyproject.toml`.

- **Files**: `src/talkover/realtime/audio.py`, `tests/protocol/test_audio.py`.
- **Scope**: base64 pcm16 decode / encode, 24 kHz to 16 kHz input resampling, g711 ulaw / alaw decode and encode, and a framer that accumulates input into exact 1 s 16 kHz units and keeps the incomplete remainder (cleared by `input_audio_buffer.clear`).
- **Depends on**: T0.1.
- **Acceptance**: round-trip tests for both codecs; a 2.5 s input yields two units plus a 0.5 s remainder; resampling a sine keeps its frequency.

### T2.3 FastAPI server, auth, health, single-session guard — `done`

- Verified 2026-09-17: `uv run pytest tests/protocol/test_server.py` passes (15 tests): 401 on bad or missing key, `session.created` on a good key, `/health` 200 idle / 503 busy / 503 not_ready, second connection gets `engine_busy` then close 1011.

- **Files**: `src/talkover/realtime/server.py`, `tests/protocol/test_server.py`.
- **Scope**: `GET /v1/realtime?model=<any>` WebSocket upgrade with `Authorization: Bearer` against the single key; `GET /health` reporting engine, ASR, and Brain readiness; second concurrent connection gets `error` with `code: "engine_busy"` then close, and HTTP 503 from `/health` while busy.
- **Depends on**: T2.1.
- **Acceptance**: tests with a fake engine cover bad key (401), good key, `/health` before and during a session, and the second-connection rejection.

### T2.4 Realtime session state machine — `done`

- Verified 2026-09-18: `uv run ruff check src/talkover/realtime tests/protocol` and `uv run ruff format --check src/talkover/realtime tests/protocol` pass; `uv run pytest tests/protocol tests/brain -q` gives 477 passed, 1 skipped (the skip is the opt-in live DeepSeek smoke), of which `tests/protocol/test_session.py` is 54 new cases. Every client event has a case asserting both the fake-engine call and the acknowledgement event, plus an end-to-end pass over the WebSocket transport with `session_factory=WebSocketSession`. Settled protocol rules are in `docs/protocol-profile.md`: the effective session defaults (§3), commit and `response.create` feeding the zero-padded partial unit before `flush_pending` (§2), and the new §8.3 state-dependent rejection table, machine-read by `tests/protocol/test_session.py`. Open: `server.create_app` still defaults to `DefaultSession`; wiring `WebSocketSession` in belongs to T2.9.

- **Files**: `src/talkover/realtime/session.py`, `tests/protocol/test_session.py`, `tests/protocol/fake_engine.py` (the scriptable fake engine T2.7 extends).
- **Scope**: implement the client → server half of the table in DESIGN.md section 5.3: `session.update` (tools, voice, formats, echo of ignored fields, `instructions` truncated to `memory_slate_max_tokens` and pushed to `set_task_slate`, truncated value echoed in `session.updated`), `input_audio_buffer.append / commit / clear`, `response.create` as `flush_pending`, `response.cancel` as `interrupt_output`, `conversation.item.create` for text messages (text user turn) and `function_call_output` (forwarded to Brain, T3.7), `conversation.item.truncate / delete` acknowledged only. Session and conversation IDs, item IDs, and the `model` echo.
- **Depends on**: T2.1, T2.2, T1.4 (protocol only).
- **Acceptance**: every client event has a test asserting the resulting fake-engine call and the acknowledgement event.

### T2.5 `DuplexStepEvent` to Realtime event mapping — `done`

- Verified 2026-09-18: `uv run ruff check src/talkover/realtime tests/protocol && uv run ruff format src/talkover/realtime tests/protocol` pass, and `uv run pytest tests/protocol -q` gives 346 passed, of which `tests/protocol/test_mapping.py` is 26 new cases (normal turn, interruption, client cancel, ASR transcript, g711 output, the function-call pair, and the acceptance case that drives the scripted `FakeEngine` sequence through `pump_engine_events`). `ResponseMapper` in `src/talkover/realtime/mapping.py` owns the whole `response.*` lifecycle; `RealtimeSession.handle_engine_event` runs turn detection, then the mapper, then the optional `on_engine_event` hook, and `RealtimeSession.emit_function_call(call_id, name, arguments)` is the method T3.7 calls for `transfer_to_human`. Settled rules are appended to `docs/protocol-profile.md` §9: the close-chain order (a cancelled response inserts `conversation.item.truncated` before `response.done` and marks its item `incomplete`), `client_cancelled` vs `turn_detected`, the best-effort `usage` shapes, the duration-based `usage` of the transcription event, and the response a function call opens for itself when the model is not speaking. `is_audio_chunk` units (detached Talker, T1.5) only extend an open response. Open: nothing for this task; the Brain side of the function-call seam is T3.7.
- 2026-09-18 (holding the close chain): the DESIGN.md section 11 risk — with the threaded Talker, `end_of_turn` closed the response before the turn's tail had been vocoded, and `mapping.py` dropped the late chunk — is settled and 5.3 / 11 now carry the rule. `ResponseMapper` marks the response **pending-close** on `end_of_turn` instead of closing it, and emits the chain when the engine's `talker_done` marker for that `generation_id` arrives (a marker for a *later* generation releases it too: the Talker only bumps the generation on a cancel, so the held turn can produce no more audio). Chunks arriving inside the hold are delivered normally, which is the whole point; a chunk with no response to extend is dropped and counted in `ResponseMapper.dropped_audio_chunks` instead of vanishing silently. `interrupted` is never held — a barge-in makes everything still in the Talker stale — and a non-listen unit arriving inside a hold belongs to the next turn, so it flushes the hold before opening its own response. `mapping.TALKER_DRAIN_TIMEOUT_SEC` (3 s, a constant and not a config key: it is a failure bound, not a tuning knob) arms an `asyncio` timer per hold, so a Talker that never reports costs the tail of one turn and never the response; the timer is cancelled by every close path, and `cancel_pending_close()` exists for a teardown that abandons the response instead of closing it. Threaded mode is **detected, not guessed**: upstream's `talker_state()` reports `metrics["talker"]["mode"] == "detached"` on every unit, which holds even a one-unit first turn that has not produced a chunk yet; any `is_audio_chunk` event latches the same flag for a fake engine that reports no metrics. Until the flag is set the T1.4 in-step path behaves exactly as before.
- Verified 2026-09-18: `uv run pytest tests/protocol tests/brain -q` gives 763 passed, 18 skipped; `tests/protocol/test_mapping.py` is 36 cases, 10 of them new — in-step `end_of_turn` still closes at once and leaves `talker_threaded` false; a held turn delivers a chunk that arrives inside the hold and closes only on the marker; the section 11 regression (two chunks and a marker after `end_of_turn`, both `response.output_audio.delta` before `response.done`); the one-unit first turn held on the metrics alone; an interrupt closing at once with the following chunk counted as dropped; an interrupt cancelling a held response rather than completing it; the next turn flushing a hold the Talker never released into two distinct responses; the drain timeout closing an abandoned response as `completed`; the timer not surviving the response it guarded; and `cancel_pending_close` dropping a hold without emitting it. `uv run ruff check . && uv run ruff format .` clean for this task's files.

- **Files**: `src/talkover/realtime/mapping.py`, `tests/protocol/test_mapping.py`.
- **Scope**: the server → client half of the table: first `is_listen: false` unit opens `response.created` / `output_item.added` / `conversation.item.created`; `text` becomes `response.output_audio_transcript.delta`; `audio_waveform` becomes `response.output_audio.delta` (24 kHz, or g711 when configured); `end_of_turn` closes the response; `interrupted` closes with `status: "cancelled"` plus `conversation.item.truncated`; ASR segment becomes `conversation.item.input_audio_transcription.completed`; Brain `transfer_to_human` becomes one `response.function_call_arguments.delta` followed by `.done`.
- **Depends on**: T2.4.
- **Acceptance**: a scripted event sequence from the fake engine yields the exact expected client event list.

### T2.6 Approximate turn detection — `done`

- Verified 2026-09-18: `uv run pytest tests/protocol -q` gives 320 passed (26 of them the new `test_turn_detection.py`), and `uv run ruff check src/talkover/realtime tests/protocol && uv run ruff format src/talkover/realtime tests/protocol` is clean for this task's files. `TurnDetector` is pure: it returns the server events instead of emitting them, and `RealtimeSession` keeps four hook lines (construct it, `configure` it on `session.update`, `on_asr_event` for the ASR signals, `detect_turn` inside `pump_engine_events`). Decisions recorded in `docs/protocol-profile.md` section 7: the retroactive start is anchored at `(unit_index - 1) * 1000`, `audio_end_ms` is clamped up to `audio_start_ms` because the ASR and unit clocks are separate approximations, `item_id` is minted with `speech_started` and taken by the next `input_audio_buffer.commit`, `idle_timeout_ms` is echoed but never acts, and `is_audio_chunk` events (T1.5 late Talker audio) never open a span.

- **Files**: `src/talkover/realtime/turn_detection.py`, `tests/protocol/test_turn_detection.py`, `docs/protocol-profile.md`.
- **Scope**: `server_vad` and `semantic_vad` map to one implementation. `speech_started` from the ASR VAD signal or emitted retroactively on `interrupted: true`; `speech_stopped` at ASR segment end. `create_response` and `interrupt_response` accepted and ignored. `turn_detection: null` disables both events. Document the approximation.
- **Depends on**: T2.4, T1.6 (interface only).
- **Acceptance**: tests for the three trigger paths and for `null`.

### T2.7 Protocol regression suite and fake engine — `done`

- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` pass; `uv run pytest tests/protocol tests/brain -q` gives 685 passed, 19 skipped, 1 xfailed, of which `tests/protocol/test_cascade_cases.py` is 125 passed, 18 skipped, 1 xfailed. The 18 skips are the Cascade-only cases, each with its reason, now also listed in the new section 11.1 of `docs/protocol-profile.md` and machine-read from it by the suite. Two more things are machine-read so the document cannot drift from the code: the effective-session JSON block of section 3 (compared against `session.created`) and the "Never emitted" list of section 9 (checked against a full turn). The single `xfail(strict=True)` is a real defect, not a missing test: `RealtimeSession._merge_audio` merges `audio.input.turn_detection` field-wise, so switching to `semantic_vad` keeps the `server_vad` defaults `threshold` / `prefix_padding_ms` / `silence_duration_ms` and the echo is an object the profile's own section 3 row rejects; switching mode should replace the object. `tests/conftest.py` now holds the shared fixtures (`fake_engine`, `realtime_config`, `realtime_app`, `realtime_client`, `realtime_connect` with the authenticating `RealtimeClient` helper), and `FakeEngine` gained `on_call` per-method scripts, `replay_delay`, `script_call` and `extend_script`. Cases run end to end over `create_app(..., session_factory=WebSocketSession)`; only the Brain and ASR side channels, which have no client-initiated path, run against `RealtimeSession` directly.

- **Files**: `tests/conftest.py`, `tests/protocol/fake_engine.py`, `tests/protocol/test_cascade_cases.py`.
- **Scope**: a scriptable fake engine implementing `EngineProtocol` from T1.4; port Cascade's Realtime client test cases as data-driven tests. Cases that depend on Cascade-only behavior are listed as skipped with a reason in `docs/protocol-profile.md`.
- **Depends on**: T2.3, T2.4, T2.5, T2.6.
- **Acceptance**: `uv run pytest tests/protocol` is green in CI.

### T2.8 Reconnect within `trailing_silence_sec` — `done`

- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean, `uv run pytest tests/protocol tests/brain -q` gives 716 passed, 19 skipped, 1 xfailed (the suite includes T2.7's cases, landing in parallel), of which `tests/protocol/test_reconnect.py` is 31 new cases. Resume is identified by `?session_id=sess_…`, or by the header `X-Talkover-Session-Id` when the dialler owns the URL; the query parameter wins. `SessionSlot` now holds the *runner* as well as the *holder*: a disconnect drops the holder, arms a `call_later(trailing_silence_sec)` timer and leaves the slot busy, so `resume` can hand the same `WebSocketSession` to a reconnection while every other connection — a foreign id, or none — still gets `engine_busy` / 1011. A resumed connection reads `session.created` with the same session id and effective session, `conversation.created` with the same conversation id, `session.updated` when a `session.update` had changed something, then the backlog. The engine pump keeps running while detached and its events go into `ReconnectSink`, a bounded FIFO of `DETACHED_EVENT_LIMIT` (256) events; past the bound the oldest is dropped, counted in `sink.dropped` and logged at WARNING — never silently. The timer is injectable (`create_app(call_later=...)`), so no case waits 8 s. Deviations from DESIGN.md 5.7, both documented in `docs/protocol-profile.md` §10.1 rather than by editing DESIGN.md: (a) releasing the session does not stop the engine here — `create_app(on_release=...)` is awaited when the slot frees and T2.9 wires the engine reset into it, because `create_app` does not own `engine.start` / `stop` and an `engine.stop()` would make `/health` report `not_ready` instead of the `idle` the acceptance asks for; (b) only a resumable runner gets a window, so the placeholder `DefaultSession` still frees its slot the moment the socket closes. Open: nothing for this task; wiring `WebSocketSession` as the default factory stays T2.9.

- **Files**: `src/talkover/realtime/session.py`, `src/talkover/realtime/server.py`, `tests/protocol/test_reconnect.py`.
- **Scope**: on disconnect keep the engine session alive for `trailing_silence_sec`; a new connection presenting the same `session_id` resumes it; otherwise release and become idle for `/health`.
- **Depends on**: T2.3, T2.4.
- **Acceptance**: tests for resume inside the window and release after it.

### T2.9 `talkover serve` and a live call — `doing`

- 2026-09-18 (the code half; the live call waits for the weights): `talkover serve` is implemented end to end except for the acceptance itself. `cli.py` loads the config (a `ConfigError` or an unsupported `engine.device` exits 2), builds the device backend, calls `apply_patches(backend)` **before** importing `talkover.engine.session` and again right after it (that module applies the patches a second time against the device auto-detected at import, so the configured backend has to win), constructs `EngineSession(config, backend=backend)` and the ASR side channel, composes everything through `build_app`, logs a startup summary (`app.startup_summary`: listen address, device and dtype, the four checkpoint paths, the resolved ASR backend, the Brain LLM and whether its key is set, the echoed model name, and the static `engine.memory` estimate) and runs uvicorn on `server.host` / `server.port`. `talkover serve --check-only` stops after `build_app`: it binds no port and loads no weights, which is what makes the wiring testable while `~/models` is still downloading. **Lifecycle**: `build_app` installs a FastAPI lifespan — `engine.start()` and `AsrSideChannel.start()` before the first request, then `TalkoverApp.aclose()`, `AsrSideChannel.aclose()` and `engine.stop()` on shutdown — so no other module owns `start` / `stop` (`create_app` gained a `lifespan` seam for it). **Release**: `on_release` closes the bridge, unbinds and resets the ASR stream, and calls `app.reset_engine`, which never calls `engine.stop()`, so `/health` returns to `idle` (T2.8's note, DESIGN.md 5.7). Upstream has no "new conversation" call short of rebuilding `DuplexLiveSession`, which would reload the weights, so the reset is layered: an engine offering `reset()` gets one, any other engine gets `interrupt_output()`, and a failure is logged rather than raised. **ASR**: `AsrSideChannel` taps `feed_pcm16` through a thin delegating engine wrapper — the units the model is fed are exactly the 16 kHz pcm16 ASR wants — queues them (bounded, `ASR_QUEUE_UNITS = 32`, oldest dropped and counted when transcription falls behind), transcribes off the loop with `AsrStream.afeed`, and fans every `AsrEvent` out to **both** `RealtimeSession.on_asr_event` and `BrainBridge.on_asr_event`; a commit requests a flush and a release resets the stream. The ASR backend is built lazily inside `start()`, and a backend that cannot be loaded is logged and the call runs without transcripts rather than failing to serve. Also in this task: `create_app` echoes `default_model(config)` (the base-model directory name, `DEFAULT_MODEL` when unset) when the client sends no `?model=`, and the `/health` `brain` field is a real probe (`app.brain_ready`: a constructed provider plus, when Talkover built it from `config.brain`, a non-empty LLM key) instead of a flag pinned to `True`. `DefaultSession` and its T2.3 tests are untouched: it is still what a bare `create_app` uses, while `build_app` always passes `WebSocketSession`.
- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean; `uv run pytest tests/protocol tests/brain -q` gives 763 passed, 18 skipped, `uv run pytest -m cpu -q` gives 157 passed (T1.9 and the `bench` task landed cases in parallel) and the whole suite is 924 passed, 22 skipped. New: `tests/protocol/test_app.py` (20 GPU-free cases — the lifespan starting and stopping the engine, `manage_engine=False`, a WebSocket round trip through the built application reaching the fake engine's `feed_pcm16` / `flush_pending`, release resetting instead of stopping with `/health` back at `idle`, an engine's own `reset()` preferred, a failing reset swallowed, the ASR fan-out to both consumers, the drop-oldest bound, a failing consumer not starving the other, the tap feeding one stream and both consumers over a live socket, a failing ASR backend not stopping the service, the `brain` probe, the `?model=` default, the startup summary and the CLI surface) and `tests/engine/test_serve_cli.py` (3 `cpu` cases for `serve --check-only`, an unsupported device and a missing file). `uv run talkover serve -c configs/serve.example.yaml --check-only` was run by hand and printed the startup summary with the memory estimate. Open: **the acceptance itself** — a live call on the M3 Max with audible replies, transcripts on both sides and a working `response.cancel`, from a browser Realtime client and from aicc with `pcm16` and with `g711_ulaw` — has **not** been run, because the checkpoints in `~/models` are still downloading. Nothing in the serve path has therefore met a real model, an ASR model or a real client; the fake engine and `FakeLLM` are the whole evidence so far. `README.md` documents the steps and marks that verification as pending.

- **Files**: `src/talkover/cli.py`, new `src/talkover/app.py` (wiring of engine thread, protocol loop, Brain), `README.md`.
- **Scope**: `serve` loads config, applies patches, builds backend, engine, ASR, and Brain, and runs uvicorn. Verify a real call from a browser Realtime client and from aicc with `input_audio_format: pcm16` and with `g711_ulaw`.
- **Depends on**: T1.7, T2.7, T2.8.
- **Acceptance**: a call on the M3 Max with audible replies, transcripts on both sides, and a working `response.cancel`; the README documents the steps.

---

## M3 · Business Brain

Acceptance for the milestone: an order lookup completes during a call and the order number is read back; keyword transfer fires within 1 s.

### T3.1 Tool schemas and execution — `done`

- Verified 2026-09-17: `uv run pytest tests/brain/test_tools.py` passes (23 tests) covering success, timeout, and 500. Endpoint paths `/tickets` and `/orders` recorded in DESIGN.md 6.6 for T3.6.

- **Files**: `src/talkover/brain/tools.py`, `tests/brain/test_tools.py`.
- **Scope**: JSON schemas for `query_ticket(ticket_id | phone)`, `query_order(order_id | phone)`, `transfer_to_human(department, reason)`. HTTP execution for the two lookups through `httpx` against `brain.business_api.base_url` with `timeout_sec` (3 s); on timeout or non-2xx return a structured failure the loop turns into "explain and suggest transfer". `transfer_to_human` is never executed locally.
- **Depends on**: T0.2.
- **Acceptance**: tests with a stubbed HTTP server cover success, timeout, and 500.

### T3.2 `BrainLLM` protocol with Anthropic and OpenAI-compatible clients — `done`

- 2026-09-17: unit tests with recorded responses pass for both clients (`uv run pytest tests/brain`: 80 passed, 1 skipped). Open: the manual DeepSeek smoke was not run because `DEEPSEEK_API_KEY` is unset; run `uv run pytest tests/brain/test_llm.py::test_deepseek_live_tool_call` with the key set. `model: deepseek-flash` is unverified against the live endpoint. T3.3 is not blocked (it uses `FakeLLM`).
- Verified 2026-09-18 on the M3 Max: the live smoke ran with `DEEPSEEK_API_KEY` set — `uv run pytest tests/brain/test_llm.py::test_deepseek_live_tool_call -q -rs` gives 1 passed (no longer skipped), which closes the manual-smoke acceptance item. Endpoint verified: `kind: anthropic`, `base_url: https://api.deepseek.com/anthropic` (`POST /anthropic/v1/messages`), `model: deepseek-flash`, `x-api-key` plus `anthropic-version: 2023-06-01`. The endpoint echoes `"model": "deepseek-flash"`, answers with `stop_reason: tool_use` and a `tool_use` block (`query_order` / `{"order_id": "SO20260917001"}`, id in DeepSeek's `call_00_…` form rather than Anthropic's `toolu_…`, which the client only ever echoes back), and the second turn carrying a `tool_result` block comes back `end_turn` with the order status in the text — so the full Brain-loop round trip works unchanged. No client or config change was needed; the defaults in `src/talkover/config.py` and `configs/*.yaml` are confirmed as written. Two observations for CUDA/A10 parity work, neither a defect: DeepSeek prepends a `thinking` block, which `anthropic.py` already skips, and its `usage` carries extra `cache_creation_input_tokens` / `cache_read_input_tokens` / `service_tier` fields, which the parser ignores. Recorded-response tests re-run green: `uv run ruff check . && uv run ruff format .` clean, `uv run pytest tests/brain -q` gives 215 passed, 0 skipped. No live test exists for `openai_compat` against DeepSeek and no OpenAI-compatible DeepSeek URL is present in the configs, so that half was not run.

- **Files**: `src/talkover/brain/llm/__init__.py`, `anthropic.py`, `openai_compat.py`, `tests/brain/test_llm.py`.
- **Scope**: `BrainLLM.complete(messages, tools) -> LLMResponse` with text and tool-call parts, streaming optional. `anthropic.py` targets the DeepSeek Anthropic-compatible endpoint by default (`base_url`, `model: deepseek-flash`, `DEEPSEEK_API_KEY`) and works unchanged against Anthropic's API. `openai_compat.py` covers vLLM / local servers. Include a `FakeLLM` in `tests/brain` for scripted tool calls.
- **Depends on**: T0.2.
- **Acceptance**: unit tests with recorded responses for both clients; a manual smoke against DeepSeek recorded in the PR.

### T3.3 Brain session: function-calling loop — `done`

- Verified 2026-09-17: `uv run pytest tests/brain` gives 162+ passed with `FakeLLM`: order lookup yields `Share` then `Result`; a fork query leaves the main conversation unchanged; the round cap (6 per invocation) is enforced. `brain.spoken_numbers`, `brain.max_rounds`, `brain.fallback_texts` added to the schema.

- **Files**: `src/talkover/brain/session.py`, `tests/brain/test_session.py`.
- **Scope**: `BrainSession` per `TaskRequest`: system prompt (business phrasing + tool descriptions, carries the phrasing that cannot fit in the 256-token slate), loop of at most N rounds, `share(text)` after each lookup result, `TaskResult` at the end. `TaskUpdate` (main) appends to the same conversation; `TaskQuery` (fork) runs on a read-only copy. Numbers in `share` text are rewritten into segmented spoken form when a config flag is on (risk mitigation from DESIGN.md section 11).
- **Depends on**: T3.1, T3.2.
- **Acceptance**: with `FakeLLM`, a scripted order lookup produces `share` then `TaskResult`; a fork query does not mutate the main conversation; the round cap is enforced.

### T3.4 `WorkerProvider` implementation and registration — `done`

- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean and `uv run pytest tests/brain tests/protocol -q` green (477 passed, 1 skipped); `tests/brain/test_provider.py` (21 cases) covers the registry key, the full task lifecycle against `FakeLLM` (start → share → result, update, fork query, transfer interaction → respond → resolve) and the Gateway-facing `WorkerRunChannel` path. Fork-lane events are emitted as `ShareEvent(kind="answer")` including the closing `Result`, because a `ProviderEvent(kind="result")` there would reach the Gateway as a `DonePayload` and end the main task. `TaskUpdate(mode="replace")` is treated as `additive`; `mode="cancel"` closes the run.

- **Files**: `src/talkover/brain/provider.py`, `tests/brain/test_provider.py`.
- **Scope**: implement `gander_runtime`'s `WorkerProvider`, implement the provider → project → run chain (DESIGN.md 6.2), translate `TaskRequest` / `TaskUpdate` / `TaskQuery` into `BrainSession.start / update / query`, emit `ProviderEvent`s, carry `transfer_to_human` as `ProviderEvent(kind="interaction")` with a `TaskInteraction` (upstream has no `client_tool` kind; DESIGN.md 6.3), route the run's `respond()` to `BrainSession.resolve()`, and register under key `business` in `ProviderFactoryRegistry`.
- **Depends on**: T3.3.
- **Acceptance**: the provider is resolvable by key; a full task lifecycle runs against `FakeLLM`.

### T3.5 Transfer pre-interception — `done`

- Verified 2026-09-18: `uv run ruff check src/talkover/brain tests/brain && uv run ruff format src/talkover/brain tests/brain` clean, `uv run pytest tests/brain tests/protocol -q` green (508 passed, 1 skipped), of which `tests/brain/test_intercept.py` is 31 new cases. Measured latency from `task_start` to the emitted `transfer_to_human` interaction: median 0.03 ms, max 0.15 ms over 50 in-process runs, against the 50 ms budget the test asserts, with `FakeLLM.call_count == 0` on every hit. Matching folds NFKC and case, then drops whitespace, punctuation and symbols, so "转人工！", "转人工, 谢谢" and "转 人 工" all hit; a keyword that normalises to nothing is dropped instead of matching everything. Wired into `BusinessProject.open_run`, so both the Gateway path and T3.7 get it: a hit records `BusinessRun.intercepted` and calls `emit_client_tool` instead of `run.start()`, `intercept=False` forces the loop, and `respond()` closes the task with a completed `TaskResult`. The emitted call carries only `department: "general"`; a `reason` would become the interaction `prompt` a client without tool support speaks, and an English sentence mid-call is worse than the provider's `TRANSFER_PROMPT` fallback. Repo-wide `uv run ruff check .` still reports two F401 in `src/talkover/realtime/session.py` from another task in flight; nothing in this task touches that file.

- **Files**: `src/talkover/brain/intercept.py`, `src/talkover/brain/provider.py` (task-start wiring), `tests/brain/test_intercept.py`.
- **Scope**: on `task_start`, match the trusted text against `brain.transfer_keywords` (substring match, normalised whitespace and punctuation) and emit `transfer_to_human` directly without an LLM call. Keywords are runtime data and stay in Chinese in the config.
- **Depends on**: T3.4.
- **Acceptance**: hit and miss cases; measured latency from `task_start` to the emitted event under 50 ms in the test.

### T3.6 Mock business API — `done`

- Verified 2026-09-17: `tests/brain/test_tools.py` and `tests/brain/test_mock_api.py` run against the mock in-process (37 tests pass); `uv run talkover mock-api` is registered and served `/healthz` and `/orders` on a test port. Default address 127.0.0.1:9100.

- **Files**: `src/talkover/brain/mock_api.py`, `src/talkover/cli.py` (`talkover mock-api` subcommand), `configs/brain.business.example.yaml`.
- **Scope**: FastAPI app serving `query_ticket` and `query_order` with a small fixture dataset including realistically formatted order numbers and amounts; optional artificial delay to exercise the timeout path.
- **Depends on**: T3.1.
- **Acceptance**: `tests/brain/test_tools.py` runs against the mock in-process; `uv run talkover mock-api` serves on the configured port.

### T3.7 Realtime to Brain wiring — `done`

- Verified 2026-09-18: `uv run ruff check . && uv run ruff format .` clean, `uv run pytest tests/protocol tests/brain -q` gives 727 passed, 19 skipped, **0 xfailed** (the 19 skips are T2.7's 18 Cascade-only cases plus the opt-in DeepSeek smoke), of which `tests/protocol/test_function_call.py` is 11 new cases; `uv run pytest -q` gives 859 passed, 22 skipped. The bridge is the new `src/talkover/realtime/brain_bridge.py`; `src/talkover/app.py` is the composition skeleton (`build_app`) that gives every connection its own `BrainBridge` and closes it from `create_app(on_release=...)`. **How the bridge obtains the run**: a `BusinessProject` is injected, and each Cerebellum `task_start` (which arrives as an `EngineStepEvent` with `is_tool_call`, through the `on_engine_event` hook) opens one with `BusinessProject.open_run(TaskRequest, autostart=True, intercept=True)` — no `ProviderFactoryRegistry`, because there is no Gateway in-process and no `WorkerRequest` to compile; `TaskRequest.instruction` is the latest trusted ASR text (the task name when ASR has produced none yet), which is what the T3.5 keyword rule matches. Decisions recorded in `docs/protocol-profile.md` §9.1 and §8.4: the call is emitted **even when the client declared no `transfer_to_human` in `session.tools`** (the transfer decision is the Brain's; the omission is logged once), an unknown `call_id` is rejected with `invalid_value` / `item.call_id` *after* the item's `conversation.item.created`, `share` / `question` / result go to the Cerebellum and never to the client, and a Brain failure is logged rather than raised as the fatal `engine_error`. Also in this task: the T2.7 defect is fixed — `_merge_audio` now replaces `audio.input.turn_detection` when `type` changes instead of merging into it, and `session_turn_detection_semantic_vad` lost its `xfail`; a native tool-call unit no longer opens a speech response in `mapping.py`; `RealtimeSession._invalid_value` became the public `invalid_value` so the bridge raises rejections with the right `event_id`. Open: the Cerebellum-facing half is stubbed behind `brain_bridge.CerebellumChannel` because `EngineProtocol` exposes neither upstream `feed_tool_response` nor `feed_runtime_event` — without it the model's `task_start` stays unanswered and Brain shares are dropped; that engine API and its wiring are T2.9 / T1.4 (listed for DESIGN.md 5.3 and 4.x).
- 2026-09-18: the channel is now real. `EngineProtocol.feed_tool_response` / `feed_worker_delivery` landed with T1.4 (DESIGN.md 4.6), so `brain_bridge.CerebellumChannel` is gone: `BrainBridge(project, engine=...)` holds the engine and `app.py` hands it the session's own. The model's `task_start` is answered while it waits, and Brain `share` / `result` reach it as `milestone` / `final` worker deliveries addressed by the task name the model chose, instead of being logged and dropped. Nothing the client sees changed. Still open for this task: `task_send` / `task_resolve` are recognised and logged but not acted on, and 4.6 is unvalidated against real weights until T1.4's `mps` acceptance runs.

- **Files**: `src/talkover/realtime/brain_bridge.py`, `src/talkover/realtime/session.py`, `src/talkover/realtime/mapping.py`, `src/talkover/app.py`, `tests/protocol/test_function_call.py`, `tests/protocol/test_cascade_cases.py`, `docs/protocol-profile.md`.
- **Scope**: `transfer_to_human` interactions from the Brain (`ProviderEvent(kind="interaction")`, DESIGN.md 6.3) become the `function_call_arguments.delta / .done` pair; `conversation.item.create` with `function_call_output` is routed to the run's `respond()` and on to `BrainSession.resolve()` (not upstream `task_resolve`, which is an authorization action); `session.tools` only needs to declare `transfer_to_human` and the lookup tools are never exposed. Brain `share` / `question` are not surfaced to the client.
- **Depends on**: T2.7, T3.4, T3.5.
- **Acceptance**: fake-engine test proves the round trip transfer request → client → `function_call_output` → `respond()` → `BrainSession.resolve()`.

### T3.8 End-to-end Brain acceptance — `todo`

- **Files**: `tests/brain/test_e2e.py` (marker `mps` or `cuda`), `docs/DESIGN.md` section 11 if the number-reading mitigation is activated.
- **Scope**: live call with the mock API: ask for an order, verify the order number is read back correctly on a listening test with realistically formatted digits; say a transfer keyword and measure time to the `function_call` event.
- **Depends on**: T2.9, T3.6, T3.7.
- **Acceptance**: order number read back correctly; transfer event within 1 s; results recorded in the porting notes.

---

## M4 · CUDA backend on the A10 — deferred

Not part of this phase. Listed so the interfaces built in M1 stay compatible with it. Acceptance when it is picked up: all M1–M3 cases pass on the A10 with only `engine.device` changed; VRAM within 24 GB; per-unit time < 1.0 s for 10 minutes.

Interfaces this phase must keep for M4:

- `DeviceBackend` with `CudaBackend` in `engine/backend/cuda.py` and `get_backend("cuda:<n>")` parsing (T1.1).
- `engine.talker_device` config key and the detached-Talker branch left reachable in `engine/talker.py` (T1.5).
- `AsrBackend` protocol with `faster_whisper.py` present and `asr.device: cpu` / `compute_type: int8` honoured by `create_asr` (T1.6).
- `memory.py` CUDA branch including the int8 Thinker and `init_vision: false` options (T1.2).
- `bench_rtf.py` token-sequence JSON dump for cross-backend comparison (T1.8).
- Config schema accepts `attn_implementation: flash_attention_2`; the `cuda` extra in `pyproject.toml` stays declared.

### T4.1 CUDA backend validation — `deferred`

- **Scope**: `uv sync --extra dev --extra cuda` on the A10; `talkover check` with `device: cuda`; run `tests/engine -m cuda`, `bench_rtf.py`, and compare the token sequence against the MPS dump from T1.8. Differences go to `docs/mps-porting-notes.md`.
- **Depends on**: T1.8, T1.6.
- **Acceptance**: bench numbers and the diff are recorded.

### T4.2 VRAM fallback ladder — `deferred`

- **Files**: `src/talkover/engine/session.py`, `src/talkover/engine/memory.py`, `configs/serve.example.yaml` comments.
- **Scope**: implement the three-step fallback from DESIGN.md section 4.5 as config options: ASR on CPU int8, int8 Thinker linear layers via `torchao`, `init_vision: false`; detached Talker on a second card when available. `memory.py` must model each option.
- **Depends on**: T4.1, T1.2.
- **Acceptance**: the chosen configuration loads on the A10 without OOM and `talkover check` predicted it correctly.

### T4.3 Stability and full regression on the A10 — `deferred`

- **Scope**: 10-minute continuous call; all `tests/protocol`, `tests/brain`, `tests/engine -m cuda`; M3 end-to-end case.
- **Depends on**: T4.2, T3.8.
- **Acceptance**: no OOM, no per-unit time above 1.0 s over 10 minutes; numbers in the porting notes.

### T4.4 Optional CUDA accelerations — `deferred`

- **Scope**: `attn_implementation: flash_attention_2`, faster-whisper on CUDA float16 when VRAM allows, upstream detached Talker on multi-card hosts.
- **Depends on**: T4.3.
- **Acceptance**: measured gain recorded; defaults in `serve.example.yaml` updated only if the gain is real.

---

## M5 · MPS real-time target

### T5.1 MPS stability at the real-time target — `todo`

- **Scope**: 10-minute continuous call on the M3 Max with per-unit time < 1.0 s; profile and fix operator fallbacks found in T1.9 where a PyTorch-level fix exists.
- **Depends on**: T1.9, T3.8.
- **Acceptance**: numbers in the porting notes; if the target is met, M5 closes here and this phase is complete.

### T5.2 Phase-two kickoff: MLX Thinker — `blocked` (until T1.9 or T5.1 fires the trigger)

- **Scope**: a design addendum in `docs/DESIGN.md` for porting the Thinker to MLX with 4-bit / 8-bit quantization while Talker and token2wav stay on PyTorch MPS; a `mlx` backend that still satisfies `DeviceBackend`. No implementation tasks are listed until the addendum is accepted.
- **Depends on**: T1.9, T5.1.
- **Acceptance**: addendum merged and its own task list appended to this file.

---

## Cross-cutting rules

- **CUDA seams**: no task in this phase may remove or bypass an interface listed under M4; engine code still never references `torch.cuda` / `torch.mps` outside `engine/backend/`.
- **Upstream**: never fork. New CUDA binding points found during any task go to `patches.py` with a porting-notes entry and an upstream patch. Upgrading the pinned commit is its own task with a porting-notes update.
- **Docs**: DESIGN.md is updated in the same change as any behavior it describes. `README.md` gets the user-facing steps at T2.9 and T3.8.
- **Style**: `uv run ruff check . && uv run ruff format .` before closing any task. English everywhere except `transfer_keywords` values.
- **Verification**: the closing note of a task states which acceptance command was run and its result.
