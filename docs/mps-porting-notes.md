# MPS Porting Notes

Records how each upstream CUDA binding point is handled on Apple Silicon, along with operator fallbacks and Python 3.12 compatibility issues encountered.

## 2026-09-16 Environment

- Upstream Omni-Interaction-Agent is pinned to commit `cf43838` (a path dependency; update this line and re-run `uv sync` when upgrading).
- `eva-decord` (the video decoding dependency of `minicpmo-utils[tts]`) has no wheel for macOS + Python 3.12, and no sdist either.
  It is restricted to Linux via `[tool.uv] override-dependencies` in `pyproject.toml`.
  Impact: the minicpmo-utils video decoding path is unavailable on macOS; `media_mode: voice` is unaffected.
- Verified: torch 2.6.0, `torch.backends.mps.is_available()` is True; mcpmft and gander_runtime import successfully as editable installs.

## 2026-09-17 Device backend (T1.1)

- CUDA binding point: `torch.cuda.synchronize / get_rng_state / set_rng_state / is_available` are reached only through
  `DeviceBackend` (`src/talkover/engine/backend/`). `MpsBackend` maps them to `torch.mps.synchronize`,
  `torch.mps.get_rng_state`, `torch.mps.set_rng_state` and `torch.backends.mps.is_available`.
  A `cpu`-marked test in `tests/engine/test_backend.py` fails the build if any `.py` under
  `src/talkover/engine` outside `backend/` mentions `torch.cuda` or `torch.mps`.
- `torch.mps` and `torch.cuda` are imported lazily inside `get_backend`, so importing
  `talkover.engine.backend` on a machine without the device is safe.
- MPS RNG API (torch 2.6.0): `torch.mps.get_rng_state()` returns a 44-byte CPU `uint8` tensor (CUDA returns 16 bytes);
  it has no per-device index argument, since MPS exposes a single device. The round trip
  `get_rng_state -> randn -> set_rng_state -> randn` reproduces identical tensors, so `bench_rtf.py` (T1.8) can pin a
  fixed seed on MPS the same way it does on CUDA.
- MPS device indexing: `torch.device("mps")` only; `MpsBackend.device(index)` rejects any index other than 0.
  `CudaBackend` keeps the ordinal from `get_backend("cuda:<n>")` and applies it to `synchronize` and the RNG state, which
  is what the M4 detached-Talker branch will need.
- `autocast_dtype()` is bf16 on all three backends. Verified on MPS: `torch.autocast("mps", dtype=torch.bfloat16)`
  produces bf16 matmul outputs and bf16 tensors allocate normally. `torch.float16` autocast is also accepted on MPS,
  but bf16 matches the upstream checkpoints and is what the engine uses.

## 2026-09-17 Upstream monkeypatches (T1.3)

Upstream is not forked. Every patch lives in `src/talkover/engine/patches.py`, one function per binding point, and
`apply_patches(backend)` installs them. `talkover.engine.session` calls it at import time, which is what guarantees the
patches are in place before anything imports `mcpmft.infer`; the first call raises `PatchOrderError` when an
`mcpmft.infer.*` module is already in `sys.modules`. `apply_patches` is idempotent: the wrapper is installed once and
reads the active backend from a module-level slot at call time, so a later call (T1.4, once the YAML config is known)
only retargets it.

Import-time device selection (the config is not loaded yet): `TALKOVER_DEVICE` when set (`mps`, `cuda`, `cuda:<n>`,
`cpu`), otherwise the first available device in the order `mps` -> `cuda` -> `cpu`, with `cpu` as the last resort.
An unknown value raises `ValueError` from `get_backend`.

### Patch 1 — the CUDA gate and device placement in `load_for_infer`

Upstream file: `minicpm_ft/mcpmft/infer/common.py` lines 83-89 (commit `cf43838`). Category: blocks model construction
on a CUDA-less machine, so it is patched now. Minimal upstream patch:

```diff
--- a/minicpm_ft/mcpmft/infer/common.py
+++ b/minicpm_ft/mcpmft/infer/common.py
@@
     model.eval()
     import torch

-    if not torch.cuda.is_available():
-        raise RuntimeError("Gander inference requires CUDA")
     if inference_args.device_map is None:
-        model.to("cuda")
+        model.to(infer_device())
     return InferBundle(model=model, tokenizer=tokenizer, processor=processor)
```

(where `infer_device()` resolves the accelerator instead of hard-coding one).

Talkover implementation: `_patch_load_for_infer_device()` wraps `mcpmft.infer.common.load_for_infer` and runs the
original inside `device_placement_shim(backend)`. The two statements cannot be replaced individually — they sit inside
the function body and resolve `torch` through a function-local `import torch` — so the shim patches the two torch
globals they reach, for the duration of that one call only:

- the CUDA availability check is made to report `True`, so the gate passes;
- `torch.nn.Module.to` rewrites a `"cuda"` / `cuda:<n>` destination to `backend.device()`; every other argument
  (dtypes, other devices) is passed through untouched.

Both originals are restored in a `finally` block, and the shim is a no-op for a CUDA backend. The shim lives in
`src/talkover/engine/backend/shims.py` rather than in `patches.py` because it names device-specific torch attributes,
which DESIGN.md 4.2 confines to `engine/backend/`; the lint-guard test in `tests/engine/test_backend.py` enforces that.
The patch is process-wide while active, so a block must wrap a single model-loading call and nothing else.
No change to the `DeviceBackend` protocol was needed — the shim only uses `name` and `device()`.

Verified: in a fresh interpreter, `import talkover.engine.session` succeeds on this CUDA-less M3 Max, and afterwards
`import mcpmft.infer.common` and `import mcpmft.infer.realtime` (where `DuplexLiveSession` lives) import without error
with `load_for_infer` already wrapped. Covered by `tests/engine/test_patches.py`, which runs the import-order cases in
a subprocess.

### Known CUDA binding points left unpatched

Category (2) — reached only at runtime, in paths the adapter layer handles in T1.4/T1.5:

| Upstream location | Binding | Plan |
| --- | --- | --- |
| `minicpm_ft/mcpmft/infer/detached_talker.py:140-143` | CUDA availability check plus `target.type != "cuda"` in `DetachedTalkerRuntime.from_thinker_model` | Resolved in T1.5: the classmethod cannot be bypassed, so `talkover.engine.talker.build_talker_runtime` does the same construction on the backend device and builds the upstream runtime through its plain constructor. `engine.talker_device` naming a second card still raises `NotImplementedError` pointing at M4 |
| `minicpm_ft/mcpmft/infer/detached_talker.py:155, 210, 246, 324` | `torch.cuda.device(target)` context managers around load, `warm_token2wav`, `synthesize`, `_prepare_voice` | Resolved in T1.5 by `talker_device_shim`: the scope becomes a no-op on a non-CUDA backend, so the runtime itself is reused unmodified (see the T1.5 section) |
| `minicpm_ft/mcpmft/infer/detached_talker.py:212, 225, 229` | CUDA RNG state and `synchronize` in `warm_token2wav` | Resolved in T1.5: the same shim routes them to `DeviceBackend.rng_state` / `set_rng_state` / `synchronize` |
| `minicpm_ft/mcpmft/infer/detached_talker.py:149-152` | `flash_attention_2` for the TTS config when the Thinker uses it | Talkover forces `sdpa` on MPS through `engine.attn_implementation`; no patch needed |
| `minicpm_ft/mcpmft/infer/asr.py:34, 326` | `AsrSettings.device` defaults to `"cuda"`; faster-whisper (CTranslate2) has no Metal backend | ASR is a side channel; T1.6 replaces the backend entirely (`mlx-whisper` on MPS) and never constructs `AsrSettings` |
| `gander_runtime/gander_runtime/asr_process.py:29-30, 123-124` | ASR subprocess pinned to CUDA through `CUDA_VISIBLE_DEVICES` | Talkover runs ASR in-process through its own `AsrBackend`; this module is not used |
| `gander_runtime/gander_runtime/cli.py:314-358, 451, 591-592` | `detached_talker_device` must be `cuda:<index>`, `CUDA_VISIBLE_DEVICES` export | Talkover builds upstream settings itself and never calls this CLI (DESIGN.md 4.1) |
| MiniCPM-o remote code (`model.init_tts` / token2wav, `trust_remote_code` modules under the checkpoint) | Not inspectable without the weights | Unknown until T1.4/T1.7 downloads the checkpoints; any hard-coded CUDA there becomes a new patch here |

Category (3) — training only, ignored: `minicpm_ft/mcpmft/modeling/omni_forward.py:472-479` (`_cross_entropy_fp32`
selects autocast on `device.type == "cuda"`; inference never reaches the loss path) and
`minicpm_ft/mcpmft/data/s3_target.py`.

No new Python 3.12 incompatibility surfaced in this task: `mcpmft.infer.common`, `mcpmft.infer.realtime` and
`mcpmft.infer.detached_talker` all import cleanly under 3.12 on macOS; the only import-level problem so far remains the
`eva-decord` one recorded above.

## 2026-09-17 ASR backends (T1.6)

- `mlx-whisper` 0.4.3 installs and runs on Python 3.12 / macOS arm64 with no patching. It is built on MLX, is
  independent of torch, and touches no CUDA binding point, so it needs no `DeviceBackend` wiring: `create_asr`
  selects it purely from `DeviceBackend.name == "mps"`.
- Model: `mlx-community/whisper-large-v3-turbo` (`weights.safetensors`, 1.61 GB) resolved from the standard
  Hugging Face cache. Direct `huggingface.co` download stalled at roughly 7 MB/min on this network;
  `HF_ENDPOINT=https://hf-mirror.com hf download mlx-community/whisper-large-v3-turbo` completed. Once cached,
  either endpoint is irrelevant at runtime.
- Timing on an M3 Max for the 5.23 s Mandarin fixture (`tests/fixtures/zh_order_16k.wav`) with `language="zh"`:
  2.22 s for the first call including the cold model load from cache, 0.44 s warm (RTF about 0.08). The test in
  `tests/engine/test_asr.py` therefore does a warm-up transcribe first and times the second call only.
- Transcription detail: whisper renders the spoken digits "八六四二" as `8642`, so the fixture assertion accepts
  either spelling.
- `faster-whisper` stays in the `cuda` extra. CTranslate2 has no Metal backend, so on an MPS engine it can only
  resolve to `cpu` + `int8`; `split_device("mps")` raises `AsrError` rather than silently falling back. The CUDA
  ASR path is written but unvalidated until M4.

## 2026-09-17 Engine session (T1.4)

`EngineSession` (`src/talkover/engine/session.py`) wraps the upstream `DuplexLiveSession`. The upstream settings are
built from the typed `TalkoverConfig` directly, never through `gander_runtime/cli.py` (DESIGN.md 4.1):

| Talkover config | Upstream | Note |
| --- | --- | --- |
| `engine.base_model` | `ModelArguments.model_name_or_path` | |
| `engine.thinker_checkpoint` / `talker_checkpoint` | `load_for_infer(checkpoint=..., talker_checkpoint=...)` | composed load: trained tensors from the checkpoint, frozen ones from the base model |
| `engine.dtype` | `ModelArguments.torch_dtype` | bf16, which is also `DeviceBackend.autocast_dtype()` |
| `engine.attn_implementation` | `ModelArguments.attn_implementation` | `sdpa` on MPS |
| `engine.media_mode` | `DuplexLiveConfig.media_mode` | `voice` -> `voice`, `video` -> upstream `omni` |
| `engine.init_vision` | `ModelArguments.init_vision` | forced `False` under `media_mode: voice` |
| `engine.context_max_units` | `DuplexParams.context_max_units` | |
| `engine.sliding_window_mode` | `DuplexParams.sliding_window_mode` | |
| `realtime.memory_slate_max_tokens` | `DuplexParams.memory_slate_max_tokens` | |
| `realtime.trailing_silence_sec` | `DuplexLiveConfig.trailing_silence_sec` | |

`device_map` stays `None`, so placement is the single `model.to(...)` at the end of `load_for_infer` that the T1.3
patch retargets to `backend.device()`. `apply_patches(backend)` runs again on the inference thread, with the configured
backend, immediately before the load.

- **`token2wav_dir` has no config key.** Upstream defaults it to `None`, and `maybe_init_tts` is skipped when it is
  unset, which leaves the Talker silent. The assets are not a separate download: they ship as
  `<engine.base_model>/assets/token2wav` in the MiniCPM-o checkpoint, so `default_token2wav_dir()` resolves them from
  `engine.base_model` and logs a warning when the directory is absent. A config key would be an improvement
  (DESIGN.md 9); it is deliberately not added here because `config.py` belongs to another task.
- **`ref_audio_path` is a constructor argument, not a config key.** Upstream's CLI requires it whenever
  `generate_audio` is on, but `OnlineRunner.prepare` only passes `prompt_wav_path` when it is set, so leaving it
  `None` keeps the checkpoint's default voice. DESIGN.md 5.3 says `voice` is echoed and ignored, which matches.
- **`stop_on_turn_end` is forced `False`.** Upstream's live session is built for a script that ends when the model
  stops speaking; a Realtime session is driven by the client's audio stream and must not close itself.
- **Text user turns have no upstream channel.** `conversation.item.create` with a text message (DESIGN.md 5.3) maps to
  `DuplexLiveSession.feed_runtime_event({"type": "user_text", "content": text})`, which is queued and prefilled inside
  `<tool_response>` markers together with the next unit's microphone audio. The only envelope upstream itself sends
  through that path is `worker_delivery`, so the model's handling of a `user_text` envelope is unvalidated; M3 has to
  confirm it and the envelope may need to change.

Threading: one dedicated inference thread owns the model. The event loop puts commands on a `queue.Queue` and awaits an
`asyncio.Future`; the thread publishes `EngineStepEvent` values through `loop.call_soon_threadsafe` into an
`asyncio.Queue`. Commands are serialized, so `interrupt_output` lands after the unit already in flight — inside one
unit under the real-time budget. `backend.synchronize()` is called at the step boundary before the wall clock is read,
so `EngineStepEvent.step_wall_time_sec` measures compute rather than submission (T1.8 reads it). Any exception on the
inference thread is terminal: it fails the pending call, is re-raised from `events()`, leaves `ready` `False`, and the
thread closes the upstream session and releases the model.

### New CUDA binding points

None. T1.4 needed no new patch beyond the T1.3 `load_for_infer` one, and `src/talkover/engine/session.py` names no
device-specific torch attribute (the lint guard in `tests/engine/test_backend.py` covers it). Converting an upstream
waveform to numpy uses duck typing (`detach()` / `.to("cpu")`) rather than a torch import.

### Talker seam for T1.5

T1.4 uses whatever upstream does in process: with `detached_talker=None` and `DuplexParams.generate_audio=True`, the
Talker and token2wav run synchronously inside `OnlineRunner.streaming_generate`, on the same device as the Thinker, and
the waveform comes back on the step event. `EngineSession` therefore takes a `talker=None` constructor argument and
raises `NotImplementedError` when anything is passed: that argument is where T1.5 injects its own Talker thread.
`engine.talker_device` naming a device other than `engine.device` is refused with `NotImplementedError` pointing at M4,
so the upstream `detached_talker.py` CUDA binding points listed above stay unreachable.

## 2026-09-18 In-process Talker thread (T1.5)

The Talker and token2wav now run on their own thread on the Thinker's device
(`src/talkover/engine/talker.py`), with the CUDA-only call sites redirected by
`talker_device_shim` in `src/talkover/engine/backend/shims.py`. Upstream is not forked: `DetachedTalkerRuntime` is
reused as it stands, and only the call sites below are intercepted.

### How `detached_talker.py`'s CUDA calls are bypassed

| Upstream call | Where | Replacement while `talker_device_shim(backend)` is held |
| --- | --- | --- |
| `torch.cuda.device(target)` (lines 155, 210, 246, 324) | around the load, `warm_token2wav`, `synthesize`, `_prepare_voice` | a context manager that does nothing: on MPS the single device needs no scope, and the ordinal it carries is meaningless. Tensors are still created with `device=self.device`, which the runtime is constructed with (`str(backend.device())`, i.e. `"mps"`) |
| `torch.cuda.get_rng_state(target)` / `set_rng_state(state, target)` (lines 212, 229) | `warm_token2wav`, so the warm-up does not disturb the sampling sequence | `DeviceBackend.rng_state()` / `set_rng_state()`; the device argument is dropped, since MPS exposes one device. The CPU half of that save/restore (`torch.random.*`) is already device neutral and runs unchanged |
| `torch.cuda.synchronize(target)` (line 225) | end of `warm_token2wav` | `DeviceBackend.synchronize()` |
| `DetachedTalkerRuntime.from_thinker_model` (lines 140-155) | the CUDA gate and `target.type != "cuda"` | not called. `build_talker_runtime` performs the same steps — `MiniCPMTTS` from the Thinker's `tts_config`, base tensors plus the Talker overlay through `load_prefixed_submodule_state_dict`, `Token2wav` attached as `audio_tokenizer` — on `backend.device()`, then hands the result to the upstream constructor |

`AsyncTalkerWorker` itself has no CUDA in it, but it has no `synchronize` either, so Talkover replaces it with
`TalkerThread`: the same interface, plus `backend.synchronize()` when a unit is handed over (the hidden states must be
materialized before the Talker thread reads them) and again when a unit is finished (the waveform must be off the device
before the consumer is told). Those two calls are the only device work the Thinker blocks on for the Talker; measured
hand-over cost on the M3 Max is 0.02 ms. The worker is put in place of upstream's by
`talkover.engine.patches.speech_worker_override`, a scoped patch around the single `DuplexLiveSession(...)` call, whose
docstring carries the minimal upstream patch (a `speech_worker_factory` argument).

### New CUDA binding points: the `stepaudio2` vocoder

token2wav is third-party (`stepaudio2`, from `minicpmo-utils[tts]`) and is written for CUDA throughout —
`stepaudio2/token2wav.py` lines 72, 85, 90, 97, 107, 114, 121-122, 131-134, 165-167, 178-184. The same shim covers it:

- `.cuda()` on tensors and modules resolves to `backend.device()`;
- `device="cuda"` passed to the torch factory functions (`tensor`, `zeros`, `ones`, `empty`, `full`, `arange`, ...) is
  rewritten to the same device;
- `torch.amp.autocast("cuda", dtype=...)` is retargeted, and disabled outright when the device cannot autocast that
  dtype — MPS supports only fp16 and bf16, and the vocoder asks for fp32 in its default configuration.

Operator and dtype findings on MPS (torch 2.6.0):

- **float64 is unsupported on MPS.** `Token2wav.__init__` builds its Hamming window with `np.hamming(...)`, which is
  float64, and then calls `.cuda()` on it. The shim narrows a float64 tensor to float32 on the way to an MPS device.
  Nothing else in the vocoder uses float64.
- **`float16=True` aborts the process on MPS.** With half weights the flow decoder mixes f16 parameters with f32
  activations and MPSGraph refuses the graph (`'mps.add' op requires the same element type for all operands and
  results`, then `failed assertion 'original module failed verification'`). token2wav must stay float32 on MPS; the
  CUDA path may keep fp16. `build_token2wav(..., float16=False)` is the default.
- No `PYTORCH_ENABLE_MPS_FALLBACK` is needed: the whole vocoder graph (flow matching + HiFT) runs on Metal.

### Environment gaps found while getting token2wav to run

All three blocked `import stepaudio2` entirely, and are now pinned in `pyproject.toml`:

- `torchaudio` resolved to 2.11.0 against the pinned torch 2.6.0, and its C++ extension failed to load
  (`Symbol not found: _aoti_torch_abi_version`). Pinned `torchaudio>=2.6,<2.7`.
- `librosa` 0.9.0 (pinned through the upstream extras) imports `pkg_resources`, which setuptools removed in 81.
  Pinned `setuptools<81`.
- `onnx` 1.19.0 (pulled in by `s3tokenizer`) needs `ml_dtypes>=0.5.1` for `float4_e2m1fn`, and that needs `numpy>=2`,
  which upstream forbids. Pinned `onnx<1.18` (resolves to 1.17.0, which drops the `ml_dtypes` dependency).

### Measured on the M3 Max (2026-09-18)

`uv run pytest tests/engine/test_talker.py -m mps` drives a fixed 25-token S3 sequence through the thread and gets one
second of 24 kHz audio back:

| Setting | First chunk | Warm chunk (1 s of audio) |
| --- | --- | --- |
| float32, `n_timesteps=10` (upstream default) | 1.0-2.0 s | 0.49 s mean (0.40-0.72 s) |
| float32, `n_timesteps=5` | 0.26 s | 0.24 s |
| float32, `n_timesteps=2` | 0.18 s | 0.14 s |

DESIGN.md 4.4 budgets 0.35 s for the Talker plus token2wav together, so the upstream default already exceeds it with
the AR stage excluded. The flow-matching step count is the obvious knob (quality unmeasured); it is currently fixed at
upstream's 10 in `build_talker_runtime`, and T1.8/T1.9 should decide whether it becomes configuration.

### Still unverified

The AR half of the Talker (`build_talker_runtime`, `warm_token2wav`, `DetachedTalkerRuntime.synthesize`) needs the
MiniCPM-o base model and the Gander Talker checkpoint, neither of which is fully downloaded yet, so the RNG and
device-scope bypasses above are exercised only by the `cpu` shim tests. The same applies to the `EngineSession` path
that hands speak tokens over (`talker_factory`), which is opt-in for exactly that reason: with no factory the Talker
keeps running inside the upstream step, as in T1.4.

## 2026-09-18 deepspeed is incompatible with the pinned torch 2.6 (T1.4 / T1.7)

Every real model load failed at `import transformers.modeling_utils` with:

```
ValueError: infer_schema(func): Parameter partition_sizes has unsupported type list[int].
```

raised from `deepspeed/compile/custom_ops/tp_collectives.py:49`.

Cause. `transformers/modeling_utils.py:157` does `if is_deepspeed_available(): import deepspeed`, i.e. merely having
deepspeed installed pulls it into every model load. deepspeed 0.19.6 then imports its AutoTP custom ops, which declare
`torch.library.custom_op` functions with PEP 585 annotations (`partition_sizes: list[int]`).
`torch._library.infer_schema` in torch 2.6 keys its supported-type table on `typing.List[int]` and does not accept the
builtin generic; support for it arrives in torch 2.7. The torch pin is an upstream constraint and does not move, and the
failure is in a third-party package, so neither a pin change nor an upstream patch applies.

Fix. deepspeed is restricted to Linux through `[tool.uv] override-dependencies` in `pyproject.toml`, the same mechanism
already used for `eva-decord`:

```toml
override-dependencies = [
  "eva-decord; sys_platform == 'linux'",
  "deepspeed; sys_platform == 'linux'",
]
```

This is safe because nothing Talkover runs imports it. `mcpmft` declares `deepspeed>=0.19,<0.20`, but the only mentions
in the upstream tree (commit `cf43838`) are `mcpmft/args.py:333` (a config field), `mcpmft/train/main.py` and
`mcpmft/train/trainer.py` — all training code, and training is an explicit non-goal. `mcpmft/infer/*` and
`gander_runtime` do not mention it at all. With deepspeed absent, `is_deepspeed_available()` is False and transformers
skips the import.

Impact: deepspeed, hjson, msgpack, ninja and py-cpuinfo are no longer installed on macOS. `mcpmft.train.*` cannot be
used locally, which Talkover never does.

Open item for the A10 (CUDA) backend. The incompatibility is platform independent — it is torch 2.6 + deepspeed 0.19.x,
not anything about MPS — so the same `ValueError` will appear on Linux as soon as a real model is loaded there. The CI
job stays green only because `pytest -m cpu` never imports transformers. When the CUDA backend is brought up, either
widen this override to all platforms (Talkover does not train on either backend) or move to torch >= 2.7, which upstream
does not currently allow. No deepspeed release in the upstream-allowed `>=0.19,<0.20` range avoids the annotation.
