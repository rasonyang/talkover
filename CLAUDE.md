# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Talkover wraps the Gander full-duplex interaction model (built on MiniCPM-o 4.5) as an OpenAI Realtime-compatible WebSocket service, plus a customer-service Brain provider. It targets single-machine operation on Apple Silicon (MPS) first, with NVIDIA A10 (CUDA) as the second backend.

**Current state: M0 skeleton.** Under `src/talkover/`, everything except `cli.py` and `config.py` is a one-line docstring; `talkover serve / check / bench` all return "not implemented". Read `docs/DESIGN.md` before implementing anything. It is the source of truth for module responsibilities, the event mapping tables, config layout, and milestones M0–M5. Design changes must be reflected there.

All documentation, comments, and code are in English. The only Chinese strings are runtime data (the `transfer_keywords` phrases customers say).

## Common commands

```bash
uv sync --extra dev --extra mps          # macOS dev environment (use --extra cuda on CUDA machines)
uv run pytest                            # all tests (testpaths=tests, asyncio_mode=auto)
uv run pytest tests/protocol/test_smoke.py::test_import   # single test
uv run pytest -m cpu                     # GPU-free tests only; other markers: mps, cuda
uv run ruff check . && uv run ruff format .   # line-length 100, py312
uv run talkover check -c configs/serve.example.yaml
uv run talkover serve -c configs/serve.example.yaml
scripts/download_models.sh               # hf CLI download of weights into ~/models (override with MODELS_DIR)
```

Python is pinned to 3.12. Package management is `uv` only, never conda (upstream uses conda + 3.10; this repo does not follow it).

## Upstream dependencies (hard constraints)

- `mcpmft[talker]` and `gander-runtime` are **path dependencies** pointing at `../Omni-Interaction-Agent/{minicpm_ft,gander_runtime}`, installed editable. That directory must sit next to this repo or `uv sync` fails.
- Upstream is pinned to commit `cf43838` (recorded in `docs/mps-porting-notes.md`). Upgrading upstream is an explicit step: update that line and re-run `uv sync`.
- **Never fork upstream.** Upstream CUDA binding points are bypassed through Talkover's adapter layer (`engine/backend/`). Monkeypatches that cannot be avoided live only in `engine/patches.py` and are contributed back upstream as minimal patches.
- `eva-decord` has no wheel for macOS + 3.12; `pyproject.toml` restricts it to Linux via `[tool.uv] override-dependencies`. `media_mode: voice` is unaffected.
- torch is pinned `>=2.6,<2.7`, `numpy<2`, transformers `==4.51.0` (all upstream constraints).
- Every CUDA binding point handled on MPS, every operator fallback, and every Python 3.12 compatibility issue gets appended to `docs/mps-porting-notes.md`.

## Architecture

Three packages, three layers. Single process, three asyncio task groups; model inference runs on a dedicated thread, the protocol layer and Brain run on the event loop:

```
realtime (protocol)  --feed_pcm16-->  engine (Gander inference)  --task_start/send/resolve-->  brain (business LLM)
         <--DuplexStepEvent--                                    <--share/question/result--
```

- **`talkover.engine`** wraps upstream `DuplexLiveSession`. Core rule: **engine code must never reference `torch.cuda` or `torch.mps` directly**. All device operations go through the `DeviceBackend` protocol (`backend/mps.py`, `cuda.py`, `cpu.py`; `cpu` is for CI smoke only). Talker runs in-process on the same device as Thinker, on its own thread; upstream's detached mode is used only for CUDA multi-GPU. ASR is a side channel (it produces transcripts and trusted text for Brain), so its backend is swappable: `mlx-whisper` by default on MPS, `faster-whisper` on CUDA. `memory.py` estimates memory before startup (A10's 24 GB is borderline; see DESIGN.md 4.5).
- **`talkover.realtime`** is FastAPI + WebSocket: `GET /v1/realtime` with a single Bearer key, and `GET /health`. External audio is pcm16 at 24 kHz; input is resampled to 16 kHz and fed to the model in 1 s units. The event mapping is in DESIGN.md 5.3. `turn_detection` is approximate (Gander has no explicit VAD). `session.instructions` only goes into the task slate (≤256 tokens); business prompts belong in Brain. One session per process; a second connection gets `engine_busy` / HTTP 503. Protocol validation rules and regression cases come from `cascade-realtime-gateway`'s `docs/protocol-profile.md`; reuse its docs and test cases only, never its code.
- **`talkover.brain`** implements the `gander_runtime` `WorkerProvider` protocol, registered under key `business`. Three tools: `query_ticket` and `query_order` call the business HTTP API inside Brain (3 s timeout); `transfer_to_human` is not executed but returned to the client as a `function_call`. `intercept.py` does rule-based keyword matching on `task_start` to trigger transfer without entering the LLM loop. The LLM is abstracted as the `BrainLLM` protocol with `llm/anthropic.py` and `llm/openai_compat.py`. Default LLM is DeepSeek 4.1 Flash through DeepSeek's Anthropic-compatible endpoint (`kind: anthropic`, `base_url: https://api.deepseek.com/anthropic`, `model: deepseek-flash`, key from `DEEPSEEK_API_KEY`).

Real-time budget: each 1 s causal unit must finish end-to-end in < 1.0 s (Thinker ≤ 0.45 s, Talker ≤ 0.35 s). `scripts/bench_rtf.py` outputs per-stage timings and a fixed-seed token sequence for MPS vs CUDA comparison.

## Test conventions

- `tests/protocol` and `tests/brain` do not depend on a GPU; they use a fake engine.
- `tests/engine` uses pytest markers `mps` / `cuda` / `cpu`. New engine tests must carry a marker.

## Non-goals

Multi-session concurrency, training, video/screen sharing, Admin API, multiple profiles, and the MLX port (phase two, triggered if Thinker single-step p95 on MPS exceeds 0.6 s).
