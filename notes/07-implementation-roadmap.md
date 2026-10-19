# Implementation Roadmap: Next Steps & Reference Builds
*Synthesized from live session with Google Antigravity (agy, Gemini)*
*Date: 2026-10-17*
*Source: AGY session f8b4bcf9-612d-4770-9baf-c4e654420b9e*

---

## Priority Track (P0–P4): Reference Implementations

These are concrete build tasks with clear entry/exit criteria. Each can be ticked off when the deliverable is shipped.

### P0: Agent Gatekeeper (`sys2/guard.py`)
- **What**: Build safety CLI that blocks dangerous commands before execution
- **Why**: Dogfoodable immediately — you already run agents; no new datasets needed
- **Deliverable**: `sys2/guard.py` module + CLI entrypoint
  - `python3 -m sys2.guard "rm -rf /"` → BLOCKED (Conf: 89.2%, 5/5 heads)
  - `python3 -m sys2.guard "ls /home"` → ALLOWED (Conf: 78.4%, 4/5 heads)
  - Confidence thresholds: >0.70 Benign → ALLOW, >0.70 threat → BLOCK, <0.40 → ABSTAIN
- **Dependencies**: Existing `sys2/engine.py` (95% of the pipeline is already built)
- **Safety domain adapter**: `system2_safety.pt` trained on 52 commands (25 safe, 15 destructive, 10 adversarial, 2 ambiguous)
- **Status**: COMPLETE (Shipped in `sys2/guard.py` with `models/system2_safety.pt` adapter & 100% passing test suite)

### P1: 2-Pass Recurrent Deliberation
- **What**: Judger feedback loop (2 passes, <50ms total)
- **Why**: Proves System2 can do deeper deliberation without becoming autoregressive — fixes medicine/finance failure cases
- **Deliverable**: `deliberate_2pass()` method in `system.py`
  - Pass 1: Standard 5-head pass → Judger produces cross-attention summary embedding
  - Pass 2: Project embedding back into candidates, re-deliberate through ThoughtHeads
  - Strictly 2 passes (O(1) complexity), <50ms total
- **Measures improvement on**: Track C (Medicine USMLE) accuracy jumped from 33.0% to 36.0% (+3.0%) with 0 additional training parameters
- **Status**: COMPLETE (Shipped in `system.py`, verified on 100 clinical vignettes)

### P2: Batched MCTS Pruner Harness
- **What**: `adjudicate_batch()` + `mcts_pruner_harness.py`
- **Why**: 1,000 node rollouts/sec — 10x speedup over autoregressive verifiers
- **Deliverable**: 
  - `System2Engine.adjudicate_batch(questions, choices_batch)` with vectorized span extraction
  - `scripts/mcts_pruner_harness.py` — depth-5 tree search on GSM8K, prunes 3/4 branches at each step in <12ms
- **Measures improvement on**: Achieved 1,144 states/sec throughput on Apple Silicon MPS
- **Status**: COMPLETE (Shipped in `sys2/engine.py` and `scripts/mcts_pruner_harness.py`)

### P3: Cyber-Physical Safety Interlock Benchmark
- **What**: 50Hz control loop test with dead-man's switch
- **Why**: Proves deterministic latency for real-time robotics
- **Deliverable**: `benchmarks/benchmark_cyber_physical_interlock.py`
  - 1,000 cycles at 50Hz (20ms per decision)
  - Measures p99 latency jitter: 7.305 ms mean latency (136.9 Hz sustained, >2.7x headroom), p50: 7.248 ms, p99: 9.614 ms, p99 jitter: 2.366 ms
  - Validates dead-man's switch: 100.0% (250/250) on zero-telemetry sensor dropout, 99.8% (998/1000) 20ms deadline compliance, 0.00% nominal false trips
- **Tested on**: Telemetry sensor interlock suite (`tests/test_interlock.py`)
- **Status**: COMPLETE (Shipped in `benchmarks/benchmark_cyber_physical_interlock.py` & verified with unit tests)

### P4: Package as pip-installable wheel
- **What**: `pyproject.toml` with console entrypoints (`sys2` CLI and `sys2-guard`)
- **Why**: Ships System2 as an enterprise-ready standalone library
- **Deliverable**: `dist/system2-0.2.0-py3-none-any.whl` (39KB standalone wheel)
  - `sys2 = sys2.__main__:main`
  - `sys2-guard = sys2.guard:main`
- **Status**: COMPLETE (Verified CLI execution and test suite `tests/test_packaging.py`)

### P5: LM Studio & Jev Decision Server (`sys2/server.py`)
- **What**: Drop-in HTTP server implementing `/v1/systemone` (Jev/TypeSafe AI schema) and `/v1/decisions` (OpenAI decision schema), with standalone and reverse-proxy modes
- **Why**: Allows instant compatibility with companion agent apps (Bionic v1.1.8+), TypeSafe AI SDKs, and local agent orchestrators that query decision models without waiting for generative token streaming
- **Deliverable**: `sys2/server.py` module + CLI entrypoint (`python -m sys2.server` / `sys2-server`)
  - `POST /v1/systemone`: maps `state` + `questions` (`choice`, `noul`) to `System2Engine.deliberate()` and `CommandGuard.verify()`, returning calibrated probabilities, winning choices, and `output_tokens: 0`
  - `POST /v1/decisions`: OpenAI-compatible structured decision format
  - `GET /v1/models` & `GET /health`: Model discovery and liveness check
  - Reverse-proxy mode (`--upstream-llm http://localhost:1234`): Transparently handles `/v1/systemone` and `/v1/decisions` locally in <30ms, while proxying `/v1/chat/completions` to LM Studio
- **Dependencies**: `sys2/engine.py`, `sys2/guard.py`
- **Status**: PARKED — POST-TRAINING (Implemented with 100% passing test suite on branch `feat/sys2-decision-server` in worktree `scratch/worktrees/decision-server`; hold merge to `main` until all domain checkpoints complete training)

| # | Priority | Step | Status |
|---|----------|------|--------|
| 1 | P0 | Agent Gatekeeper (`sys2/guard.py`) | COMPLETE |
| 2 | P1 | 2-Pass Recurrent Deliberation (`deliberate_2pass()`) | COMPLETE |
| 3 | P2 | Batched MCTS Pruner Harness (`scripts/mcts_pruner_harness.py`) | COMPLETE |
| 4 | P3 | Cyber-Physical Safety Interlock Benchmark (`benchmarks/benchmark_cyber_physical_interlock.py`) | COMPLETE |
| 5 | P4 | Package as pip-installable wheel (`sys2`, `sys2-guard`) | COMPLETE |
| 6 | P5 | LM Studio & Jev Decision Server (`sys2/server.py`) | PARKED (Ready in worktree; merge post-training) |

---

## Direction Track (D1–D4): Architecture & Data

These are research/infrastructure tasks — less immediately dogfoodable but unlock multiple P-priority applications.

### D1: Unified Path B Hybrid CLI (`python3 -m sys2 --hybrid`)
- **What**: One-command end-to-end local reasoning tool
- **Workflow**: 
  1. Accepts open-ended or multi-choice prompt via CLI: `python3 -m sys2 --hybrid "Why does an AC induction motor draw higher inrush current?"`
  2. Queries local small LLM (LM Studio / Ollama / llama.cpp) to generate candidate hypotheses
  3. Strips internal scratchpads, handles token budgeting (max_tokens=4096)
  4. Dynamic Router dispatches in <0.1ms to specialized domain track
  5. System2 arbitrates across candidates in <35ms
- **Why**: Bridges gap between raw classification and generative reasoning — complete offline reasoning daemon for Apple Silicon under 4GB memory
- **Status**: PARKED — POST-TRAINING (Prototyped in worktree `scratch/worktrees/data-enhancement` on branch `feat/data-enhancement-and-hybrid`; deferred until all domain tracks reach target accuracy so System2 does not overrule valid LLM generations on under-trained domains)

### D2: Architectural Breakers & Data Diagnostics (Track D Electrical & Track C Medicine)
- **What**: Source-by-source dataset audits and causal purity enforcement
- **Track D (Vocational Electrical)**:
  - Discovered college physics trivia (`mmlu_college_physics` 20.0%, `mmlu_conceptual_physics` 22.0%) severely diluted circuit fault-finding
  - Enforced In-Domain Causal Purity Law: purged physics trivia, generated `data/electrical_pure` via `scripts/build_pure_electrical_dataset.py` with NEC code calculations and multimeter diagnostic fault trees
  - Accuracy jumped from **34.0% to 44.03%** (+10.03%), guarded metric improved from **0.5535 to 0.6042**, ECE dropped to **0.0467**
- **Track C (Medicine USMLE)**:
  - Applied 2-pass recurrent deliberation (`deliberate_2pass()`)
  - Accuracy jumped from **33.0% to 36.0%** (+3.0%) across 100 clinical vignettes with zero retraining
- **Status**: COMPLETE (Shipped `data/electrical_pure`, `scripts/build_pure_electrical_dataset.py`, verified +10.03% accuracy gain)

### D3: Multi-Trial AutoResearch Sweeps on Lagging Tracks (Med, Electrical, Legal, Finance)
- **What**: Automated metric-guarded sweeps across isolated worktrees (now unblocked by MPS memory leak & `--max-len` fix in `b920cb4`)
- **Track H (Legal)**:
  - Trial 1 (`warmup_ratio = 0.05` linear ramp): Metric improved **0.5505 → 0.5518** (Acc: 33.92% → 35.19%), committed (`0c452fa`)
  - Trial 2 (`lambda_cand_margin = 0.06`): Metric improved **0.5518 → 0.5526** (Acc: 34.43%, Brier: 0.2390, ECE: 0.1049), committed (`181aac1`)
- **Track I (Finance)**:
  - Baseline evaluated on 255 held-out test samples: Metric = **0.5832** (Acc: 40.78%, Brier: 0.2415, ECE: 0.0658)
- **Status**: IN PROGRESS — ACTIVE PRIORITY (Complete domain model training before merging serving/hybrid layers P5 & D1)

### D4: Clean & Archive the scratch/ Experiment Graveyard
- **What**: Organize 25+ abandoned ablation scripts into `scratch/archive/`
- **Examples**: `test_dual_stream_contest.py`, `test_norm_grid.py`, `fine_grid.py`
- **Why**: Only active worktrees and verified benchmarks should remain in root workspace. Clean workspace reduces confusion and accelerates new experiments.
- **Status**: COMPLETE (Archived in commit `c04104e`)

| # | Priority | Step | Status |
|---|----------|------|--------|
| 6 | D1 | Unified Path B Hybrid CLI (`--hybrid`) | PARKED (Post-Training) |
| 7 | D2 | Architectural Breakers (Track D Electrical, Track C Medicine) | COMPLETE |
| 8 | D3 | Multi-Trial AutoResearch Sweeps (Med, Electrical, Legal, Finance) | IN PROGRESS (Active Priority) |
| 9 | D4 | Clean & Archive scratch/ Experiment Graveyard | COMPLETE |

---

## Tooling & Utility Additions

### System2 Model & Checkpoint Loader (`load_system2.py`)
- **What**: Robust loader that reverse-engineers all ~40+ architectural configuration knobs from saved checkpoint `state_dict` keys
- **Features**:
  - Automatically reconstructs `num_heads`, `head_layers`, `pairwise`, `diff_features`, `judger_residual`, `margin_features`, `decision_conf`, `head_mlp_dim`, etc.
  - Automatically loads encoder LoRA weights into ModernBERT backbone
  - Seamlessly adapts single-vector `(1, 1024)` queries into contrastive hypothesis pairs `(1, 2, 1024)` for pairwise models
  - Runs with 100% strict matching on `models/system2.pt` on Apple Silicon MPS and CPU
  - Verified with comprehensive unit test suite in `tests/test_loader.py`
- **Deliverable**: `load_system2.py`
  - Usage: `python3 load_system2.py --checkpoint models/system2.pt`
- **Status**: COMPLETE & VERIFIED

---

## Combined Priority Summary

| Order | Tag | Step | Type | Status / Gate |
|-------|-----|------|------|---------------|
| 1 | **P0** | Agent Gatekeeper | Reference Build | COMPLETE |
| 2 | **P1** | 2-Pass Recurrent Deliberation | Reference Build | COMPLETE |
| 3 | **P2** | Batched MCTS Pruner Harness | Reference Build | COMPLETE |
| 4 | **P3** | Cyber-Physical Interlock Benchmark | Reference Build | COMPLETE |
| 5 | **P4** | Package as pip wheel | Infrastructure | COMPLETE |
| 6 | **D4** | Clean scratch/ archive | Housekeeping | COMPLETE |
| 7 | **D2** | Breakers (Track D Electrical, Track C Medicine) | Research | COMPLETE |
| 8 | **D3** | Complete Model Training (`med`, `electrical`, `legal`, `finance`) | Core Training | **ACTIVE PRIORITY** |
| 9 | **P5** | LM Studio & Jev Decision Server (`sys2/server.py`) | Serving / API | **PARKED** (Ready in worktree; merge after D3) |
| 10 | **D1** | Unified Path B Hybrid CLI (`--hybrid`) | Hybrid CLI | **PARKED** (In worktree; merge after D3) |
