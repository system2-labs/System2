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
- **Deliverable**: Modified `system.py` with `deliberate_2pass()` method
  - Pass 1: Standard 5-head pass → Judger produces cross-attention summary embedding
  - Pass 2: Project embedding back into candidates, re-deliberate through ThoughtHeads
  - Strictly 2 passes (O(1) complexity), <50ms total
- **Measures improvement on**: Track C (Medicine) and Track I (Finance) from hybrid benchmark
- **Status**: NOT STARTED

### P2: Batched MCTS Pruner Harness
- **What**: `adjudicate_batch()` + `mcts_pruner_harness.py`
- **Why**: 1,000 node rollouts/sec — 10x speedup over autoregressive verifiers
- **Deliverable**: 
  - `System2Engine.adjudicate_batch(questions, choices_batch)` with vectorized span extraction
  - `scripts/mcts_pruner_harness.py` — depth-5 tree search on GSM8K, prunes 3/4 branches at each step in <12ms
- **Measures improvement on**: Tree search depth achievable vs. autoregressive 70B verifier
- **Status**: NOT STARTED

### P3: Cyber-Physical Safety Interlock Benchmark
- **What**: 50Hz control loop test with dead-man's switch
- **Why**: Proves deterministic latency for real-time robotics
- **Deliverable**: `benchmarks/benchmark_cyber_physical_interlock.py`
  - 1,000 cycles at 50Hz (20ms per decision)
  - Measures p99 latency jitter
  - Validates dead-man's switch: confidence <0.50 → fail-safe triggers
- **Tested on**: nuScenes driving scenarios, robot trajectories (ALOHA/DROID), drone obstacle courses
- **Status**: NOT STARTED

### P4: Package as pip-installable wheel
- **What**: `pyproject.toml` with console entrypoints (`sys2` CLI and `sys2-guard`)
- **Why**: Ships System2 as an enterprise-ready standalone library
- **Deliverable**: `pip install sys2` with `sys2` and `sys2-guard` commands
- **Status**: NOT STARTED

| # | Priority | Step | Status |
|---|----------|------|--------|
| 1 | P0 | Agent Gatekeeper (`sys2/guard.py`) | COMPLETE |
| 2 | P1 | 2-Pass Recurrent Deliberation | NOT STARTED |
| 3 | P2 | Batched MCTS Pruner Harness | NOT STARTED |
| 4 | P3 | Cyber-Physical Safety Interlock Benchmark | NOT STARTED |
| 5 | P4 | Package as pip-installable wheel | NOT STARTED |

---

## Direction Track (D1–D4): Architecture & Data

These are research/infrastructure tasks — less immediately dogfoodable but unlock multiple P-priority applications.

### D1: Unified Path B Hybrid CLI (`python3 -m sys2 --hybrid`)
- **What**: One-command end-to-end local reasoning tool
- **Workflow**: 
  1. Accepts open-ended or multi-choice prompt via CLI: `python3 -m sys2 --hybrid "Why does an AC induction motor draw higher inrush current?"`
  2. Queries local small LLM (LM Studio / Ollama / llama.cpp) to generate candidate hypotheses
  3. Strips internal `` scratchpads, handles token budgeting (max_tokens=4096)
  4. Dynamic Router dispatches in <0.1ms to specialized domain track
  5. System2 arbitrates across candidates in <35ms
- **Why**: Bridges gap between raw classification and generative reasoning — complete offline reasoning daemon for Apple Silicon under 4GB memory
- **Status**: NOT STARTED

### D2: Architectural Breakers for Track C (Medicine) & Track I (Finance)
- **What**: Directly attack the two tracks with near-random accuracy (~30–32%)
- **Track C (Medicine)**:
  - Patient vignettes are 200–500 tokens of symptoms
  - Test `asym_span_mode: "causal_attn"` where choice representations cross-attend back into question's clinical symptom spans before ThoughtHeads
- **Track I (Finance)**:
  - Theoretical physics trivia dilutes circuit fault-finding
  - Ingest pure vocational electrician code calculations, multimeter isolation scenarios to expand train split without trivia contamination
- **Current baseline**: Track C accuracy ~29.77%, Track I accuracy ~66.49%
- **Goal**: Push both tracks above 75% accuracy
- **Status**: NOT STARTED

### D3: Multi-Trial AutoResearch Sweeps on Track H (Legal) & Track I (Finance)
- **What**: Automated metric-guarded sweeps across isolated worktrees
- **Now that**: Both worktrees capped at `--max-len 1024` (5 epochs in ~11 minutes), we can run targeted multi-trial sweeps
- **Track H (Legal)**: Test warmup_ratio: 0.05 linear ramp with cosine annealing cooling → stable Judger cross-attention, then candidate margin tuning (λ_margin ∈ [0.04, 0.08])
- **Track I (Finance)**: Test soft target calibration temperature (τ ∈ [0.45, 0.65]) and asymmetric span weighting
- **Current best**: Metric 0.7395, accuracy 66.5% (both tracks)
- **Goal**: Push both tracks above 75% metric
- **Status**: NOT STARTED (worktrees have 170 trials each, best metric 0.7395, committed=False)

### D4: Clean & Archive the scratch/ Experiment Graveyard
- **What**: Organize 25+ abandoned ablation scripts into `scratch/archive/`
- **Examples**: `test_dual_stream_contest.py`, `test_norm_grid.py`, `fine_grid.py`
- **Why**: Only active worktrees and verified benchmarks should remain in root workspace. Clean workspace reduces confusion and accelerates new experiments.
- **Status**: NOT STARTED

| # | Priority | Step | Status |
|---|----------|------|--------|
| 6 | D1 | Unified Path B Hybrid CLI (`--hybrid`) | NOT STARTED |
| 7 | D2 | Architectural Breakers (Track C Medicine, Track I Finance) | NOT STARTED |
| 8 | D3 | Multi-Trial AutoResearch Sweeps (Track H Legal, Track I Finance) | NOT STARTED |
| 9 | D4 | Clean & Archive scratch/ Experiment Graveyard | NOT STARTED |

---

## Combined Priority Summary

| Order | Tag | Step | Type | Dependencies |
|-------|-----|------|------|-------------|
| 1 | **P0** | Agent Gatekeeper | Reference Build | COMPLETE |
| 2 | **P1** | 2-Pass Recurrent Deliberation | Reference Build | P0 complete, hybrid CLI (D1) |
| 3 | **P2** | Batched MCTS Pruner Harness | Reference Build | P1 (vectorized batching) |
| 4 | **P3** | Cyber-Physical Interlock Benchmark | Reference Build | None (standalone) |
| 5 | **P4** | Package as pip wheel | Infrastructure | P0, P2, P3 |
| 6 | **D1** | Unified Path B Hybrid CLI | Research | None (enables P0–P3) |
| 7 | **D2** | Breakers (Track C + I) | Research | D1 (Path B data format) |
| 8 | **D3** | AutoResearch Sweeps (H + I) | Research | D2 (improved data) |
| 9 | **D4** | Clean scratch/ archive | Housekeeping | Any time |
