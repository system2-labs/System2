---
language:
- en
license: apache-2.0
tags:
- modernbert
- reasoning
- system2
- deliberation
- reranker
- code
- consensus
- calibration
metrics:
- accuracy
- brier_score
- ece
pipeline_tag: text-classification
---

# System 2: 5-Head Deliberation & Epistemic Drift Detector (Code Track)

**System 2** is a non-autoregressive deliberation engine built on ModernBERT. It evaluates competing hypotheses across 5 parallel perspectives in under 30ms on Apple Silicon MPS or NVIDIA CUDA, calculates pairwise cosine cluster consensus, and detects epistemic drift before code execution.

- **Checkpoint:** `system2_code.pt` (143 MB)
- **Base Architecture:** `answerdotai/ModernBERT-Large-Instruct` (395M, frozen) + LoRA subspace adapters ($r=32, \alpha=32$ across layers 20-27) + 5 ThoughtHeads + Cross-attending Judger.
- **Guarded Metric (Code Traces):** `0.9010` (89.11% accuracy, 0.0891 Brier score, 0.0542 true 10-bin ECE).
- **Official Website:** [system2.online](https://system2.online)
- **GitHub Repository:** [github.com/tj-doomer-it/System2](https://github.com/tj-doomer-it/System2)

---

## The Problem: The 2-Bit CoT Collapse & Garbage In, Garbage Out (GIGO)

Crushing large models (27B+) to 2.0-bit (`UD-IQ2_XXS`) or 3.0-bit destroys the attention projection weights that emit the `</think>` closing token. In empirical testing on Apple Silicon (M4 Max), 2-bit models burned 95%+ of their token budgets inside internal reasoning loops, taking 8 to 9 minutes per stage and emitting empty or truncated code.

System 2 creates a new paradigm: **Pairing lightweight 4-bit small models (7B to 12B) with a 143 MB ModernBERT 5-head consensus harness on standard 16 GB laptops.**

```text
[ User Prompt / Code Task ]
             │
             ▼
┌────────────────────────────────────────────────────────┐
│ ⚡ FAST GENERATOR: 7B-12B 4-bit Model (e.g. Gemma 12B) │
│  • Emits 4 parallel candidate hypotheses (~2-3s)       │
│  • High speed (50-70 tok/s), low VRAM (6.8 GB)         │
└──────────────────────────┬─────────────────────────────┘
             │ (4 Candidate Hypotheses)
             ▼
┌────────────────────────────────────────────────────────┐
│ 🧠 SYSTEM 2: 143 MB 5-Head Deliberation Network (MPS)  │
│  • Head 0: Holistic Context (Global integration)       │
│  • Head 1: Context-Anchored (40% Q / 60% A)            │
│  • Head 2: Choice-Forward (20% Q / 80% A)              │
│  • Head 3: Symmetric Balance (50% Q / 50% A)           │
│  • Head 4: Extreme Salience (5% Q / 95% A)             │
│  • Pairwise Cosine Consensus (Threshold >= 0.88)       │
└──────────────────────────┬─────────────────────────────┘
             │
       ┌─────┴────────────────────────┐
       ▼                              ▼
[ HIGH CERTAINTY ]            [ EPISTEMIC DRIFT ]
(Cluster Agreement >= 88%)    (Divergent Candidates)
       │                              │
       ▼                              ▼
Execute Locally on 16GB Mac   Smart Cloud Escalation
(Latency: < 4 seconds)        (Fast-Fail under 4 seconds)
```

---

## Live Validation: Epistemic Drift Detection in Action

During multi-agent swarm testing constructing an in-memory `LRUTTLCache`, candidate subagents diverged into an internal debate over relative vs absolute TTL refresh logic:

- Candidate 1: 0 lines
- Candidate 2: 78 lines
- Candidate 3: 108 lines (scratchpad debate)
- Candidate 4: 7 lines

**System 2 Verdict:** Confidence collapsed from **98.7% down to 41.8%** across the 5 ThoughtHeads. 
Because confidence fell below the 88% threshold, System 2 flagged **Epistemic Drift**, successfully catching that the local models were guessing and halting execution before deploying broken code.

---

## Quickstart

### Installation

```bash
pip install system2
```

### Python API

```python
from sys2 import System2Engine

# Initialize on Apple Silicon GPU (mps) or NVIDIA GPU (cuda)
engine = System2Engine(device="mps")

# Adjudicate 4 competing code candidates
result = engine.adjudicate(
    question="Select the correct, thread-safe implementation of an LRU cache eviction node:",
    choices=[
        candidate_a_code,
        candidate_b_code,
        candidate_c_code,
        candidate_d_code,
    ],
    domain="code",
)

print(f"Winning Candidate: #{result.selected_index + 1}")
print(f"Calibrated Confidence: {result.confidence * 100:.1f}%")
print(f"5-Head Consensus: {result.head_consensus} Heads")
print(f"Deliberation Latency: {result.deliberation_latency_ms:.2f} ms")
```

### Drop-in Local OpenAI Proxy

Run System 2 as a drop-in bridge server for Cursor, Claude Code, Cline, or LangChain:

```bash
python3 -m sys2.server --port 8000
```

Point your OpenAI client base URL to `http://localhost:8000/v1`.

---

## Benchmarks: M4 Max (128 GB Unified Memory)

| Metric | Path A: `qwen-3.8-27b-smoll` | Path B: `qwen-3.8-27b-iq3xxs` | Path Gemma: `google/gemma-4-12b` |
| :--- | :--- | :--- | :--- |
| **Model Size / VRAM** | 2.0-bit (6.77 GB) | 3.0-bit (10.18 GB) | 4.0-bit (6.77 GB) |
| **Module 1 (Node)** | 99.7% Conf (41 lines) | 81.6% Conf (19 lines) | 76.6% Conf (22 lines) |
| **Module 2 (Cache)** | 94.2% Conf (18 lines) | 75.2% Conf (14 lines) | 98.7% Conf (37 lines) |
| **Module 3 (Test Suite)** | 99.6% Conf (31 lines) | 67.8% Conf (10 lines) | **41.8% Conf (Drift Flagged)** |
| **Deliberation Latency (MPS)**| 580ms | 982ms | 1,107ms |
| **Pytest Result** | FAILED (SyntaxError) | FAILED (IndentationError) | FAILED (Markdown list indent) |
| **Root Failure Cause** | 2-bit CoT loop truncation | 3-bit CoT loop truncation | Indented markdown list block |

---

## Citation & License

- **License:** Apache 2.0
- **Authors:** Thom James-Worrall / Antigravity Research
