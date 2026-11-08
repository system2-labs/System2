# Deconstructing the System2 Evaluation Metrics & Benchmark Telemetry

This document provides a comprehensive technical reference for the evaluation metrics, calibration formulations, speed benchmarks, and cognitive diversity indicators used in the **Unified System2 Evaluation & Adversarial Red-Team Suite**.

---

## 1. Experimental Setup & Hardware Configuration

All metrics and telemetry reported in the suite were captured under the following controlled environment:

- **Hardware Target**: Apple Silicon Mac (M-series unified memory GPU via PyTorch `mps` backend with automatic fallback to `cpu`).
- **Base Encoder Backbone**: `answerdotai/ModernBERT-Large-Instruct` (395M parameters, strictly frozen weights).
- **Subspace Adaptation**: Top-8 layer LoRA adapters (layers 20–27, rank $r=32$, scaling $\alpha=32$, ~1.57M trainable parameters per domain).
- **Deliberation Architecture**: Non-autoregressive feedforward pass with 5 dense parallel ThoughtHeads ($2\times$ transformer self-attention layers each), pairwise candidate contrast via Arithmetic Centroid ($d_i = x_i - \frac{1}{4}\sum_{j=1}^4 x_j$), per-head logit $z$-score standardization, and a Cross-Attending Judger.
- **Evaluation Scope**: 13 domain-specialized tracks (`science`, `code`, `med`, `electrical`, `auto`, `culinary`, `math`, `legal`, `finance`, `physics`, `safety`, `eval`, `logic`).
- **Sample Allocation**: 100 stratified held-out test split items per track (totaling 2,600 full forward deliberations across Oracle and E2E modes).
- **Deterministic Seeding**: Global seed `42` (`torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)`). Sequence ceiling capped dynamically at 1024 tokens.
- **GPU Timing Synchronization**: Explicit `torch.mps.synchronize()` barriers around forward passes to isolate computation time from asynchronous queue dispatch.

---

## 2. Core Evaluation Metrics Breakdown

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SYSTEM2 EVALUATION METRICS                                     │
├───────────────────────────────┬──────────────────────────────────┬───────────────────────────────┤
│    Task Accuracy & Fidelity   │       Calibration & Risk         │  Speed & Cognitive Diversity  │
├───────────────────────────────┼──────────────────────────────────┼───────────────────────────────┤
│  • Oracle Acc                 │  • Oracle Brier                  │  • Delib Latency p95          │
│  • E2E Acc                    │  • E2E Brier                     │  • Disagreement               │
│                               │  • Oracle Guarded                │                               │
│                               │  • E2E Guarded                   │                               │
└───────────────────────────────┴──────────────────────────────────┴───────────────────────────────┘
```

### 2.1 Oracle Accuracy vs. End-to-End (E2E) Accuracy

#### `Oracle Acc` (Upper Bound In-Domain Fidelity)
- **Definition**: The classification accuracy achieved when an evaluation item is manually dispatched directly to its ground-truth domain adapter:
  $$\text{Oracle Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left(\operatorname{argmax}_c \left(\text{System2}_{\text{oracle}}(q_i, \mathbf{c}_i)\right) == y_i\right)$$
- **Purpose**: Isolates the reasoning fidelity of each domain's specialized weights from any upstream prompt routing ambiguity.
- **Observed Macro Average**: **87.2%** (ranging from 62.0% on complex causal chains to 100.0% on formal logic, eval meta-judging, and automotive fault trees).

#### `E2E Acc` (Autonomous Pipeline Fidelity)
- **Definition**: The classification accuracy achieved when an unannotated prompt is ingested "cold" by the autonomous pipeline. The two-stage `DomainRouter` first scans lexical signatures (<0.05 ms) and dense centroid projections (<0.8 ms), hot-swaps the selected adapter in memory (<3 ms), and deliberates:
  $$\text{Track}^* = \operatorname{DomainRouter}(q_i) \implies \text{Adjudicate}\left(q_i, \mathbf{c}_i, \text{domain}=\text{Track}^*\right)$$
- **Purpose**: Reflects real-world deployment performance. The delta $(\text{Oracle Acc} - \text{E2E Acc})$ measures **router misclassification cost**.
- **Observed Macro Average**: **85.9%** (retaining 98.5% of Oracle fidelity, with minor router leakage concentrated in subtle statutory legal queries).

---

### 2.2 Oracle Brier Score vs. E2E Brier Score

$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \left(\text{Confidence}_i - \mathbb{I}(\text{Prediction}_i == y_i)\right)^2$$

- **Definition**: The mean squared error between the model's calibrated decision confidence $\hat{p} \in [0, 1]$ and the binary correctness indicator $y \in \{0, 1\}$.
- **Interpretation**:
  - **`0.000`**: Mathematically perfect probability calibration (100% confident when correct; 0% confident when incorrect).
  - **`1.000`**: Pathological overconfidence (100% confident on false predictions).
  - **`0.1875`**: Baseline score achieved by an uncalibrated uniform guesser ($0.25$ confidence across 4 choices).
- **Why It Matters**: Standard accuracy ignores probability honesty; an autonomous agent that asserts high certainty when wrong will cause catastrophic downstream actions. Brier heavily punishes overconfident errors.
- **Observed Macro Average**: **`0.0992`** (Oracle) and **`0.1038`** (E2E), with specialized tracks like Logic (`0.0298`) and Eval (`0.0306`) demonstrating near-perfect confidence calibration.

---

### 2.3 True 10-Bin Expected Calibration Error (ECE)

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \operatorname{acc}(B_m) - \operatorname{conf}(B_m) \right|$$

- **Definition**: Partitions the confidence interval $[0, 1]$ into $M=10$ equidistant bins ($B_1 = [0.0, 0.1], \dots, B_{10} = (0.9, 1.0]$). For every bin containing predictions, it calculates the absolute divergence between the empirical empirical accuracy $\operatorname{acc}(B_m)$ and average predicted confidence $\operatorname{conf}(B_m)$, weighted by bin occupancy $\frac{|B_m|}{N}$.
- **Decoupled Verification**: Implemented in pure PyTorch in `evals/metrics.py` and validated for numerical parity against `eval.py::compute_ece`.
- **Observed Macro Average**: **0.1651** across 10-bin partitions.

---

### 2.4 The Guarded Metric (`Oracle Guarded` & `E2E Guarded`)

$$\text{Guarded Metric} = 0.5 \times \text{Accuracy} + 0.5 \times (1.0 - \text{Brier})$$

- **The Metric Invariant**: Mandated by `GEMINI.md`, the Guarded Metric is the foundational optimization target of System2. It harmonizes raw reasoning capacity with epistemic humility.
- **Mechanics**: If an experimental trial raises accuracy from 85% to 88% (+3%) but becomes reckless, causing Brier error to surge from 0.08 to 0.16 (+0.08), the guarded metric drops:
  $$\Delta \text{Guarded} = 0.5(0.03) + 0.5(-0.08) = +0.015 - 0.040 = -0.025$$
  Under `runner.sh`, this triggers an automatic `git reset --hard HEAD` rollback.
- **Observed Macro Average**: **`0.8862`** (Oracle) and **`0.8777`** (E2E).

---

### 2.5 Tail Deliberation Latency (`Delib Latency p95`)

- **Definition**: The 95th percentile wall-clock time (in milliseconds) required to execute the complete feedforward deliberation pipeline:
  1. Asymmetric pairwise span mean pooling via frozen ModernBERT (4 candidates $\times$ prompt).
  2. 5 parallel ThoughtHeads ($2\times$ transformer blocks per head).
  3. Arithmetic Centroid Contrast calculation ($[x_i \,\|\, d_i]$).
  4. Per-head logit $z$-score normalization.
  5. Multi-head Cross-Attending Judger selection and calibration scaling.
- **Observation on Apple Silicon MPS**:
  - **Macro Median**: **14.2 ms**
  - **Macro p95**: **19.66 ms** (with individual domain p95s ranging from 11.23 ms on safety to 26.05 ms on legal).
- **Comparison**: Deliberates $>50\times$ faster than an autoregressive 7B LLM emitting even a single generation token.

---

### 2.6 Inter-Head Cognitive Disagreement

$$\text{Disagreement} = 1.0 - \frac{1}{\binom{5}{2}} \sum_{1 \le i < j \le 5} \mathbb{I}(\text{Head}_i == \text{Head}_j)$$

- **Definition**: The mean pairwise disagreement rate across all $\binom{5}{2} = 10$ unique pairs among the 5 ThoughtHeads:
  - **`0.000` (Unanimous Equilibrium)**: Every head independently votes for the exact same choice ($5/5$ consensus).
  - **`0.600` (Fractured Deliberation)**: A split vote (e.g., 3 heads vote option A, 2 heads vote option B; pairwise agreement is $4/10 = 0.400$, yielding $0.600$ disagreement).
- **Domain Behavioral Segregation**:
  - **Axiomatic / Rule-Dominant Domains**: Formal Logic (`0.000`), Eval (`0.000`), Auto (`0.006`), Safety (`0.026`). The 5 heads rapidly converge on invariant deductive features.
  - **Ambiguity / Multi-Step Causal Domains**: Science (`0.222`), Math (`0.210`), Code (`0.207`), Finance (`0.200`). The asymmetric span views (holistic vs. question-anchored vs. choice-forward) actively generate competing candidate perspectives, and the Cross-Attending Judger breaks ties.

---

## 3. Reference Scorecard Summary

| Track | Domain Description | Oracle Guarded | E2E Guarded | Oracle Acc | E2E Acc | Oracle Brier | E2E Brier | Delib p95 | Disagree |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `science` | Track A: Science & Causal Reasoning | `0.6972` | `0.6975` | 62.0% | 62.0% | 0.2255 | 0.2250 | 22.18ms | 0.222 |
| `code` | Track B: Polyglot Code Traces | `0.8076` | `0.8076` | 78.0% | 78.0% | 0.1649 | 0.1649 | 23.53ms | 0.207 |
| `med` | Track C: Clinical Diagnostics | `0.9060` | `0.9067` | 89.0% | 89.0% | 0.0781 | 0.0765 | 22.94ms | 0.099 |
| `electrical` | Track D: Electrical Systems | `0.9422` | `0.9310` | 94.0% | 93.0% | 0.0555 | 0.0680 | 13.53ms | 0.055 |
| `auto` | Track E: Automotive Diagnostics | `0.9825` | `0.9755` | 100.0% | 99.0% | 0.0349 | 0.0390 | 22.76ms | 0.006 |
| `culinary` | Track F: Culinary Food Science | `0.7510` | `0.7463` | 69.0% | 68.0% | 0.1880 | 0.1874 | 12.16ms | 0.141 |
| `math` | Track G: Quantitative Reasoning | `0.9142` | `0.9142` | 94.0% | 94.0% | 0.1117 | 0.1117 | 25.34ms | 0.210 |
| `legal` | Track H: Statutory Compliance | `0.8729` | `0.8013` | 87.0% | 76.0% | 0.1242 | 0.1574 | 26.05ms | 0.118 |
| `finance` | Track I: Financial Reasoning | `0.8570` | `0.8516` | 82.0% | 81.0% | 0.1060 | 0.1069 | 12.51ms | 0.200 |
| `physics` | Track J: Physics & Astrophysics | `0.8543` | `0.8488` | 80.0% | 80.0% | 0.0914 | 0.1024 | 11.91ms | 0.175 |
| `safety` | Track K: Command Gatekeeper | `0.9658` | `0.9598` | 98.0% | 97.0% | 0.0484 | 0.0504 | 11.23ms | 0.026 |
| `eval` | Track L: Meta-Judge Evaluation | `0.9847` | `0.9847` | 100.0% | 100.0% | 0.0306 | 0.0306 | 25.52ms | 0.000 |
| `logic` | Track M: Formal & Deductive Logic | `0.9851` | `0.9851` | 100.0% | 100.0% | 0.0298 | 0.0298 | 25.87ms | 0.000 |
| **MACRO** | **Across All 13 Domains** | **`0.8862`** | **`0.8777`** | **87.2%** | **85.9%** | **0.0992** | **0.1038** | **19.66ms** | **0.112** |
