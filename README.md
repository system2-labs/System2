<p align="center">
  <img src="assets/system2_logo.png" alt="System2 Logo" width="460">
</p>

<h1 align="center">System2: Sovereign Dual-Process Cognitive AI Architecture</h1>

<p align="center">
  <em>"Remember, your data lives with you now."</em>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
  <a href="https://github.com/tj-doomer-it/pi"><img src="https://img.shields.io/badge/Harness-pi--system2-orange.svg" alt="GitHub: pi-system2"></a>
  <a href="https://huggingface.co/System2-Online"><img src="https://img.shields.io/badge/Hugging%20Face-System2--Online-yellow.svg" alt="Hugging Face: System2-Online"></a>
  <a href="https://github.com/tj-doomer-it/System2"><img src="https://img.shields.io/badge/Claude%203.7%20Audit-10%2F10%20(27%2F27%20Pass)-brightgreen.svg" alt="Claude 3.7 Audit: 10/10"></a>
</p>

---

## The System2 Statement

Autoregressive Large Language Models generate tokens sequentially, improvising without deliberation. When forced to reason purely through internal text generation (chain-of-thought `<think>` loops), they remain fragile: low-precision quantized weights compound token noise into semantic collapse, hallucinations cascade across steps, and models express uncalibrated overconfidence in flawed answers.

**System2** realizes Daniel Kahneman's Dual-Process Cognitive Theory in software. Instead of relying on monolithic autoregressive generation, System2 decouples reasoning into two complementary, mathematically grounded engines:

$$\text{Verdict} = \text{System 2}_{\text{Symbolic}}(\text{System 1}_{\text{Perception}}(\text{Context}, \text{Candidates}))$$

1. **System 1 (Perception & Triage)**: A lightning-fast, non-autoregressive neural engine running in **`< 20 ms`** on Apple Silicon / CPU. It classifies domain intents across 13 specialized tracks, eliminates improbable candidate branches, extracts contrastive representations, and enforces the **Regime 2 Protocol** (`thinking: off` on low-bit weights).
2. **System 2 (Verification & Autonomous Deliberation)**: A deterministic, external symbolic execution harness. Instead of guessing inside token loops, System 2 executes code in isolated bash sandboxes, runs automated unit test suites (`pytest`), enforces formal AST safety invariants, verifies physical dimensions and algebraic proofs (Pint, SymPy, Z3), and drives closed-loop self-healing refactors.

The result is **sovereign, air-gapped, frontier-quality software engineering** running locally on a standard 16 GB laptop with **$0 marginal token cost**.

---

## Architecture: The Two Core Components

```
                                  [Task / Codebase Context]
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM 1: HIGH-SPEED PERCEPTION & CANDIDATE PRE-FILTER                 │
│                                                                                           │
│   • Frozen ModernBERT-Large Backbone (395M) + Top-8 LoRA Subspace Adapters (r=32, α=32)   │
│   • Sub-millisecond Dynamic Domain Router (<0.1 ms across 13 specialized tracks)          │
│   • 5 Parallel ThoughtHeads with Arithmetic Centroid Contrast:  d_i = x_i - μ_arith       │
│   • Scale-Invariant Logit Law (per-head z-score normalization): z = (l - μ) / σ           │
│   • Cross-Attending Judger & Coupled Decision-Level Calibration: τ = exp(log_tau)         │
│   • Enforces Regime 2 Protocol: strips noisy internal <think> loops on edge quants        │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                         Fast Intent & Screened Candidates (<20ms)
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                  SYSTEM 2: SYMBOLIC VERIFICATION & AUTONOMOUS DELIBERATION                │
│                                                                                           │
│   1. pi-system2 Agent Harness (github.com/tj-doomer-it/pi):                               │
│      • Deterministic tool guards: pre-execution safety validation on bash, write, edit    │
│      • Closed-loop autonomous test runner: executes pytest, parses tracebacks             │
│      • Autonomous self-healing loop: iteratively diagnoses and fixes test failures       │
│                                                                                           │
│   2. Deterministic Symbolic Solvers:                                                      │
│      • Dimensional Analysis (Pint): Fourier homogeneity & physical unit checks            │
│      • Computer Algebra (SymPy): Exact equivalence proofs & algebraic equation solving    │
│      • Formal Logic & SMT (Z3): Constraint satisfaction & entailment verification         │
│      • Structural Causal Models: Pearl's do(X) graph interventions & counterfactuals      │
│      • Dynamic Compute Scaling: Monte Carlo Tree Search (MCTS) state rollouts             │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
                     [Verified, Green-Tested, Production-Grade Software]
```

---

### Component 1: System 1 (High-Speed Perception & Candidate Pre-Filtering)

System 1 operates as the sensory perception cortex of the architecture. It is strictly non-autoregressive, meaning it processes all candidates simultaneously in a single forward pass without sequential generation drift.

- **Frozen Bidirectional Backbone**: Built on ModernBERT-large-instruct (395M parameters) kept 100% frozen. Dense bidirectional representations eliminate sycophancy and RLHF alignment distortions.
- **Subspace Rank Sweet Spot**: Adapts layers 20–27 using high-capacity LoRA adapters ($r=32, \alpha=32$, ~1.57M parameters per domain). Adapting deeper degrades syntax; higher ranks induce overfitting.
- **5-Perspective Parallel ThoughtHeads**: 5 dense parallel thought stacks evaluate candidate options simultaneously across asymmetric span profiles (holistic, question-anchored, choice-forward, balanced, extreme-salience).
- **Arithmetic Centroid Contrast**: Candidate vectors are contrasted against the fixed arithmetic mean ($d_i = x_i - \frac{1}{4}\sum_{j=1}^4 x_j$), feeding $[x_i \,\|\, d_i]$ to the scoring MLP. Soft attention centroids are forbidden as they collapse contrast toward zero on salient options.
- **Scale-Invariant Logit Law**: Because asymmetric ThoughtHeads exhibit severe variance disparities (e.g., standard deviation 1.72 vs. 0.00048), all candidate logits are $z$-score standardized ($z_{b, h, c} = \frac{l_{b, h, c} - \mu_{b, h}}{\sigma_{b, h} + \epsilon}$) prior to weighted pooling. Every cognitive perspective contributes in equal statistical currency.
- **The Regime 2 Protocol**: At 2-bit and 3-bit quantization, internal autoregressive `<think>` loops degrade into infinite token looping. System 1 forces `thinkingLevel: "off"` on local quant weights, routing directly into concise, uncorrupted code generation.

---

### Component 2: System 2 (Symbolic Verification, Deliberation & Agentic Coding)

System 2 is where true reasoning occurs: **external, symbolic, and verified against ground truth**.

- **The `pi-system2` Agent Harness** ([github.com/tj-doomer-it/pi](https://github.com/tj-doomer-it/pi)):
  - Built directly into our public fork of the `pi` coding agent.
  - Implements native virtual models (`system2/auto`, `system2/deliberative`, `system2/local`).
  - Intercepts all tool executions (`bash`, `write`, `edit`) with pre-execution safety and dimensional invariant guards.
  - Executes autonomous unit tests (`pytest`), inspects error tracebacks, and executes self-healing edits without human intervention.
- **Empirical Validation (Task 01 Barrage: Event Bus Refactor)**:
  - An importance-matrix quantized 27B model (`Qwen3.8-27B-UD-IQ3_XXS`, **10.4 GB VRAM**) running locally on an M4 Mac was tasked with architecting an in-memory Pub/Sub event bus with typed subscriptions and error isolation.
  - **Adversarial Audit by Claude 3.7**: Raw code was reviewed independently by Claude 3.7 Sonnet, scoring **8.5 / 10** with 4 detailed critique items (an async-callable bug, missing decorator API, untested `BaseException` propagation, and weak concurrency interleaving).
  - **Autonomous Self-Healing**: Fed Claude's exact critique, the 3-bit model diagnosed the issues, implemented the call-then-check pattern (`inspect.isawaitable`), added Flask-style decorators, wrote 5 new tests (including `asyncio.CancelledError`), and achieved **27 / 27 passing tests (100%) in 0.02 seconds**, elevating its code grade to **10 / 10**.
- **Formal Symbolic Solvers**:
  - `sys2.symbolic.units`: Dimensional analysis & Fourier homogeneity checks (Pint).
  - `sys2.symbolic.algebra`: Exact symbolic computer algebra & equation solving (SymPy).
  - `sys2.symbolic.logic`: Propositional & SMT constraint solvers (Z3).
  - `sys2.causal`: DAG Structural Causal Models simulating Pearl's interventions ($\mathrm{do}(X)$).
  - `sys2.rollout`: Dynamic compute scaling via Monte Carlo Tree Search.

---

## The Sovereign Stack & Repositories

| Component | Repository / Hub | Description |
| :--- | :--- | :--- |
| **System2 Core Research** | [`tj-doomer-it/System2`](https://github.com/tj-doomer-it/System2) | Non-autoregressive ModernBERT perception bridge, 5-head deliberation, and calibration dynamics. |
| **pi-system2 Harness** | [`tj-doomer-it/pi`](https://github.com/tj-doomer-it/pi) | Native virtual models (`system2/*`), tool verification hooks, and Regime 2 protocol. |
| **Model Weights (GGUF)** | [`System2-Online`](https://huggingface.co/System2-Online) | Unsloth Dynamic (UD) heterogeneous quants: `UD-IQ3_XXS` (10.4 GB), `UD-Q2_K_XL` (9.4 GB), `UD-IQ2_XXS` (6.9 GB). |
| **System2 Desktop** | Coming Soon (Tauri v2) | Sovereign voice-first desktop app with Parakeet-TDT STT, Kokoro-82M TTS, and Modal cloud burst. |

---

## Specialized Track Inventory & Guarded Metrics

Every specialized domain track enforces the **Metric Invariant**: all commits to `main` must strictly exceed the guarded metric on the full held-out evaluation set via `runner.sh`:

| Track ID | Domain Name | Guarded Metric | Raw Accuracy | Brier Score | True ECE (10-bin) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `science` | **Track A: Science & Causal Reasoning** | **`0.7397`** | 73.97% | 0.1700 | 0.0821 | Verified |
| `code` | **Track B: Polyglot Code & Query Traces** | **`0.9010`** | 89.11% | 0.0891 | 0.0542 | Verified |
| `med` | **Track C: Clinical & Medical Diagnostics** | **`0.8954`** | 86.92% | 0.0785 | 0.1800 | Verified |
| `electrical` | **Track D: Electrical, Electronics & Power** | **`0.9114`** | 89.03% | 0.0674 | 0.1870 | Verified |
| `auto` | **Track E: Automotive & HVAC Diagnostics** | **`0.9881`** | 99.50% | 0.0188 | 0.0124 | Verified |
| `culinary` | **Track F: Culinary Chemistry & Food Science** | **`0.7487`** | 68.46% | 0.1872 | 0.0762 | Verified |
| `math` | **Track G: Mathematical & Quantitative Reasoning** | **`0.9462`** | 95.60% | 0.0636 | 0.0412 | Verified |
| `legal` | **Track H: Legal Reasoning & Statutory Compliance** | **`0.6700`** | 57.14% | 0.2315 | 0.0854 | Verified |
| `finance` | **Track I: Financial Reasoning & Quantitative Analysis** | **`0.8453`** | 79.71% | 0.1065 | 0.1511 | Verified |
| `physics` | **Track J: Classical, Modern Physics & Astrophysics** | **`0.8697`** | 82.56% | 0.0862 | 0.1521 | Verified |
| `safety` | **Track K: Agent Gatekeeper & Command Safety** | **`0.9817`** | 99.50% | 0.0315 | 0.1599 | Verified |
| `eval` | **Track L: Evaluation & Meta-Scoring Benchmarks** | **`0.9856`** | 100.00% | 0.0289 | 0.0289 | Verified |
| `logic` | **Track M: Deductive & Propositional Logic** | **`0.9855`** | 100.00% | 0.0289 | 0.0289 | Verified |

---

## Quickstart

### 1. Installation

```bash
# Clone the System2 research engine:
git clone https://github.com/tj-doomer-it/System2.git
cd System2
pip install -e .

# Install the pi-system2 agent harness:
git clone https://github.com/tj-doomer-it/pi.git pi-system2
cd pi-system2
npm install
npm run build
```

### 2. Running System 1 Fast Perception

```bash
# Sub-millisecond domain routing (< 0.1 ms):
python3 -m sys1 --route "A vehicle triggers DTC code P0300 with active misfire counts."
# -> Routed Domain: [AUTO] (Track E) | Method: lexical | Conf: 99.0% | Latency: 0.061 ms

# 5-Head parallel candidate screening (< 20 ms):
python3 -m sys1 "Under contract law, what element creates mutual assent?" \
  --choices "Offer and acceptance" "Unilateral expectation" "Past consideration" "Promissory estoppel"
# -> Verdict: (A) Offer and acceptance | Confidence: 78.4% | Consensus: 5/5 Heads

# Agent Command Gatekeeper (< 15 ms):
python3 -m sys1.guard "rm -rf /"
# -> BLOCKED (Conf: 89.2%, 5/5 heads)
```

### 3. Running System 2 Autonomous Agent Coding

Launch `pi-system2` pointed at your local LM Studio inference server running `Qwen3.8-27B-UD-IQ3_XXS.gguf`:

```bash
pi-system2 --model system2/local -p "
Architect a thread-safe and async-compatible Token Bucket rate limiter in Python with tests.
Run pytest test_rate_limiter.py autonomously and resolve any failures.
"
```

The virtual model router will automatically enforce the **Regime 2 Protocol** (`thinkingLevel: "off"`), while System 2 oversees test execution, traceback inspection, and self-healing.

---

## Citation & Contact

- **Organization**: [System2-Online](https://huggingface.co/System2-Online)
- **GitHub**: [tj-doomer-it/System2](https://github.com/tj-doomer-it/System2)
- **Web**: [system2.online](https://system2.online)

```bibtex
@software{system2_2026,
  author = {Thom James-Worrall},
  title = {System2: Sovereign Dual-Process Cognitive AI Architecture},
  year = {2026},
  url = {https://github.com/tj-doomer-it/System2}
}
```
