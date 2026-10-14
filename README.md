# System2: Parallel Hypothesis Reasoning Architecture

A from-scratch, non-autoregressive "System 2" deliberation architecture designed to run and train locally on Apple Silicon.

System2 generates parallel hypotheses, estimates calibrated confidence for each across asymmetric span formulations, attends across competing candidates with arithmetic centroid contrast, normalizes logit variances in equal standard-deviation currency, and prunes flawed reasoning branches in $<30\text{ ms}$.

---

## Architecture Overview

```
                      [Input Prompt / Context]
                                 │
                                 ▼
           [ModernBERT-large-instruct Encoder (395M, Frozen)]
             + LoRA Subspace Adapters (Layers 20–27, r=32, α=32)
                                 │ (1024-dim dense representation)
        ┌────────────────────────┼────────────────────────┐
        ▼                        ▼                        ▼
 [ThoughtHead 1]          [ThoughtHead 2]          [ThoughtHead 5]
 (Asymmetric Span v0)     (Asymmetric Span v1)     (Asymmetric Span v4)
  2x Transformer Layers    2x Transformer Layers    2x Transformer Layers
  Arithmetic Centroid:     Arithmetic Centroid:     Arithmetic Centroid:
    d_i = x_i - μ_arith      d_i = x_i - μ_arith      d_i = x_i - μ_arith
  Logit z-score norm:      Logit z-score norm:      Logit z-score norm:
    z_1 = (l - μ)/σ          z_2 = (l - μ)/σ          z_5 = (l - μ)/σ
        │                        │                        │
  (Emb 1, Conf 1)          (Emb 2, Conf 2)          (Emb 5, Conf 5)
        └────────────────────────┼────────────────────────┘
                                 ▼
                  [Cross-Attending Judger & Calibrator]
              - Multi-head cross-attention over 5 candidates
              - Coupled decision-level calibration scale τ = exp(log_tau)
              - Branch pruning & counterfactual elimination
                                 │
                                 ▼
                 [Winning Hypothesis & Calibrated Verdict]
```

### Core Design Principles
1. **Frozen Bidirectional Encoder**: ModernBERT-large / ModernBERT-Large-Instruct (395M) weights remain 100% frozen. Dense bidirectional embeddings eliminate autoregressive drift, sycophancy, and post-hoc RLHF alignment artifacts.
2. **LoRA Subspace Rank Sweet Spot**: High-capacity low-rank adapters ($r=32, \alpha=32$, ~1.57M params) adapt the top 8 layers (20–27) per domain without warping lower-layer lexical syntax.
3. **5-Head Deliberation Equilibrium**: 5 full dense parallel ThoughtHeads evaluate candidate choices simultaneously across asymmetric span perspectives (holistic, question-anchored, choice-forward, balanced, extreme-salience).
4. **Arithmetic Centroid Contrast Invariance**: Candidate vectors are contrasted against the fixed arithmetic mean: $d_i = x_i - \frac{1}{4}\sum_{j=1}^4 x_j$, feeding $[x_i \,\|\, d_i]$ to the scoring MLP. Dynamic attention soft centroids are prohibited as they collapse contrast toward zero on salient options.
5. **Scale-Invariant Logit Law**: All candidate choice logits are $z$-score standardized per head ($z_{b, h, c} = \frac{l_{b, h, c} - \mu_{b, h}}{\sigma_{b, h} + \epsilon}$) prior to weighted pooling, ensuring every cognitive perspective contributes in equal statistical currency.
6. **Strictly Non-Autoregressive Feedforward**: Single-pass forward deliberation in $<30\text{ ms}$ on Apple Silicon GPU (`mps`) or $<250\text{ ms}$ on CPU.

---

## Early Stage vs. Architectural Evolution

The table below contrasts the initial canary prototype with the mature System2 multi-track architecture:

| Dimension | Early Stage Prototype (Phase 1) | Architectural Evolution (Current System2) |
| :--- | :--- | :--- |
| **Evaluation Scope** | AG News 4-class topic classification canary | Multi-step reasoning across Science, Code, Medical, Electrical, Auto, and Culinary |
| **Encoder Strategy** | Completely frozen ModernBERT without adapters | Frozen ModernBERT + top-8 layer LoRA subspace adapters ($r=32, \alpha=32$) |
| **ThoughtHead Structure** | Homogeneous generic pooling | 5 Asymmetric Span Formulations (Holistic, Question-anchored, Choice-forward, Balanced, Extreme-salience) |
| **Candidate Contrast** | Uncontrasted raw candidate vectors | **Arithmetic Centroid Contrast**: $d_i = x_i - \frac{1}{4}\sum_{j=1}^4 x_j \implies [x_i \,\|\, d_i]$ |
| **Inter-Head Arbitration** | Raw logit summation (drowned by high-variance heads) | **Scale-Invariant Logit Law**: per-head $z$-score normalization ($z = \frac{l - \mu}{\sigma + \epsilon}$) |
| **Calibration Method** | Uncalibrated BCE loss | Coupled decision-level calibration gradient with learnable temperature $\tau = \exp(\text{log\_tau})$ |
| **Domain Architecture** | Monolithic multi-class classifier | **In-Domain Causal Purity**: 6 domain-specialized LoRA + ThoughtHead checkpoints |
| **Prompt Dispatch** | Manual dataset flag required | **Two-Stage Dynamic Domain Router** (Lexical Gate + Dense Centroid Projection in $<0.1\text{ ms}$) |
| **Inference Integration** | Standalone classification script | **Path B Hybrid Deliberation**: Adjudicates 4 independent zero-knowledge LLM samples |

---

## Specialized Track Inventory & Guarded Metrics

Every specialized track enforces the **Metric Invariant**: all commits must strictly exceed the guarded metric on the full held-out evaluation set via `runner.sh`.

| Track ID | Domain Name | Guarded Metric | Held-Out Accuracy | Brier Score | Checkpoint | Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `science` | **Track A: Science & Causal Reasoning** | **`0.7397`** | 73.97% | 0.1700 | [`models/system2.pt`](file:///Users/thomjames-worrall/System2/models/system2.pt) | Verified |
| `code` | **Track B: Polyglot Code & Query Traces** | **`0.9010`** | 89.11% | 0.0891 | [`models/system2_code.pt`](file:///Users/thomjames-worrall/System2/models/system2_code.pt) | Verified |
| `med` | **Track C: Clinical & Medical Diagnostics** | **`0.5397`** | 29.38% | 0.2145 | [`models/system2_med.pt`](file:///Users/thomjames-worrall/System2/models/system2_med.pt) | Verified |
| `electrical` | **Track D: Electrical, Electronics & Power** | **`0.5496`** | 32.09% | 0.2217 | [`models/system2_electrical.pt`](file:///Users/thomjames-worrall/System2/models/system2_electrical.pt) | Verified |
| `auto` | **Track E: Automotive & HVAC Diagnostics** | **`0.9881`** | 99.50% | 0.0188 | [`models/system2_auto.pt`](file:///Users/thomjames-worrall/System2/models/system2_auto.pt) | Verified |
| `culinary` | **Track F: Culinary Chemistry & Food Science** | **`0.7487`** | 68.46% | 0.1872 | [`models/system2_culinary.pt`](file:///Users/thomjames-worrall/System2/models/system2_culinary.pt) | Verified |
| `math` | **Track G: Mathematical & Quantitative Reasoning** | **`0.9462`** | 95.60% | 0.0636 | [`models/system2_math.pt`](file:///Users/thomjames-worrall/System2/models/system2_math.pt) | Verified |

---

## Staged Roadmap & Milestone Tracking

- [x] **Phase 1: Pipeline Canary**:
  - End-to-end encoder $\to$ parallel heads $\to$ cross-attending judger on AG News (Metric: `0.8998`, Accuracy: `88.55%`).
- [x] **Phase 2: Judger & Diversity Optimization**:
  - Align Judger training target to select confident *and correct* heads.
  - Implement arithmetic centroid contrast $[x_i \,\|\, d_i]$ and pairwise diversity penalties.
  - Coupled decision-level calibration gradient with learnable $\tau$.
  - Scale-invariant $z$-score logit pooling across asymmetric thought heads.
- [x] **Phase 2.5: Path B Hybrid LLM Deliberation Benchmark**:
  - Pair local generative models with System 2 feedforward deliberation.
  - Validated across 4 tiers: Gemma-e4b (4B) $\to$ Gemma-12B (12B) $\to$ Qwen-27B $\to$ Qwen-35B.
  - Proved that an ultra-lightweight 4B/12B model + System 2 achieves **100% accuracy parity** and higher calibrated confidence than a standalone 35B model.
- [x] **Phase 3: Multi-Step Reasoning Benchmarks**:
  - [x] **AI2 ARC (Science)**: Full integration of ARC-Challenge + ARC-Easy in-domain data (`0.7397`).
  - [x] **Polyglot Code Execution (Code)**: Multi-language traces across Python, Rust, C++, C#, Java, SQL, Bash (`0.9010`).
  - [x] **GSM8K & MATH (Symbolic Reasoning)**: Multi-step mathematical word problems with type-aware distractors (**`0.9462`** guarded metric, **`95.60%`** held-out accuracy).
- [x] **Phase 4: Dynamic Domain Routing & Dispatch**:
  - Two-stage hierarchical router (Lexical Gate + Dense ModernBERT Centroid Projection).
  - Sub-millisecond dispatch ($<0.1\text{ ms}$) across specialized tracks with **100.0% accuracy** on 60-question test suite.

---

## Quickstart

### 1. Requirements
- Python 3.10+
- PyTorch with Apple Silicon MPS support (`torch.backends.mps.is_available() == True`)
- Hugging Face `transformers` and `datasets`

### 2. Interactive Prompt Routing
Test the sub-millisecond dynamic domain router from the CLI:
```bash
python3 scripts/route_prompt.py "Can authentic sourdough bread be baked using industrial glue and Portland cement?"
# -> Dispatched Track: [CULINARY] (Confidence: 96.0%, Latency: 0.062 ms)

python3 scripts/route_prompt.py "A vehicle triggers DTC code P0300 with active misfire counts across cylinders 1, 3, and 5."
# -> Dispatched Track: [AUTO] (Confidence: 99.0%, Latency: 0.062 ms)
```

### 3. Deliberation Benchmark
Run the 4-tier model scaling deliberation benchmark comparing independent zero-knowledge LLM samples adjudicated by System 2 on CPU:
```bash
python3 scripts/benchmark_four_model_scaling.py
```

### 4. AutoResearch Loop
Run the autonomous metric-guarded training loop on any track worktree:
```bash
cd scratch/worktrees/track-e
./runner_auto.sh "experiment: sharpen calibration smoothing to 0.12"
```
The runner trains on Apple Silicon MPS, judges on the full held-out test split, and commits if improved or executes `git reset --hard HEAD` if regressed.
