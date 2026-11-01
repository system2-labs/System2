# System2: Non-Autoregressive Multi-Head Deliberation Architecture
## Scientific Specification & Training Dynamics

**Status:** Living Architectural & Mathematical Specification  
**Version:** 2.0 (Post Phase 5)  
**Target Hardware:** Apple Silicon (`mps`), CUDA (`cuda`), CPU (`cpu`)  
**Base Model:** `ModernBERT-large` / `ModernBERT-Large-Instruct` (395M parameters)

---

## 1. Architectural Overview & Parameter Budgets

System2 replaces sequential autoregressive generation with a feedforward, multi-head parallel deliberation engine. Decisions across competing hypotheses are computed in a single forward pass without token-by-token generation loops or autoregressive drift.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       INPUT (Question + Choices A..D)                       │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ModernBERT-large Backbone (395M)                         │
│                    Layers 0–19: 100% FROZEN (Weights)                       │
│                    Layers 20–27: LoRA Subspace Adapters                     │
│                    r = 32, α = 32 (~1.57M parameters) [TRAINED]             │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │ 1024-dim Contextual Embeddings
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Asymmetric Span Partitioning (5 Modes)                    │
│      Holistic | Question-Anchored | Choice-Forward | Balanced | Counterfactual│
└──────────┬─────────────────┬─────────────────┬──────────────┬───────────────┘
           │                 │                 │              │               │
           ▼                 ▼                 ▼              ▼               ▼
     ┌───────────┐     ┌───────────┐     ┌───────────┐  ┌───────────┐   ┌───────────┐
     │ThoughtHead│     │ThoughtHead│     │ThoughtHead│  │ThoughtHead│   │ThoughtHead│
     │     0     │     │     1     │     │     2     │  │     3     │   │     4     │
     └─────┬─────┘     └─────┬─────┘     └─────┬─────┘  └─────┬─────┘   └─────┬─────┘
           │ (e₀, l₀, c₀)    │ (e₁, l₁, c₁)    │              │               │
           └─────────────────┼─────────────────┼──────────────┼───────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              Judger Module                                  │
│             Multi-Head Cross-Attention over Candidates (MHA)                │
│                 Scale-Invariant z-score Logit Arbitration                   │
│             Coupled Decision-Level Calibration Gradient (τ)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   Winning Hypothesis & Calibrated Verdict                   │
│                   Confidence c ∈ [0, 1] | Pruned if c < 0.30                │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Allocation & Trainability Matrix

| Component | Base Dimension | Layer Range | Parameter Count | Trainable? | Purpose |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Encoder Lexical Layers** | 1024 | Layers 0–19 | ~280M | **No (Frozen)** | Preserves core bidirectional syntax & semantic grounding without catastrophic forgetting. |
| **Encoder LoRA Subspace** | 1024 | Layers 20–27 | ~1.57M ($r=32$) | **Yes** | Domain adaptation across top representation layers. |
| **ThoughtHead 0..4 Encoders** | 512 | 2 layers / head | ~33.6M total | **Yes** | 5 parallel hypothesis generation branches with independent transformer self-attention. |
| **ThoughtHead Contrast MLPs**| 1024 $\to$ 256 $\to$ 1 | N/A | ~1.3M total | **Yes** | Arithmetic centroid contrast scoring $[x_i \,\|\, d_i]$. |
| **ThoughtHead Calibrators** | 512 $\to$ 64 $\to$ 1 | N/A | ~165K total | **Yes** | Scalar confidence estimation per hypothesis branch. |
| **Judger Cross-Attention** | 512 (8 heads) | 1 layer | ~1.05M | **Yes** | Inter-hypothesis competition in latent feature space. |
| **Judger Decision Scorer** | 512 $\to$ 64 $\to$ 1 | N/A | ~33K | **Yes** | Final branch arbitration and calibration scale $\tau$. |
| **TOTAL** | — | — | **~432M** | **~37.7M (8.7%)** | Fully fits inside single-device Apple Silicon Unified Memory. |

---

## 2. Forward Pass Mechanics & Module Formulations

### 2.1 Asymmetric Cognitive Spans
To prevent the 5 parallel ThoughtHeads from collapsing into redundant feature extractors, each head receives tokens formatted under distinct cognitive spans:
- **Head 0 (Holistic Context)**: Standard bidirectional pooling across the full sequence $[Q \,\|\, C_i]$.
- **Head 1 (Choice-Forward Focus)**: Weighted sequence pooling prioritizing candidate choice spans over question tokens.
- **Head 2 (Question-Anchored Focus)**: Weighted sequence pooling prioritizing causal constraints within the premise/question.
- **Head 3 (Contrastive Contextual)**: Encodes candidate choices conditioned on adjacent alternative choices.
- **Head 4 (Extreme Salience / Counterfactual)**: Focuses attention mass on candidate boundary tokens.

### 2.2 ThoughtHead Architecture
Each ThoughtHead $h \in \{0, \dots, 4\}$ maps the 1024-dimensional pooled encoder representation $u \in \mathbb{R}^{B \times 4 \times 1024}$ into candidate logits $l_h$ and branch confidence $c_h$:

1. **Input Projection & Intra-Choice Attention**:
   $$x^{(0)} = W_{\text{proj}} u + b_{\text{proj}}, \quad x^{(0)} \in \mathbb{R}^{B \times 4 \times 512}$$
   $$x = \text{TransformerEncoder}_{h}(x^{(0)}), \quad \text{depth} = 2, \, d_{\text{model}} = 512, \, n_{\text{head}} = 8$$

2. **Arithmetic Centroid Contrast Invariance**:
   For candidate choice $i \in \{0, 1, 2, 3\}$, the contrast vector $d_i$ is computed against the fixed arithmetic mean across all 4 choices:
   $$\bar{x} = \frac{1}{4}\sum_{j=0}^3 x_j, \quad d_i = x_i - \bar{x}$$
   The candidate embedding is concatenated with its contrast vector:
   $$v_i = [x_i \,\|\, d_i] \in \mathbb{R}^{1024}$$
   $$\text{logit}_{h, i} = W_2 \cdot \text{GELU}(W_1 v_i + b_1) + b_2, \quad \text{logit}_{h} \in \mathbb{R}^{B \times 4}$$

   > **Invariant**: Dynamic attention soft centroids are prohibited because they collapse contrast toward zero on salient candidates. The arithmetic mean preserves an invariant linear reference frame.

3. **Intra-Head Confidence Calibration**:
   Pooled branch representation $e_h = \text{mean}(x) \in \mathbb{R}^{B \times 512}$ produces calibrated branch confidence:
   $$c_h = \sigma\left(W_{c2} \cdot \text{GELU}(W_{c1} e_h + b_{c1}) + b_{c2}\right) \in (0, 1)$$

### 2.3 The Judger Module (Cross-Candidate Attention)
The Judger enables candidate hypotheses to compete directly in latent embedding space:

1. **Multi-Head Self-Attention**:
   Let $E = [e_0, e_1, e_2, e_3, e_4] \in \mathbb{R}^{B \times 5 \times 512}$ represent the 5 head embeddings:
   $$\tilde{E} = E + \text{MHA}(Q=\text{LN}(E), K=\text{LN}(E), V=\text{LN}(E))$$

2. **Judger Arbitration Scores**:
   $$s = W_{s2} \cdot \text{GELU}(W_{s1} \tilde{E} + b_{s1}) + b_{s2}, \quad s \in \mathbb{R}^{B \times 5}$$
   $$\alpha = \text{softmax}(s, \text{dim}=-1)$$
   The winning head index is:
   $$h^* = \arg\max_{h} s_h$$

3. **Coupled Decision-Level Calibration**:
   The winning confidence is calibrated via a learnable log-temperature $\tau = \exp(\theta_{\tau})$:
   $$\text{conf}^* = \sigma\left(\frac{s_{h^*} - \bar{s}}{\tau}\right)$$
   If $\text{conf}^* < 0.30$, the deliberator abstains (selective classification).

---

## 3. Scale-Invariant Logit Law (Inference Arbitration)

Because individual ThoughtHeads operate across asymmetric span formulations, raw output logits develop severe inter-head variance disparities (e.g., standard deviations of $1.72$ vs. $0.00048$). Unnormalized summation allows high-variance heads to drown out low-variance heads.

All candidate choice logits are $z$-score standardized per head prior to weighted pooling:
$$z_{b, h, c} = \frac{l_{b, h, c} - \mu_{b, h}}{\sigma_{b, h} + \epsilon}$$
where:
$$\mu_{b, h} = \frac{1}{C}\sum_{c=1}^C l_{b, h, c}, \quad \sigma_{b, h} = \sqrt{\frac{1}{C}\sum_{c=1}^C (l_{b, h, c} - \mu_{b, h})^2}$$

The ensemble candidate score is computed as:
$$S_{b, c} = \sum_{h=0}^4 w_h \cdot z_{b, h, c}$$
where $w_h$ represents normalized head weights (or Judger softmax weights $\alpha_{b, h}$).

---

## 4. Multi-Objective Composite Loss Formulation

System2 models are trained with a **single combined backward pass** on a composite loss:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cal}}\mathcal{L}_{\text{cal}} + \lambda_{\text{judge}}\mathcal{L}_{\text{judge}} + \lambda_{\text{judge\_cal}}\mathcal{L}_{\text{judge\_cal}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{cand\_margin}}\mathcal{L}_{\text{cand\_margin}} + \lambda_{\text{balance}}\mathcal{L}_{\text{balance}}$$

### 4.1 Task Loss ($\mathcal{L}_{\text{task}}$) & Cognitive Autonomy Invariant
Each ThoughtHead independently computes cross-entropy against the ground-truth label $y$:
$$\mathcal{L}_{\text{task}} = \frac{1}{N}\sum_{h=0}^{N-1} \text{CrossEntropy}(l_h, y)$$

> **The Cognitive Autonomy Invariant ($\lambda_{\text{ens\_ce}} = 0$)**:  
> Ensemble-level cross-entropy is **never** backpropagated into the parallel ThoughtHeads. Backpropagating a joint loss diffuses gradient credit across heads, encouraging heads to free-ride and collapsing distinct cognitive perspectives toward consensus.

### 4.2 Head Confidence Calibration Loss ($\mathcal{L}_{\text{cal}}$)
Calibrates each head's predicted scalar confidence against whether that specific head was objectively correct:
$$\mathcal{L}_{\text{cal}} = \frac{1}{N}\sum_{h=0}^{N-1} \text{BCE}\left(c_h, \, \mathbb{I}[\arg\max(l_h) = y]\right)$$

### 4.3 Judger Selection Oracle Loss ($\mathcal{L}_{\text{judge}}$)
Supervises the Judger to select heads that are both correct and confident:
$$T = \begin{cases}
\arg\max_{h \in \mathcal{H}_{\text{correct}}} c_h & \text{if } \mathcal{H}_{\text{correct}} \neq \emptyset \\
\arg\max_{h} c_h & \text{otherwise}
\end{cases}$$
$$\mathcal{L}_{\text{judge}} = \text{CrossEntropy}(s, T, \text{label\_smoothing}=0.15)$$

### 4.4 Coupled Decision Calibration Loss ($\mathcal{L}_{\text{judge\_cal}}$)
Directly trains the winning decision confidence $\text{conf}^*$ against ground-truth success:
$$\mathcal{L}_{\text{judge\_cal}} = \text{BCE}\left(\text{conf}^*, \, \mathbb{I}[\arg\max(S) = y]\right)$$

### 4.5 Latent Diversity Loss ($\mathcal{L}_{\text{div}}$)
Penalizes pairwise cosine similarity between normalized head embeddings $e_i, e_j$:
$$\mathcal{L}_{\text{div}} = \frac{2}{N(N-1)}\sum_{0 \le i < j < N} \frac{e_i \cdot e_j}{\|e_i\|_2 \|e_j\|_2}$$

### 4.6 Contrastive Candidate Margin Loss ($\mathcal{L}_{\text{cand\_margin}}$)
Enforces a minimum margin $\gamma$ between the ground-truth candidate score and the highest distractor score:
$$\mathcal{L}_{\text{cand\_margin}} = \frac{1}{N}\sum_{h=0}^{N-1} \max\left(0, \, \gamma - \left(l_{h, y} - \max_{j \neq y} l_{h, j}\right)\right)$$

### 4.7 Judger Load-Balancing Loss ($\mathcal{L}_{\text{balance}}$)
Penalizes representation collapse into a single dominant head across the batch:
$$\bar{\alpha} = \frac{1}{B}\sum_{b=1}^B \alpha_b, \quad \mathcal{L}_{\text{balance}} = \sum_{h=0}^{N-1} \bar{\alpha}_h \log\left(N \cdot \bar{\alpha}_h + 10^{-8}\right)$$

---

## 5. Training Dynamics & AutoResearch Protocol

### 5.1 Hyperparameters Baseline
- **Optimizer**: `AdamW` ($\beta_1 = 0.9, \beta_2 = 0.999, \text{weight\_decay} = 0.01$).
- **Learning Rate Schedule**: Cosine Annealing with linear warmup (5% steps), decaying to $1 \times 10^{-5}$.
- **Base Learning Rates**:
  - LoRA parameters: $\eta_{\text{lora}} = 3 \times 10^{-4}$
  - ThoughtHeads & Judger: $\eta_{\text{heads}} = 5 \times 10^{-4}$
  - Calibration scale $\tau$: $\eta_{\tau} = 1 \times 10^{-2}$
- **Batch Size**: 16 with dynamic sequence padding up to 1024 tokens.

### 5.2 The AutoResearch Protocol (`runner.sh`)
System2 utilizes Karpathy-style automated experimentation constrained by strict invariants:
1. **The Metric Invariant**: Never manually lower or overwrite `.best_metric`. All commits to `main` must strictly exceed the composite evaluation metric on the held-out validation set:
   $$\text{Metric} = 0.5 \times \text{Accuracy} + 0.5 \times (1 - \text{Brier})$$
2. **The Pre-Test Commit Law**: All structural and architectural code must be committed to Git *before* executing `runner.sh`. If a hyperparameter trial regresses, `runner.sh` executes `git reset --hard HEAD` to revert only trial configurations without destroying code infrastructure.
3. **Strictly Non-Autoregressive**: The architecture must remain 100% feedforward. Any proposal adding token autoregression or sequential decoding loops is rejected.

---

## 6. Implementation Code Mapping

| Concept / Math Symbol | Implementation Reference |
| :--- | :--- |
| ModernBERT Frozen Base | [`bert.py:10-45`](file:///Users/thomjames-worrall/system2/bert.py#L10-L45) |
| Top-8 LoRA Adaptation | [`lora.py:15-80`](file:///Users/thomjames-worrall/system2/lora.py#L15-L80) |
| ThoughtHead 2-Layer Transformer | [`heads.py:14-35`](file:///Users/thomjames-worrall/system2/heads.py#L14-L35) |
| Arithmetic Centroid Contrast $[x_i \,\|\, d_i]$ | [`heads.py:44-55`](file:///Users/thomjames-worrall/system2/heads.py#L44-L55) |
| Judger Cross-Attention & Calibration $\tau$ | [`system.py:15-68`](file:///Users/thomjames-worrall/system2/system.py#L15-L68) |
| Scale-Invariant Logit Arbitration ($z$-score) | [`system.py:140-195`](file:///Users/thomjames-worrall/system2/system.py#L140-L195) |
| Composite Multi-Objective Loss Loop | [`train.py:460-585`](file:///Users/thomjames-worrall/system2/train.py#L460-L585) |
| Autonomous Training Loop Harness | [`runner.sh:1-95`](file:///Users/thomjames-worrall/system2/runner.sh#L1-L95) |
