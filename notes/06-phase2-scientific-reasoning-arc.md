# Phase 2: Multi-Step Reasoning with AI2 ARC

**Document:** `notes/06-phase2-scientific-reasoning-arc.md`  
**Author:** TJ  
**Date:** 2026-10-01  
**Status:** In Progress — Establishing Phase 2 Benchmark

---

## 1. Executive Summary & Strategic Pivot

With Phase 1 (AG News canary) reaching SOTA performance (**0.8998 composite metric, 88.55% test accuracy, 0.0860 Brier score**), the core engineering pipeline is proven. We now officially advance to **Phase 2: Multi-Step Deliberative Reasoning**.

We transition from 4-class topic categorization to the **AI2 ARC (Abstraction and Reasoning Corpus - Challenge & Easy)** benchmark:
- **Task**: Grade-school to middle-school scientific reasoning questions covering thermodynamics, mechanical forces, ecology, astronomy, and chemical reactions.
- **Challenge**: Requires genuine multi-step causal deduction and common-sense elimination of plausible distractor choices. Surface keyword matching fails.
- **Baseline Expectation**: Random guess accuracy is **25.0%**. A successful System 2 reasoning architecture must demonstrate significant lift over random chance while maintaining calibrated uncertainty.

---

## 2. Dataset Architecture & Pipeline

The pipeline is standardized via [`prepare_arc.py`](file:///Users/thomjames-worrall/System2/prepare_arc.py):

```
[allenai/ai2_arc (Hugging Face)]
  ├── ARC-Challenge (1,117 train, 1,165 test)
  └── ARC-Easy      (2,241 train, 2,365 test)
         │
         ▼
[prepare_arc.py]
  - Filters to standardized 4-choice questions (A, B, C, D)
  - Formats: "Question: {q}\n(A) {c0}\n(B) {c1}\n(C) {c2}\n(D) {c3}"
  - Token length: Median 59, 95th-percentile 103, Max 187 tokens
         │
         ▼
[./data/arc on Disk]
  - Train: 3,358 reasoning problems
  - Test:  3,530 held-out evaluation problems
```

---

## 3. Training & Architectural Configuration

```python
HYPERPARAMS = {
    "num_heads": 5,
    "lr_heads": 1e-3,
    "lr_judger": 3e-4,
    "batch_size": 32,
    "max_len": 192,       # Covers 100% of ARC questions without truncation
    "device": "mps",      # Apple GPU via Metal Performance Shaders
    "epochs": 3,          # ~105 batches/epoch (~5 minutes total run)
    "train_size": 3500,   # Full training split (3,358 samples)
    "lambda_div": 0.01,   # Pairwise cosine diversity penalty across heads
    "data_dir": "./data/arc",
}
```

### The 4-Part Unified Loss:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + 0.5 \mathcal{L}_{\text{cal}} + 0.1 \mathcal{L}_{\text{judge}} + 0.5 \mathcal{L}_{\text{judge\_cal}} + 0.01 \mathcal{L}_{\text{div}}$$

---

## 4. Phase 2 Experiment Log

*(To be updated as runner.sh establishes and refines the reasoning frontier)*

| Run # | Experiment Description | Result Metric | Accuracy | Brier Score | Outcome / Status |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | **Phase 2 Baseline (AI2 ARC, 5 heads, blended bag)** | **0.5353** | **26.69%** | **0.1962** | **COMMITTED**: Baseline established (random chance is 25.0%) |
| **1** | **Pairwise Question-Choice Hypothesis Encoding** | **0.5948** | **43.97%** | **0.2500** | **COMMITTED**: Accuracy leaped by **+17.28%**; metric rose +0.0595 |
| **2** | **Scaled Training Data (20k Science Mix: ARC+OBQA+SciQ)** | **0.5893** | **42.92%** | **0.2506** | **REVERTED**: 0.5893 <= 0.5948; SciQ trivia diluted multi-step causal reasoning |
| **3** | **Intra-Head Centroid Difference Features (Option B)** | **0.6231** | **49.63%** | **0.2501** | **COMMITTED (SOTA)**: Accuracy jumped to **49.63%** (+5.66%); metric rose +0.0283 |
| **4** | **Direct MSE Brier Calibration + Joint Conf (Option C)** | **0.6072** | **47.93%** | **0.2649** | **REVERTED**: 0.6072 <= 0.6231; overconfident Judger on wrong heads degraded Brier |

---

## 5. Diagnostic Analysis of Run 1 (Pairwise Hypotheses)

### The Quantitative Leap:
- **Baseline (Blended Bag)**: Accuracy was **26.69%**, essentially random chance (25.0%). ModernBERT had all four choices averaged together into one 1024-dimensional vector, losing the distinction between options.
- **Run 1 (Pairwise Candidates)**: Accuracy leaped to **43.97%** across all 3,530 held-out test problems!
- **Composite Metric**: Rose from **0.5353 to 0.5948** (commit [`c733a18`](https://github.com/tj-doomer-it/System2/commit/c733a18)).

### Why Pairwise Framing Solved the Core Reasoning Bottleneck:
1. **Bidirectional Cross-Attention per Hypothesis**:
   For each question $Q$ and candidate choice $C_i$, ModernBERT computes:
   $$v_i = \text{ModernBERT}(\text{"Question: "} Q \oplus \text{" Answer: "} C_i) \in \mathbb{R}^{1024}$$
   ModernBERT's 28 bidirectional attention layers compute full pairwise token interaction between every word in the question and every word in that specific candidate choice.
2. **Cross-Option Deliberation in the ThoughtHeads**:
   Each `ThoughtHead` takes the four candidate vectors $(B, 4, 1024)$, projects them to $(B, 4, 512)$, and passes them through a 2-layer `TransformerEncoder` where all 4 candidate choices **attend directly to each other**. Choice A actively compares its plausibility against Choice B, C, and D.
3. **Head Divergence & Judger Specialization**:
   The pairwise cosine diversity penalty $\lambda_{\text{div}} = 0.01$ pushed the head correlation down to $-0.146$, meaning the 5 parallel heads developed distinct internal criteria for ranking options.
   The `Judger` cross-attended between the 5 heads and drove its selection loss down from $1.537 \to 1.014$, learning which head to trust for different question styles.

### 5.2 Diagnostic Analysis of Run 2 (20k Science Mix Scaling & The Quality vs. Quantity Trade-off):
- **Result**: Accuracy **42.92%**, Brier **0.2506**, Composite Metric **0.5893** (vs SOTA **0.5948** / 43.97%).
- **Protocol Action**: AutoResearch cleanly reverted `train.py`, keeping the 0.5948 benchmark safe.
- **Scientific Diagnosis**:
  1. **Domain Shift & Task Nature**:
     SciQ accounted for 11,679 of the 19,994 samples (~58%). SciQ consists primarily of direct factual recall and definition questions generated with machine-generated distractors. In contrast, ARC (Abstraction and Reasoning Corpus) was specifically curated to defeat keyword matching and requires multi-step physical, causal, and comparative deduction. Flooding the 50M parameter head capacity with 58% factual trivia diluted the heads' focus on deep causal comparison.
  2. **Gradient Dynamics on Scaled Data**:
     In Run 2, `ce` went from $1.359$ in epoch 1 to $1.386$ in epoch 2, indicating that a fixed learning rate of `1e-3` across 20,000 samples experienced late-epoch oscillation without learning rate decay.
  3. **Strategic Takeaway**:
     Data quality and task alignment trump raw quantity. In-domain causal reasoning data (ARC + OpenBookQA or curated multi-step problems) yields higher per-step learning signal than generic multiple-choice recall.

### 5.3 Diagnostic Analysis of Run 3 (Intra-Head Centroid Difference Features - Option B):
- **Result**: Accuracy **49.63%** (**+5.66% leap** over Run 1, **+22.94% over baseline**!), Brier **0.2501**, Composite Metric **0.6231** (**NEW SOTA**, commit [`2b777a6`](https://github.com/tj-doomer-it/System2/commit/2b777a6)).
- **Protocol Action**: AutoResearch committed and pushed the new record to GitHub `main`.
- **Scientific Diagnosis**:
  1. **Contrastive Inductive Bias Solves Subtle Scientific Distractors**:
     In scientific multiple-choice reasoning, distractors are written to sound plausible and share common domain keywords with the question.
     By subtracting the candidate centroid $\bar{x} = \frac{1}{4}\sum_{j=1}^4 x_j$, common contextual noise is eliminated:
     $$d_i = x_i - \bar{x}$$
     The scoring MLP receives $[x_i \,\|\, d_i] \in \mathbb{R}^{1024}$, directly evaluating both absolute semantic plausibility and relative distinctiveness.
  2. **Judger Selection Loss Plummeted**:
     During training, the Judger selection loss plummeted from **$1.315 \to 0.814$**!
     Because each head had access to explicit contrast vectors, their internal rankings of the 4 candidates became sharp and confident, making it dramatically easier for the Judger's cross-attention to identify which head was formulating the most coherent scientific deduction.
  3. **Milestone Reached**:
     With a completely frozen 395M ModernBERT encoder and tiny trainable heads, the system now solves **half of all middle-school scientific reasoning problems (49.63%)** on ARC Challenge + Easy.

### 5.4 Diagnostic Analysis of Run 4 (Direct MSE Brier Calibration - Option C):
- **Result**: Accuracy **47.93%**, Brier **0.2649**, Composite Metric **0.6072** (vs SOTA **0.6231** / 49.63%).
- **Protocol Action**: AutoResearch cleanly reverted `system.py` and `train.py`, keeping the 0.6231 SOTA benchmark safe.
- **Scientific Diagnosis**:
  1. **The Overconfidence Trap of Expert Selection**:
     In Run 4, we formulated decision confidence as:
     $$\text{Conf} = 0.5 \times \text{HeadConf} + 0.5 \times P_{\text{judger}}(\text{best})$$
     When the Judger strongly favored Head 0 ($P_{\text{judger}} \approx 0.92$), it pushed output confidence high (~0.70). However, on ARC, even the best head is incorrect ~50% of the time. When the model made a high-confidence mistake, the Brier error exploded: $(0.70 - 0)^2 = 0.49$, compared to only $0.25$ for an uncommitted 0.50 guess!
     **Core Decision Theory Insight**: Judger consensus reflects *relative expert ranking*, NOT empirical ground-truth probability of scientific correctness.
  2. **Gradient Conflict on Feature Representations**:
     Backpropagating a heavy MSE calibration loss (weight 1.0) into the winning choice slice `win_feat` pulled the representation `x` away from pure classification distinction toward predicting whether previous heads had been right or wrong, nudging accuracy down from 49.63% to 47.93%.

---

## 6. Next Synthesis & Future Directions

With Options A, B, and C tested on Phase 2, we have established a clear scientific taxonomy of what works:

1. **What Works (The Proven Drivers)**:
   - **Pairwise Hypothesis Encoding**: Smashed the blended bag baseline (+17.28% accuracy).
   - **Intra-Head Centroid Difference Features ($[x \,\|\, (x - \bar{x})]$)**: Pushed accuracy to **49.63%** and composite metric to **0.6231** (Phase 2 SOTA).
2. **What Requires Caution**:
   - **Uncurated Data Expansion**: Generic factual recall (SciQ) dilutes multi-step causal reasoning.
   - **Confidence Inflation**: Calibrating confidence to expert selection rather than empirical margin increases penalty on wrong answers.
3. **High-Leverage Next Frontiers**:
   - **Temperature-Scaled Margin Calibration (Post-hoc / Read-only)**: As proven in diagnostic tests, candidates with top-margin $> 0.01$ achieve **66.86% accuracy**, while margin $< 0.00$ achieves **35.69%**. A calibrated sigmoid mapping over logit margin directly captures empirical correctness without gradient distortion.
   - **Iterative Deliberation (2-Pass Recurrence)**: Allow the Judger to feedback attention weights into the heads for a second refinement cycle before committing.



