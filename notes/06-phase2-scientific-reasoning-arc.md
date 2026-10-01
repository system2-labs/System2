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
| **1** | **Pairwise Question-Choice Hypothesis Encoding** | **0.5948** | **43.97%** | **0.2500** | **COMMITTED (SOTA)**: Accuracy leaped by **+17.28%**; metric rose +0.0595 |
| **2** | **Scaled Training Data (20k Science Mix: ARC+OBQA+SciQ)** | **0.5893** | **42.92%** | **0.2506** | **REVERTED**: 0.5893 <= 0.5948; SciQ trivia diluted multi-step causal reasoning |

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

---

## 6. Next Steps: Intra-Head Difference Features (Option B) & Calibration Tuning (Option C)

With raw data expansion tested and analyzed, our next high-leverage focus turns to architectural inductive biases and calibration:

### 1. The Data Volume Bottleneck:
- Our current training set is only **3,358 samples** (ARC-Challenge + ARC-Easy train splits).
- 3,358 samples is very small for training a multi-head reasoning network to generalize across physics, biology, chemistry, and earth science.
- Because ModernBERT remains completely frozen, each epoch on 3,358 samples takes only **~50 seconds** on Apple Silicon (Metal/MPS).

### 2. High-Value Scientific & Common-Sense Datasets for Pre-Training / Data Expansion:
We can dramatically expand the training volume with zero architecture changes:
1. **OpenBookQA** (~5,000 questions):
   - Grade-school science questions modeled after open-book exams, requiring multi-hop reasoning over core scientific facts.
2. **SciQ** (~13,679 questions):
   - Middle and high school science exam questions with 4 multiple-choice options covering physics, chemistry, biology, and earth science.
3. **CommonsenseQA** (~12,102 questions):
   - Multiple-choice questions requiring common-sense relational reasoning (from ConceptNet).
4. **Combined Corpus**:
   - Merging ARC (3.3k) + SciQ (13.6k) + OpenBookQA (5k) gives **~22,000 high-quality scientific reasoning problems** (~6.5x more data).
   - Training time for 22,000 samples at 63 problems/sec on MPS: **~5.8 minutes per epoch**!

### 3. Immediate Architectural Next Steps on ARC:
1. **Cross-Candidate Subtraction / Difference Features**:
   - In addition to self-attention across candidates, augment candidate representations with relative difference vectors: $(x_i - \frac{1}{4}\sum_j x_j)$ to explicitly highlight what makes choice $i$ different from the average distractor.
2. **Confidence Calibration Tuning**:
   - In Run 1, Brier score was $0.2500$ (with accuracy $43.97\%$). Calibrating head confidences to better reflect true uncertainty will raise the composite metric further.
3. **Scale Epochs and Learning Rate Schedule**:
   - Train for 5–8 epochs with a cosine decay learning rate schedule to allow the ThoughtHeads to fully settle into their specialized reasoning niches.


