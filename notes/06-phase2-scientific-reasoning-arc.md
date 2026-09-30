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
| **0** | **Phase 2 Baseline (AI2 ARC, 5 heads)** | **0.5353** | **26.69%** | **0.1962** | **COMMITTED**: Baseline established (random chance is 25.0%) |

---

## 5. Diagnostic Analysis of the Phase 2 Baseline

### The Bottleneck: Mean-Pooling All Choices vs. Per-Option Hypothesis Framing
In the Phase 1 setup, input text is a single passage (a news headline), and the model outputs a topic class.
In the naive Phase 2 baseline, the prompt concatenated the question and all four choices:
```
Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
(A) dry palms
(B) wet palms
(C) palms covered with oil
(D) palms covered with lotion
```
When mean-pooled into a single 1024-dim vector, ModernBERT averages all four choices together into one generic bag of embeddings. The downstream 2-layer `ThoughtHead` is then forced to guess which option was correct from this blended summary.

### The Breakthrough Hypothesis: Pairwise Question-Choice Hypothesis Encoding
In true System 2 reasoning, **each choice is literally a distinct hypothesis**:
- *Hypothesis 1*: $\text{Question} \oplus \text{Choice A}$ (dry palms)
- *Hypothesis 2*: $\text{Question} \oplus \text{Choice B}$ (wet palms)
- *Hypothesis 3*: $\text{Question} \oplus \text{Choice C}$ (palms with oil)
- *Hypothesis 4*: $\text{Question} \oplus \text{Choice D}$ (palms with lotion)

When ModernBERT encodes each (Question, Choice) pair separately, bidirectional cross-attention directly evaluates the semantic interaction between the condition ("warm hands by rubbing") and each specific option.
The parallel `ThoughtHead`s score the compatibility of each hypothesis, and the `Judger` cross-attends across all four hypotheses to select the winning derivation and prune the unviable ones!

