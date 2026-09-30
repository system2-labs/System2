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
| **0** | **Phase 2 Baseline (AI2 ARC, 5 heads)** | *Running...* | — | — | *Establishing Phase 2 benchmark* |
