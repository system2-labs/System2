# Baseline Run & Staged Hypotheses

**Date:** 2026-09-30  
**Commit:** [`2bf8eb8`](https://github.com/tj-doomer-it/System2/commit/2bf8eb8)  
**Status:** Completed & Recorded in `.best_metric`

---

## 1. Baseline Run Summary

The baseline run finished cleanly, recorded `.best_metric`, and pushed commit `2bf8eb8` to GitHub `main`.

### 📊 Baseline Results (20k train, 2 epochs, 7.6k test)

| Metric | Score | Expectation from `04-handoff.md` | Status |
| :--- | :---: | :---: | :---: |
| **Composite Metric** | **0.8570** | 0.55 – 0.70 | **Well exceeded** |
| **Accuracy** | **84.61%** | 60% – 75% | **Well exceeded** |
| **Brier Score (Calibration)** | **0.1321** | — | Strong baseline calibration |

### 🔍 Loss Trajectory Observations

- **Epoch 1**: `ce = 1.093` | `cal = 0.541` | `judge = 0.558`
- **Epoch 2**: `ce = 0.669` | `cal = 0.435` | `judge = 1.032`

Notice how classification error (`ce`) dropped by nearly 40% and calibration error (`cal`) dropped by 20%, but **judge loss almost doubled from 0.558 to 1.032**. 

This confirms the exact pathology diagnosed in the architecture review: 
`target = confs.max(dim=1).indices` penalizes the judger whenever it prefers a head that isn't the single highest raw confidence head, even if that head is wrong.

---

## 2. Assessment of Hypotheses & Decisions

We have established our benchmark ($0.8570$). The following 3 experiments are scheduled sequentially using `./runner.sh`:

### Test 1: Fix Judger Target in `train.py` (High Priority)
- **Diagnosis**: Change `target` from naive `confs.max()` to the highest-confidence head *that actually got the label correct* (falling back to highest-confidence if no heads predicted correctly).
- **Hypothesis**: Will stabilize judger loss in epoch 2, align the judger with true ground truth, and prevent the judger from being trained to endorse confident-wrong hypotheses.
- **Decision**: Implement in `train.py` and run `./runner.sh "fix: judger targets correct and confident heads"`.

### Test 2: Head Diversity Incentive (Orthogonality / Cosine Distance)
- **Diagnosis**: All 5 heads receive the exact same 1024-dim mean-pooled sentence embedding and the same loss, leading to symmetry collapse (heads becoming redundant copies).
- **Hypothesis**: Adding a small pairwise cosine similarity penalty across head representations:
  $$\mathcal{L}_{\text{div}} = \frac{1}{\binom{N}{2}} \sum_{i < j} \cos(e_i, e_j)$$
  will encourage heads to learn distinct feature subspaces and specialize.
- **Decision**: Add diversity loss $\lambda_{\text{div}} \mathcal{L}_{\text{div}}$ in `train.py` and run `./runner.sh "feat: add cosine diversity loss between thought heads"`.

### Test 3: Single-Head Probe Ablation (`num_heads = 1`)
- **Diagnosis**: The core architectural question for Phase 1: Does parallel-head hypothesis generation + cross-attention provide actual value over a single head on AG News?
- **Hypothesis**: If 5 heads + judger does not beat `num_heads = 1`, the parallel-hypothesis machinery is dead weight for simple 4-class classification and requires harder reasoning tasks (e.g. GSM8K).
- **Decision**: Test `num_heads = 1` in `train.py` via `./runner.sh "ablation: 1 head vs 5 heads"`. If metric $< 0.8570$, `runner.sh` automatically rejects and reverts.

---

## 3. Experiment Execution Log

*(To be updated after each runner execution)*

| Run # | Experiment Description | Result Metric | Accuracy | Brier Score | Outcome / Action |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **0** | Baseline (5 heads, raw conf target) | 0.8570 | 84.61% | 0.1321 | **Committed** baseline |
| **1** | Fix Judger Target (correct & confident) | *Pending* | — | — | *Executing next* |
| **2** | Head Diversity Penalty | *Pending* | — | — | *Scheduled* |
| **3** | Single-Head Ablation (`num_heads = 1`) | *Pending* | — | — | *Scheduled* |
