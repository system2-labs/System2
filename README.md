# System2: Parallel Hypothesis Reasoning Architecture

A from-scratch, non-autoregressive "System 2" decision architecture designed to run and train locally on Apple Silicon.

System2 generates parallel hypotheses, estimates calibrated confidence for each, attends across competing candidates, and prunes weak reasoning paths before outputting a decision.

---

## Architecture Overview

```
                      [Input Text]
                           │
                           ▼
          [ModernBERT-large Encoder (395M, Frozen)]
                           │ (1024-dim context vector)
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 [ThoughtHead 1]    [ThoughtHead 2]    [ThoughtHead N]
   (2x Tx Layer)      (2x Tx Layer)      (2x Tx Layer)
   1024 -> 512        1024 -> 512        1024 -> 512
        │                  │                  │
   (Emb 1, Conf 1)    (Emb 2, Conf 2)    (Emb N, Conf N)
        └──────────────────┬──────────────────┘
                           ▼
             [Cross-Attending Judger]
         - Multi-head self-attention over candidates
         - Calibrated scoring & weak hypothesis pruning (<0.3)
                           │
                           ▼
               [Selected Hypothesis & Decision]
```

### Core Design Principles
1. **Frozen Bidirectional Encoder**: Uses ModernBERT-large (395M) strictly as a feature extractor. It produces dense representations without autoregressive drift, sycophancy, or post-hoc RLHF alignment artifacts.
2. **Parallel Thought Heads**: $N$ independent transformer heads generate candidate decision vectors and per-head calibrated confidence scores simultaneously.
3. **Cross-Attending Judger**: Candidates attend to one another via multi-head attention. The Judger scores candidates relative to the full hypothesis space and prunes unviable paths.
4. **Calibrated Confidence**: Confidence is trained using strictly proper scoring incentives (BCE against correctness / Brier score), penalizing overconfident errors.
5. **No Autoregressive Generation**: Fast, single-pass decision making with transparent internal uncertainty.

---

## Current Benchmark & Baseline

Phase 1 validation is benchmarked on **AG News** (4-class topic classification) as a canary pipeline on Apple Silicon (M-series GPU via Metal Performance Shaders / MPS).

| Metric | Benchmark SOTA | Notes |
| :--- | :---: | :--- |
| **Composite Metric** | **0.8998** | $0.5 \times \text{Accuracy} + 0.5 \times (1 - \text{Brier})$ (Baseline was 0.8570) |
| **Test Accuracy** | **88.55%** | 7,600 held-out test samples (+3.94% over baseline) |
| **Brier Score** | **0.0860** | Direct Judger calibration loss (35% error reduction) |

*Trained on 20,000 samples for 2 epochs on MPS (~18 minutes on Apple Silicon).*

---

## Repository Structure

```
├── bert.py            # ModernBERT-large encoder wrapper (frozen)
├── heads.py           # ThoughtHead modules (1024->512, hypothesis & confidence heads)
├── system.py          # System2 top-level module (N heads + cross-attending Judger)
├── data.py            # Dataset loader and tokenization utilities
├── train.py           # Supervised training harness (CE + calibration BCE + judger loss)
├── train_smoke.py     # Fast canary smoke test (200 samples, 1 epoch)
├── eval.py            # Read-only evaluation oracle (Accuracy + Brier score)
├── runner.sh          # Karpathy-style AutoResearch loop runner
├── 01-system-architecture.md  # Architectural design document
├── 02-training-plan.md        # Staged training and milestone roadmap
├── 03-implementation-spec.md  # Code-level implementation specification
└── 04-handoff.md              # Project status, invariants, and handoff log
```

---

## Quickstart

### 1. Requirements
- Python 3.10+
- PyTorch with Apple Silicon MPS support (`torch.backends.mps.is_available() == True`)
- Hugging Face `transformers` and `datasets`

### 2. Smoke Test (Canary)
Run a quick 200-sample canary run (~30 seconds) to verify MPS pipeline and checkpointing:
```bash
python3 train_smoke.py
```

### 3. Evaluate Checkpoint
Run the read-only oracle on the test split:
```bash
python3 eval.py ./models/system2.pt
```
Emits a single JSON metric line:
```json
{"metric": 0.857, "accuracy": 0.8461, "brier": 0.1321}
```

### 4. AutoResearch Loop
The project adheres to an autonomous iterative research loop (`runner.sh`):
1. Modify hyperparameters or architectural knobs in `train.py`.
2. Run `./runner.sh "<description of change>"`.
3. The runner trains the model, runs `eval.py`, and:
   - **Commits and records** the new metric if it improves upon `.best_metric`.
   - **Reverts the working tree** if the metric is worse or equal.

```bash
./runner.sh "experiment: test 8 heads with cosine diversity penalty"
```

---

## Staged Roadmap

- [x] **Phase 1: Pipeline Canary**: End-to-end encoder $\to$ parallel heads $\to$ cross-attending judger $\to$ eval loop on AG News (achieved baseline metric **0.8570**).
- [ ] **Phase 2: Judger & Diversity Optimization**:
  - Align Judger training target to select confident *and correct* heads.
  - Implement pairwise orthogonality / cosine diversity loss across heads.
  - Ablation: Single-head probe baseline comparison.
- [ ] **Phase 3: Reasoning Benchmarks**: Transition from topic classification to multi-step reasoning tasks (GSM8K, AI2 ARC).
- [ ] **Phase 4: Dynamic Head Allocation**: Dynamically scale active head count based on query complexity.
