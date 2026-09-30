# Full-Stack Requirements & System Blueprint

**Document:** `notes/05-full-stack-requirements-and-blueprint.md`  
**Author:** TJ  
**Date:** 2026-10-01  
**Context:** The Complete Technical Specification to Build, Train, and Deploy System2

---

## Executive Overview

To build and train a functional **System2** reasoning and calibrated decision system, you do not need multi-million-dollar compute clusters or closed proprietary APIs. Because the architecture decouples representation extraction (frozen encoder) from hypothesis generation and deliberation (trainable heads + judger), the computational footprint is radically compact.

Below is the complete architectural specification across **Hardware, Base Models, Data, Software, and Mathematical Objectives**.

---

## 1. Hardware & Compute Specifications

```
                       [Compute Tiers]
                              │
         ┌────────────────────┴────────────────────┐
         ▼                                         ▼
   [Local Option]                           [Cloud / Scale]
  Apple Silicon Mac                        Single NVIDIA GPU
  (M2/M3/M4 Pro/Max)                      (RTX 4090 / A10G / A100)
  32GB–128GB Unified Memory               16GB–24GB VRAM
  Trainable: ~50M–150M params             Trainable: up to 20 heads
  Runtime: ~15–20 min / 2 epochs          Runtime: ~5–8 min / 2 epochs
```

### Local Option (Current Verified Setup)
- **Machine**: Apple Silicon Mac (M2/M3/M4 Pro or Max) with 32GB to 128GB of unified memory.
- **Trainable Footprint**: 
  - The 395M-parameter ModernBERT encoder is **frozen** and only executes feedforward passes.
  - The actual trainable parameters (5 `ThoughtHead` modules + cross-attending `Judger`) total only **~50M to 150M parameters**.
- **Execution Speed**: A complete 2-epoch training run over 20,000 samples takes **~15 to 20 minutes** on Apple Silicon via Metal Performance Shaders (`mps`).

### Cloud / Linux Option (For Scaling Up)
- **Machine**: A single consumer or cloud GPU with 16GB to 24GB VRAM (e.g., NVIDIA RTX 4090, A10G, or A100).
- **Advantage**: Scales easily to 10–20 thought heads, larger batch sizes ($64+$), and longer sequence lengths ($512+$ tokens).
- **Cluster Footprint**: Requires **zero distributed multi-node clustering, TPU pods, or NCCL orchestrators**.

---

## 2. The Foundation: Pre-Trained Encoders

System2 requires a high-capacity, non-autoregressive bidirectional encoder to act as an objective, unbiased "fact reader":

| Encoder Model | Parameters | Context Window | Best Use Case |
| :--- | :---: | :---: | :--- |
| **ModernBERT-large** *(Default)* | 395M | Up to 8,192 tokens | General reasoning, topic classification, decision triage |
| **PubMedBERT / BioBERT** | 110M–340M | 512 tokens | Clinical intake, emergency triage, biomedical literature |
| **CodeBERT / StarCoder-Encoder** | 125M–300M | 1,024–4,096 tokens | Agent security gatekeeper, terminal command screening |

### Why the Encoder Must Stay Frozen Initially
1. **Compute Efficiency**: Pretraining a 400M+ encoder from scratch requires weeks of GPU cluster time and terabytes of raw text. Freezing cuts training time by $>90\%$.
2. **Neutral Representation**: A frozen encoder provides stable, high-fidelity embeddings without catastrophic forgetting.
3. **Anti-Sycophancy**: By completely avoiding autoregressive fine-tuning and conversational RLHF alignment, the encoder cannot be manipulated by tone or conversational framing.

---

## 3. Data Requirements (The Fuel)

Calibrated decision heads require datasets with **verifiable, deterministic ground truth**:

### Phase 1: Decision & Classification Tasks (Canary Pipeline)
- **Target Volume**: 20,000 to 100,000 labeled examples.
- **Representative Datasets**:
  - `AG News` (4 topic classes, 120k samples) — *Current baseline testbed*
  - `Banking77` (72 fine-grained customer intent classes)
  - `MASSIVE` (60+ cross-domain intents across multiple languages)
- **Core Purpose**: Proves gradient stability, device placement, tensor flow, and baseline Brier calibration.

### Phase 2: Deliberative Reasoning Tasks (True System 2)
- **Target Volume**: 10,000 to 50,000 problems with verifiable answers.
- **Representative Datasets**:
  - `GSM8K` (8,500 grade-school math word problems with step-by-step arithmetic)
  - `AI2 ARC` (7,800 multiple-choice science reasoning questions)
  - `SciFact` (Expert-verified scientific claims with evidence abstracts)
- **Core Purpose**: Gives competing heads room to explore distinct hypothesis spaces, formulate intermediate equations, and allow the cross-attending Judger to resolve conflicting deductions.

---

## 4. Mathematical Objectives & Multi-Loss Formulation

System2 trains all components end-to-end using a **4-part composite loss** computed in a single unified backward pass:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cal}} \mathcal{L}_{\text{cal}} + \lambda_{\text{judge}} \mathcal{L}_{\text{judge}} + \lambda_{\text{div}} \mathcal{L}_{\text{div}}$$

```
                                [Composite Autograd Graph]
                                            │
         ┌──────────────────┬───────────────┴───────────────┬──────────────────┐
         ▼                  ▼                               ▼                  ▼
    (1. Task Loss)   (2. Calibration)                (3. Judger Loss)   (4. Diversity Loss)
     Cross-Entropy    Binary Cross-Entropy            Cross-Entropy      Pairwise Cosine
     on every head    vs. 𝟏[head == true_label]       on Judger scores   sim between heads
```

### 1. Multi-Head Task Loss ($\mathcal{L}_{\text{task}}$)
Every head is optimized on ground-truth targets to ensure all heads master the task:
$$\mathcal{L}_{\text{task}} = \frac{1}{N} \sum_{n=1}^N \text{CrossEntropy}(\text{logits}_n, y)$$

### 2. Confidence Calibration Loss ($\mathcal{L}_{\text{cal}}$)
Binary Cross-Entropy (BCE) compares each head's predicted confidence against its empirical correctness:
$$\text{target}_n = \mathbf{1}[\operatorname{argmax}(\text{logits}_n) == y]$$
$$\mathcal{L}_{\text{cal}} = \frac{1}{N} \sum_{n=1}^N \text{BCE}(\operatorname{conf}_n, \text{target}_n)$$

### 3. Judger Selection Loss ($\mathcal{L}_{\text{judge}}$)
Trains the cross-attention Judger to award its highest score to the most confident, correct hypothesis:
$$\mathcal{L}_{\text{judge}} = \text{CrossEntropy}(\text{scores}, \text{target}_{\text{best}})$$

### 4. Anti-Groupthink Diversity Loss ($\mathcal{L}_{\text{div}}$)
Penalizes pairwise cosine similarity between normalized head embeddings $e_i, e_j$ to break symmetry and prevent groupthink collapse:
$$\mathcal{L}_{\text{div}} = \frac{1}{\binom{N}{2}} \sum_{i < j} \cos(e_i, e_j)$$

---

## 5. Software & Tooling Stack

- **Core Runtime**: Python 3.10+, PyTorch (configured with Apple `mps` or NVIDIA `cuda`).
- **Data & Model Hub**: Hugging Face `transformers` and `datasets`.
- **The "Judge" (Read-Only Evaluation Oracle)**:
  - An immutable oracle (`eval.py`) that strictly scores checkpoints on held-out test splits.
  - Strictly proper composite metric:
    $$\text{Metric} = 0.5 \times \text{Accuracy} + 0.5 \times (1 - \text{Brier Score})$$
- **Autonomous AutoResearch Harness**:
  - `runner.sh` script adhering to Karpathy’s `test-training` pattern.
  - Automatically compiles, trains, queries the oracle, records `.best_metric`, commits improvements, and resets regressions.

---

## 6. Summary Checklist to Build & Train

- [x] **1. Compute**: Modern Apple Silicon Mac (M-series, 32GB+ RAM) or single 24GB GPU.
- [x] **2. Encoder**: Downloaded ModernBERT-large weights (frozen feature extractor).
- [x] **3. Data**: 20k–50k labeled problem-answer pairs with deterministic validation.
- [x] **4. Loss Formulation**: 4-part loss (`Task` + `Calibration` + `Judger` + `Diversity`).
- [x] **5. AutoResearch Harness**: Immutable evaluation oracle + automated commit/revert runner loop.
