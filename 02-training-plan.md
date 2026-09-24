# System2 — Training Plan

**Status:** Planning document v0.1
**Created:** 2026-09-24

---

## 1. Data Pipeline

### 1.1 Source Datasets

**Tier 1 — Reasoning (critical, scarce):**

| Dataset | Size | Description | Source |
|---|---|---|---|
| GSM8K | 8.5K problems | Grade-school math word problems with solutions | [GSM8K](https://huggingface.co/datasets/gsm8k) |
| MATH | 13K problems | Competition-level math problems with step-by-step solutions | [MATH](https://huggingface.co/datasets/HuggingFaceH4/MATH) |
| AI2 ARC | 7.8K questions | Grade-school science multiple-choice questions | [AI2 ARC](https://huggingface.co/datasets/allenai/ai2_arc) |
| Grove | ~5K reasoning traces | General reasoning with step-by-step solutions | [Grove](https://huggingface.co/datasets/ai2-grove) |

**Tier 2 — Classification (abundant, easy):**

| Dataset | Size | Labels | Use Case | Source |
|---|---|---|---|---|
| AG News | 120K samples | 4 categories | News classification | [AG News](https://huggingface.co/datasets/ag_news) |
| Banking77 | 15K samples | 72 intents | Customer service intent | [Banking77](https://huggingface.co/datasets/PragmaticWavertree/Banking77) |
| MASSIVE | 23K samples | 71 intents | Voice assistant intents | [MASSIVE](https://huggingface.co/datasets/MASSENGER/MASSIVE) |

**Tier 3 — General language (pre-training corpus):**

| Dataset | Size | Description | Source |
|---|---|---|---|
| CommonCrawl subset | ~10GB text | General web text for encoder pre-training | Hugging Face datasets |
| Wikipedia | ~6GB text | Encyclopedic text, clean, structured | Hugging Face datasets |

### 1.2 Synthetic Data Generation

For reasoning traces where ground truth is sparse:

```python
# Pseudocode: Generate reasoning traces using Qwen3.8
def generate_reasoning_trace(query):
    # 1. Prompt Qwen3.8 with chain-of-thought instructions
    prompt = f"Problem: {query}\nSolve step by step."
    answer = qwen.generate(prompt, temperature=0.3)
    
    # 2. Extract structured reasoning steps
    steps = extract_steps(answer)
    
    # 3. Verify each step (if verification available)
    verified_steps = [s for s in steps if verify(s)]
    
    return verified_steps if len(verified_steps) > 0 else None
```

**Reference:** "Self-Alignment: Training Language Models with Self-Generated Data" — using LLMs to create training data for themselves.

**Caveat:** Synthetic data carries the base model's biases and blind spots. Use sparingly and validate against held-out ground-truth datasets.

---

## 2. Training Pipeline (Path A: Build Our Own)

### Phase 1: Encoder Pre-training (Week 1-2)

**Goal:** Pre-train ModernBERT-large on a general corpus so it produces useful hidden representations for downstream heads.

```
Pre-training data: CommonCrawl subset (10GB) + Wikipedia (6GB)
Model: ModernBERT-large (395M params)
Training: Next-token prediction (standard BERT objective)
Hardware: M4 Pro (128GB)
Estimate: ~1 week (single GPU, batch size 32)
Output: Pre-trained encoder checkpoint
```

**Reference:** Standard BERT pre-training methodology (Devlin et al., 2019). ModernBERT adds RoPE embeddings, SwiGLU activations, and other optimizations (Shoeybi et al., 2024).

### Phase 2: Head Initialization (Week 2)

**Goal:** Initialize 5-10 parallel thought heads with random weights.

```
Heads: 5-10 transformer blocks (~30-50M params each)
Initialization: Random (Xavier uniform)
Input: Encoder's hidden states (last token embedding)
Output per head: One hypothesis + confidence score
```

### Phase 3: Head Fine-tuning (Week 3-4)

**Goal:** Train each head to produce accurate hypotheses.

```
Training data: GSM8K, MATH, AI2 ARC, AG News, Banking77
Method: RLCD (Reinforcement Learning from Calibrated Decisions)
Reward: Negative Brier score (strict proper scoring rule)
Optimizer: REINFORCE with group-mean baseline (GRPO-style)
Learning rate: 1e-5 (AdamW)
Batch size: 16 (limited by GPU memory)
Steps: ~10K (3-4 epochs over training data)
Estimate: ~1 week
```

**Reference:** Laya's RLCD training — trained with strict proper scoring rules so that honest probability reporting maximizes expected reward.

### Phase 4: Judger Training (Week 5)

**Goal:** Train the judger/pruner to correctly score and select among hypotheses.

```
Input: All (hypothesis, confidence) pairs from all heads
Task: Identify the best hypothesis (highest probability of being correct)
Method: RLCD on judger's scoring function
Training data: Same as head fine-tuning, but labeled by which hypothesis is correct
Model: Small transformer (~50M params)
Output: Selected hypothesis + calibrated confidence
```

### Phase 5: End-to-End Fine-tuning (Week 6-7)

**Goal:** Fine-tune all components together using autoresearch loop.

```
Loop: Propose configuration → Judge on validation set → Keep better, discard worse
Frozen: Encoder (after pre-training)
Tunable: Head weights, judger weights
Validation: AG News test set + GSM8K test set
Metrics: Accuracy, calibration (ECE, Brier score), latency
```

**Reference:** Karpathy's autoresearch loop (propose → judge → commit/revert). Your implementation: https://github.com/tj-doomer-it/test-training

---

## 3. Training Pipeline (Path B: Augment Existing)

### Phase 1: Base Model Setup (Day 1-2)

```
Install Splash: brew install incoai/tap/splash
Serve model: splash serve --model incoai/Qwen3.8-27B-Splash
Test inference: curl localhost:8000/v1/chat/completions
Verify: ~74 tok/s on M4 Pro (Qwen3.8-27B, 4-bit)
```

### Phase 2: Query Wrapper Development (Day 3-5)

**Goal:** Build the parallel query interface.

```python
# Pseudocode: Parallel query system
import requests

def parallel_query(query, num_heads=5):
    """Send N parallel queries to base model, receive N completions."""
    prompts = generate_framings(query, num_heads)
    completions = []
    for prompt in prompts:
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "Qwen3.8-27B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.7
            }
        )
        completions.append(response.json()["choices"][0]["message"]["content"])
    return completions

def generate_framings(query, num_heads):
    """Generate N different ways to frame the query."""
    framings = [f"Consider this: {query}"]
    framings.append(f"Analyze: {query}")
    framings.append(f"Interpret: {query}")
    # ... add more framings
    return framings[:num_heads]
```

### Phase 3: Judger Training (Day 6-10)

**Goal:** Train a lightweight judger (~50M params) to evaluate and select from completions.

```
Training data: Queries + verified good answers (from Tier 1 & 2 datasets)
Method: RLCD (same as Path A)
Model: Small transformer (50M params, same as Path A's judger)
Training steps: ~5K (2 epochs)
Estimate: ~3 days
```

### Phase 4: End-to-End Testing (Day 11-14)

```
Test on: Held-out test set (GSM8K test, AG News test)
Metrics: Accuracy, calibration, latency, throughput
Compare: Single-pass vs. System2 (parallel heads)
Iterate: Adjust number of heads, confidence threshold, framing strategy
```

---

## 4. Evaluation Metrics

### 4.1 Accuracy

- **Exact match:** Does the hypothesis match the ground-truth answer?
- **Soft accuracy:** How close is the hypothesis to the ground truth (for open-ended questions)?

### 4.2 Calibration

- **Expected Calibration Error (ECE):** Are confidence scores accurate? (If confidence = 80%, is the model correct 80% of the time?)
- **Brier score:** How well-calibrated are the probability estimates? (Lower = better)
- **ECE bins:** Divide predictions into confidence bins (0-10%, 10-20%, ... 90-100%) and compare average confidence to actual accuracy within each bin.

**Reference:** Guo et al. (2017) "On Calibration of Modern Neural Networks"

### 4.3 Latency

- **Time per query:** How long does it take to generate + score + select?
- **Throughput:** How many queries per second?
- **Memory usage:** How much RAM does the model use?

### 4.4 System 2 Quality

- **Pruning ratio:** What fraction of hypotheses are pruned? (Higher = better pruning)
- **Convergence rate:** How many iterations until the best hypothesis emerges?
- **Diversity:** Are the hypotheses diverse, or do they all converge to the same answer? (Diversity is a feature, not a bug — it lets the system explore multiple reasoning paths.)

---

## 5. Hardware Requirements

### Path A (Build Our Own)

| Component | Size (4-bit) | Memory | Notes |
|---|---|---|---|
| Encoder (ModernBERT-large) | 1.6GB | 1.6GB | Pre-trained, frozen after Phase 1 |
| 10 Thought heads | 1.6GB | 1.6GB | Trained in Phases 3-5 |
| Judger | 0.2GB | 0.2GB | Trained in Phase 4 |
| **Total model** | **~3.4GB** | **~3.4GB** | Fits on any 16GB+ device |
| Training buffers | ~2GB | 2GB | Batch size 16, gradients |
| **Peak during training** | — | **~5-6GB** | Well within 128GB |

**Training time estimates:**
- Encoder pre-training: ~1 week (M4 Pro, single GPU)
- Head fine-tuning: ~1 week (M4 Pro, single GPU)
- Judger training: ~3 days (M4 Pro, single GPU)
- End-to-end fine-tuning: ~2 weeks (M4 Pro, single GPU)
- **Total: ~5-6 weeks**

### Path B (Augment Existing)

| Component | Size (4-bit) | Memory | Notes |
|---|---|---|---|
| Base model (Qwen3.8-27B) | 17GB | 17GB | Pre-trained, frozen |
| Query wrapper | ~0.5GB | 0.5GB | Lightweight, inference-only |
| Judger | 0.2GB | 0.2GB | Trained in Phase 3 |
| **Total model** | **~17.7GB** | **~17.7GB** | Fits on 48GB+, comfortable on 128GB |

**Training time estimates:**
- Query wrapper development: ~3 days
- Judger training: ~3 days
- End-to-end testing: ~4 days
- **Total: ~1-2 weeks**

---

## 6. Risks and Mitigations

### 6.1 Training Data Risks

**Problem:** Clean reasoning traces are rare. Most datasets have final answers, not step-by-step reasoning.

**Mitigation:** Use synthetic data generation (Qwen3.8) to create training traces. Validate synthetic data against held-out ground-truth sets. Use Tier 1 datasets (GSM8K, MATH) where possible, supplement with synthetic data for more complex queries.

### 6.2 Calibration Risks

**Problem:** Standard LLMs are poorly calibrated — they over-confidently predict wrong answers. RLCD addresses this, but may not be sufficient.

**Mitigation:** Train with strict proper scoring rules (RLCD). Validate calibration on held-out test sets. Adjust confidence threshold dynamically based on query difficulty.

### 6.3 Latency Risks

**Problem:** Parallel hypothesis generation increases latency compared to single-pass models.

**Mitigation:** Use fast inference runtimes (Splash, MTPLX). Tune the number of heads (5-10 is a tradeoff between accuracy and speed). Use caching for repeated queries.

### 6.4 Scaling Risks

**Problem:** What happens when we scale beyond 10 heads? What happens when we add more complex queries?

**Mitigation:** Experiment with 3, 5, 10, 20 heads. Measure accuracy vs. head count. Find the sweet spot. Start simple, iterate.

---

## 7. Milestones

### Milestone 1: Baseline System (Week 2, Path B)

- [ ] Install Splash or MTPLX
- [ ] Serve Qwen3.8-27B locally
- [ ] Build a single-pass query wrapper (no parallelism)
- [ ] Test on a small dataset
- **Success criterion:** System produces correct answers at reasonable speed

### Milestone 2: Parallel Query Layer (Week 3, Path B)

- [ ] Build parallel query wrapper (5 heads)
- [ ] Test on the same dataset
- **Success criterion:** Parallel queries improve accuracy or confidence over single-pass

### Milestone 3: Judger Training (Week 4, Path B)

- [ ] Build judger (~50M params)
- [ ] Train with RLCD on validation set
- **Success criterion:** Judger correctly selects the best hypothesis >80% of the time

### Milestone 4: End-to-End System (Week 5, Path B)

- [ ] Connect all components (encoder → heads → judger)
- [ ] Test on held-out test set
- **Success criterion:** System achieves calibrated accuracy >75% on GSM8K test

### Milestone 5: Prototype v1 (Week 6-8, Path A)

- [ ] Pre-train encoder (ModernBERT-large)
- [ ] Initialize 5-10 thought heads
- [ ] Train heads with RLCD
- [ ] Train judger
- [ ] End-to-end fine-tuning via autoresearch loop
- **Success criterion:** System achieves calibrated accuracy >70% on GSM8K test, with latency <500ms per query

---

## 8. Reference Implementations

### Laya (decision model)

- **Repository:** https://github.com/NandhaKishorM/laya
- **Key insight:** Non-autoregressive, single forward pass, trained with RLCD
- **Architecture:** ModernBERT-large (395M) + decision head (26M) = 421M total
- **Training:** RLCD (strict proper scoring rules), REINFORCE with group-mean baseline
- **Benchmarks:** 0.766 accuracy on typed-decisions (4 workflows), 0.950 on AG News (4 labels)
- **Latency:** 33ms per question (T4 GPU), 6-8× faster than TypeSafe Jev

### Splash (inference runtime)

- **Repository:** https://github.com/incoai/splash
- **Key insight:** Apple Silicon-optimized inference with speculative decoding
- **Features:** Per-model draft models (DFlash 2), custom Metal kernels, zero-copy memory
- **Performance:** 74 tok/s (Qwen3.8-27B, M5 Pro 48GB), 2× faster than next-fastest engine

### MTPLX (inference runtime)

- **Repository:** https://github.com/youssofal/MTPLX
- **Key insight:** Native multi-token prediction (uses the model's own MTP heads)
- **Features:** No external draft model needed, 1.6-2.2× speedup on Apple Silicon
- **Supported models:** Qwen 3.5/3.6/3.8 (with built-in MTP heads)

### Karpathy's autoresearch (training pattern)

- **Repository:** https://github.com/tj-doomer-it/test-training (your implementation)
- **Pattern:** Propose → Judge → Commit (better) / Revert (worse)
- **Use case:** Hyperparameter optimization, architecture search, training data selection

---

## Sources

- Selsam, D. (2026). "Personal Statement on AI Risk." OpenAI.
- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Devlin, J., et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding."
- Shoeybi, M., et al. (2024). "ModernBERT: Modernized BERT Architecture." Hugging Face.
- Guo, C., et al. (2017). "On Calibration of Modern Neural Networks."
- Laya documentation. ConvAI Innovations. https://huggingface.co/convaiinnovations/laya
- Splash documentation. inco.ai. https://github.com/incoai/splash
- MTPLX documentation. https://github.com/youssofal/MTPLX
- GSM8K dataset. Hugging Face. https://huggingface.co/datasets/gsm8k
- MATH dataset. Hugging Face. https://huggingface.co/datasets/HuggingFaceH4/MATH
