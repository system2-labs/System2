# System2 — Multi-Thread Reasoning Model

**Status:** Planning document v0.1
**Created:** 2026-09-24
**Author:** TJ (Moretti assisting)
**Goal:** Design a small, laptop-trainable reasoning model that generates parallel hypotheses, scores them, and prunes weak ones — a System 2 architecture that runs on Apple Silicon.

---

## 1. Core Problem

Current large language models are **autoregressive** — they generate tokens one at a time, sequentially. They are System 1: fast, parallel in encoding, sequential in output.

The alternative: generate **parallel hypotheses** (different interpretations, strategies, or reasoning paths), score each one, discard the weak ones, and iterate. This is System 2: slow, deliberate, hypothesis-driven.

**Reference:** Kahneman, *Thinking, Fast and Slow* (2011) — System 1 (fast, intuitive) vs System 2 (slow, deliberative) thinking.

---

## 2. Two Paths

### Path A: Build Our Own (Full Stack)

Build a new architecture from the ground up: encoder + parallel thought heads + judge/pruning layer, trained end-to-end on a laptop.

**Stack:**
- **Encoder:** ModernBERT-large (395M params) or mmBERT-base (322M)
- **Thought heads:** 5-10 parallel transformer layers (~30-50M each), each generating one hypothesis
- **Judger/Pruner:** Calibrated scoring layer (like Laya's RLCD), trained with strict proper scoring rules
- **Training loop:** Karpathy's autoresearch pattern (propose → judge → keep better, discard worse)

**On a 128GB M4 Pro:**
- Encoder: ~1.6GB (4-bit)
- 10 thought heads × ~40M: ~1.6GB (4-bit)
- Judger: ~50M: ~0.2GB (4-bit)
- **Total model: ~3.4GB** (fits in ~5GB at 8-bit, ~3GB at 4-bit)
- Remaining ~123GB: context, working memory, training buffers, datasets

**Training requirements:**
- Validation dataset with known reasoning traces (GSM8K-style for math, but general-domain)
- RLCD (reinforcement learning from calibrated decisions) or DPO for hypothesis selection
- Estimated: 2-4 weeks of training on M4 Pro for a v1 prototype

**Advantages:**
- True System 2 architecture — parallel hypotheses, calibrated uncertainty, pruning
- No autoregressive token generation — single forward pass for all hypotheses
- Fully transparent, no hidden RLHF alignment layers
- Runs locally, no external dependencies

**Disadvantages:**
- No pre-trained foundation — must train encoder + heads from scratch
- Training data for reasoning traces is limited and noisy
- No text generation — outputs are hypotheses, not prose (unless we add a generation head)
- Unproven at scale — no one has done this

---

### Path B: Augment an Autoregressive Model

Take an existing autoregressive model (Qwen3.8, Mistral, Llama) and add a parallel reasoning layer on top. The base model does normal generation; a secondary system adds parallel hypothesis testing.

**Stack:**
- **Base model:** Qwen3.8-27B (4-bit, ~17GB) via Splash or MTPLX runtime
- **Added layer:** 3-5 parallel "thought heads" (~200M total) that query the base model in parallel
- **Judger:** Lightweight scorer (~50M) that evaluates hypothesis quality
- **Interface:** The system wraps an existing model — when a query comes in, it:
  1. Sends the query to the base model (normal generation)
  2. Simultaneously spawns 3-5 parallel "thought" prompts to the base model
  3. Scores each thought using the judger
  4. Keeps the best hypothesis, discards weak ones
  5. Returns the best hypothesis (with calibrated uncertainty)

**On a 128GB M4 Pro:**
- Base model (Qwen3.8-27B, 4-bit): ~17GB
- Thought heads (inference wrappers): ~0.5GB
- Judger: ~0.2GB
- **Total: ~17.7GB** (within 48GB, comfortable on 128GB)
- **Note:** Splash can run Qwen3.8-27B at 74 tok/s on M5 Pro; your M4 Pro will be comparable

**Training requirements:**
- Fine-tune only the thought heads and judger (base model stays frozen)
- Training data: queries + verified good answers (reasoning traces)
- Estimated: 3-5 days of fine-tuning on M4 Pro

**Advantages:**
- Leverages existing model's general language knowledge
- Faster to prototype — just add the reasoning layer
- Can use Splash or MTPLX for inference (both are fast on Apple Silicon)
- Produces normal text output (base model's strength)
- Smaller training burden — only heads + judger need training

**Disadvantages:**
- Not a true System 2 architecture — still generates autoregressively under the hood
- Overhead of running the base model for each hypothesis
- The base model's RLHF alignment layer still exists (Selsam's concern)
- Less efficient than a purpose-built model

---

## 3. Architecture Comparison

| | **Path A: Build Our Own** | **Path B: Augment Existing** |
|---|---|---|
| **Architecture** | Encoder + parallel heads + judger (single pass) | Base AR model + parallel query layer + judger |
| **Model size** | ~3.4GB (4-bit) | ~17.7GB (Qwen3.8 base + overhead) |
| **Inference speed** | ~33ms for decisions (like Laya) | ~200-500ms (base model + parallel queries) |
| **Training cost** | 2-4 weeks (full model) | 3-5 days (heads + judger only) |
| **Output type** | Hypotheses + probabilities (decisions) | Normal text (via base model) |
| **True System 2?** | Yes — parallel, pruned, single pass | Partial — parallel queries but AR under the hood |
| **Hardware** | Fits anywhere (even 16GB laptops) | Needs ~48GB RAM minimum |
| **Risk** | Unproven, no precedent | Proven base model, unproven reasoning layer |

---

## 4. Detailed Design: Path A (Build Our Own)

### 4.1 Encoder Layer

**Model:** ModernBERT-large (395M params)
- Bidirectional transformer, fully fine-tuned
- Input: text (up to 1024 tokens, extensible to 8192)
- Output: contextual representation (hidden states)

**Reference:** "ModernBERT" — modern optimizations to BERT architecture (2024)

**Why ModernBERT:**
- Fast (single forward pass)
- Bidirectional (understands context both ways)
- Proven on decision tasks
- Small enough to fine-tune on a laptop

### 4.2 Thought Heads (5-10 parallel)

Each thought head is a small transformer (~30-50M params) that:
1. Takes the encoder's output as input
2. Generates one "hypothesis" (a reasoning path, answer, or interpretation)
3. Produces a confidence score (calibrated via RLCD)

**Design:**
```
Encoder output (hidden states)
  └──> Thought Head 1 → Hypothesis A + Confidence 0.85
       └──> Thought Head 2 → Hypothesis B + Confidence 0.62
       └──> Thought Head 3 → Hypothesis C + Confidence 0.43
       └──> ... (5-10 heads total)
```

**Each head produces:**
- A hypothesis (text, structured data, or decision)
- A calibrated confidence score (0-1, honest via RLCD training)

**Reference:** Laya's decision head (2 transformer layers + option-marker scorer)

### 4.3 Judger/Pruner Layer

Takes all (hypothesis, confidence) pairs and:
1. Scores each hypothesis using a calibrated scoring function
2. Drops hypotheses below a confidence threshold (e.g., <0.3)
3. Returns the best remaining hypothesis, or None if all are weak

**Training:** RLCD (Reinforcement Learning from Calibrated Decisions)
- Reward = negative Brier score (strict proper scoring rule)
- Updates via REINFORCE with group-mean baseline (GRPO-style)
- Honest probability reporting maximizes expected reward

**Reference:** Laya's RLCD training (strict proper scoring rules)

### 4.4 Training Pipeline

**Phase 1: Encoder + Heads Pre-training**
- Pre-train the encoder on a general corpus (e.g., CommonCrawl subset)
- Initialize thought heads with random weights

**Phase 2: Head Fine-tuning**
- Use a dataset of queries + verified good answers
- Each head tries to produce the correct hypothesis
- Train with RLCD to maximize calibrated accuracy

**Phase 3: Judger Training**
- Train the judger to correctly score hypotheses (identify the best one)
- Use RLCD on a validation set

**Phase 4: End-to-End Fine-tuning**
- Freeze the encoder, fine-tune heads + judger together
- Use autoresearch loop: propose head configurations, judge on validation set, keep better

**Reference:** Karpathy's autoresearch loop (propose → judge → commit/revert)

---

## 5. Detailed Design: Path B (Augment Existing)

### 5.1 Base Model

**Model:** Qwen3.8-27B (4-bit, ~17GB)
- Served via Splash or MTPLX runtime
- Normal autoregressive generation (text output)
- Context window: up to 256K (Splash)

**Why Qwen3.8:**
- Fast on Apple Silicon (Splash: 74 tok/s, MTPLX: ~150 tok/s)
- 27B parameters at 4-bit = ~17GB RAM
- General language knowledge (doesn't need to be retrained)
- Well-documented, actively maintained

### 5.2 Parallel Query Layer

Instead of generating text sequentially, the system:
1. Receives a user query
2. Spawns 3-5 parallel "thought" prompts to the base model
3. Each thought prompt is a different framing/interpretation of the query
4. Waits for all completions
5. Scores each completion

**Example:**
```
User: "Is this customer service ticket about billing?"

Normal model: "Yes, this is about billing." (single pass)

System2 model:
  Head 1: "Yes, the user mentions being charged twice." (conf: 0.87)
  Head 2: "Could be billing or could be a complaint about service." (conf: 0.42)
  Head 3: "Unclear — more context needed about account status." (conf: 0.31)
  
Judger: Selects Head 1 (confidence 0.87, well above threshold 0.3)
Returns: "Yes, this is about billing. (confidence: 87%)"
```

### 5.3 Judger Layer

Same as Path A: a small calibrated scoring model (~50M params) that:
- Takes all (hypothesis, base_model_output) pairs
- Scores each one for quality and relevance
- Returns the best hypothesis with calibrated uncertainty

**Training:** Same RLCD training as Path A, but applied to the judger + query wrappers.

### 5.4 Training Pipeline

**Phase 1: Base Model Setup**
- Install Splash or MTPLX
- Serve Qwen3.8-27B locally
- Test normal inference

**Phase 2: Query Wrapper Development**
- Develop the parallel query interface (send N prompts, receive N completions)
- Test with different prompt framings

**Phase 3: Judger Training**
- Build a dataset of queries + verified good answers
- Train the judger to select the best hypothesis from a set
- Use autoresearch loop to optimize head configurations

**Phase 4: End-to-End Testing**
- Test the full system on a held-out test set
- Measure accuracy, speed, and calibration quality

---

## 6. Training Data

Both paths need training data. The hardest part of this project is not architecture — it's data.

### 6.1 Available Datasets

**Math reasoning:**
- **GSM8K** — grade-school math word problems with solutions
- **MATH** — competition-level math problems
- **GSM-Hard** — harder variants

**General reasoning:**
- **CoT (Chain-of-Thought) datasets** — problems with step-by-step solutions
- **AI2 ARC** — grade-school science questions
- **Grove** — reasoning benchmarks

**Decision/Classification:**
- **AG News** — text classification (4 labels)
- **Banking77** — customer service intent classification (77 labels)
- **Laya's typed-decisions dataset** — invoice processing, security incidents, customer service

**Reasoning traces (critical for System 2):**
- No large, clean datasets exist for general reasoning traces
- This is the gap — most datasets have final answers, not reasoning paths
- May need to synthetically generate some via existing models

### 6.2 Synthetic Data Generation

We could use an existing model (Qwen3.8) to generate training data:
1. Feed queries to Qwen3.8 with chain-of-thought prompting
2. Extract reasoning traces
3. Use as training data for our heads/judger
4. Caveat: synthetic data carries the base model's biases

**Reference:** "Self-Alignment" papers — using LLMs to generate training data for themselves

---

## 7. Implementation Details

### 7.1 Frameworks

**MLX (Apple Silicon):**
- Apple's ML framework for running models on M-series chips
- Splash and MTPLX both use MLX under the hood
- Native Metal acceleration, no GPU drivers needed

**Reference:** "MLX: Machine Learning on Apple Silicon" (Apple, 2024)

**Splash runtime:**
- `brew install incoai/tap/splash`
- Serves OpenAI/Anthropic-compatible API
- Speculative decoding with per-model draft models

**MTPLX runtime:**
- `brew install youssofal/mtplx/mtplx`
- Native multi-token prediction (uses the model's own MTP heads)
- 1.6-2.2× speedup on Apple Silicon

**Laya (for decision models):**
- `pip install laya`
- Non-autoregressive decision model (~33ms per call)
- Trained with RLCD (strict proper scoring rules)

### 7.2 Repository Structure

```
system2/
├── 01-system-architecture.md    # This document
├── 02-training-plan.md          # Detailed training pipeline
├── 03-implementation-spec.md    # Code-level specifications
├── src/
│   ├── encoder/                 # ModernBERT encoder wrapper
│   ├── heads/                   # Parallel thought heads
│   ├── judger/                  # Calibrated scoring layer
│   ├── runner.py                # Main inference loop
│   └── train.py                 # Training script (autoresearch loop)
├── data/
│   ├── train/                   # Training datasets
│   └── eval/                    # Evaluation datasets
├── configs/
│   ├── head_config.json         # Thought head configurations
│   └── eval_config.json         # Evaluation settings
└── experiments/                 # Logged training runs
```

### 7.3 Reference Implementations

**Laya (decision model):**
- https://github.com/NandhaKishorM/laya
- Non-autoregressive, single forward pass, ~33ms
- Trained with RLCD (strict proper scoring rules)
- 421M parameters (ModernBERT-large + decision head)

**Splash (inference runtime):**
- https://github.com/incoai/splash
- Apple Silicon optimization, speculative decoding
- OpenAI/Anthropic-compatible API

**MTPLX (inference runtime):**
- https://github.com/youssofal/MTPLX
- Native multi-token prediction
- 1.6-2.2× speedup on Apple Silicon

**Karpathy's autoresearch (training pattern):**
- https://github.com/karpathy/autoresearch (conceptual)
- Your implementation: https://github.com/tj-doomer-it/test-training

---

## 8. Risks and Caveats

### 8.1 Technical Risks

1. **No precedent:** No one has built a System 2 model at this scale. We're inventing this.
2. **Training data scarcity:** Clean reasoning traces are rare. We may need to generate synthetic data.
3. **Evaluation difficulty:** How do we know the model is "better"? Standard benchmarks measure autoregressive accuracy, not System 2 reasoning quality.
4. **Hardware limitations:** Even on 128GB, training a 3-5B parameter model will be slow (weeks, not hours).

### 8.2 Political Risks (Selsam's Argument)

1. **Compute access:** The people with large-scale compute are incentivized to grow autoregressive models, not build alternatives.
2. **Open-source funding:** No one is funding System 2 research at scale (yet).
3. **Benchmark gaming:** Even if we build this, we may not have a way to evaluate it against standard benchmarks.

### 8.3 Timeline Risks

1. **Path A (build our own):** 2-3 months for v1 prototype, 6-12 months for v2 with reasoning traces.
2. **Path B (augment existing):** 2-4 weeks for v1 prototype, 2-3 months for v2.

---

## 9. Recommendations

**Start with Path B (augment existing).** Reasons:

1. **Faster to prototype** — you can test the core idea (parallel hypotheses + pruning) in days, not weeks
2. **Lower risk** — if the idea fails, you've lost weeks, not months
3. **Proven base** — Qwen3.8 has real language knowledge; you only need to add the reasoning layer
4. **Learn by doing** — building Path B will teach you what a real System 2 model needs, making Path A easier later

**After validating the idea with Path B, iterate toward Path A.** Once you know what works (how many heads, what scoring function, what training data), you can build a purpose-built model from scratch.

**Concrete next steps:**
1. Install Splash or MTPLX
2. Serve Qwen3.8-27B locally
3. Build a simple parallel query wrapper (3 thoughts per query)
4. Build a simple judger (calibrated scoring)
5. Test on a small dataset
6. Iterate

---

## 10. Glossary

**System 1:** Fast, intuitive, parallel processing (Kahneman). Current LLMs operate in System 1 mode.

**System 2:** Slow, deliberate, hypothesis-driven processing (Kahneman). This project aims to build a System 2 model.

**RLCD:** Reinforcement Learning from Calibrated Decisions. Training method where the reward is based on strictly proper scoring rules (honest probability reporting maximizes reward).

**Speculative decoding:** A technique where a small "draft" model generates multiple tokens, which are then verified by a larger model in a single forward pass.

**MTP (Multi-Token Prediction):** A technique where models have built-in heads that predict multiple tokens ahead, enabling faster decoding without external draft models.

**Autoresearch:** A training pattern (Karpathy) where an agent proposes changes to a system, a judge evaluates them, and better changes are committed while worse ones are reverted.

**Calibrated probability:** A probability estimate that is "honest" — if the model says "70% confidence," it is correct 70% of the time.

**Hypothesis:** A proposed reasoning path, answer, or interpretation. In System 2, multiple hypotheses are generated in parallel, scored, and pruned.

---

## Sources

- Kahneman, D. (2011). *Thinking, Fast and Slow*. Farrar, Straus and Giroux.
- Selsam, D. (2026). "Personal Statement on AI Risk." OpenAI. https://explainx.ai/blog/dan-selsam-openai-eval-awareness-ai-risk-statement-2026
- "ModernBERT: Modernized BERT Architecture." Hugging Face.
- "Laya: Multilingual, Non-Autoregressive System 1 Decision Model." ConvAI Innovations. https://huggingface.co/convaiinnovations/laya
- "Splash: Local Inference Engine for Apple Silicon." inco.ai. https://github.com/incoai/splash
- "MTPLX: Native MTP Speculative Decoding on Apple Silicon." youssofal. https://github.com/youssofal/MTPLX
- Karpathy, A. (2024). "autoresearch" — autonomous research loop pattern. https://github.com/tj-doomer-it/test-training
