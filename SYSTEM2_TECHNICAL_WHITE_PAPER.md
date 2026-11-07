# System 2: Non-Autoregressive Epistemic Deliberation & Consensus Harness for Edge AI

**Technical White Paper**
**Authors:** Thom James-Worrall / Antigravity Research
**License of reference implementation:** Apache 2.0
**Live assets:**
- Hugging Face model: https://huggingface.co/tjdoomer/system2-code
- GitHub repository: https://github.com/tj-doomer-it/System2
- Website: https://system2.online

**Evaluation hardware (all benchmarks):** Apple MacBook Pro (Mac16,6), Apple M4 Max, 128 GB Unified Memory, LM Studio / `llama.cpp` and Inco AI Splash runtimes, System 2 on PyTorch MPS.

---

## 1. Executive Summary & Lead Manifesto

> "Small local models have a 15% to 25% stochastic error rate, whether from CoT loops, hallucinated syntax, or formatting quirks. If you run them raw, they break in production. System 2's job is not to magically make a broken model write code; it is to catch those failures in 500ms on Apple Silicon MPS and say: **Certainty is 41.8%, do not run this code.** That is the automated gatekeeper that makes edge AI safe."

Autonomous coding agents increasingly execute what a model emits: they write files, run tests, and issue terminal commands. A single language model reports the same fluent confidence for a correct implementation and for a broken one. Reliability on local hardware is therefore not primarily a generation problem. It is a calibration and arbitration problem.

System 2 is a non-autoregressive **epistemic filter**, not a text generator. A fast local generator produces four independent candidate hypotheses. A 143 MB checkpoint (`system2_code.pt`) on a frozen ModernBERT-Large-Instruct backbone then evaluates all candidates in a single feedforward pass, with five parallel ThoughtHeads, a cross-attending Judger and a calibrated confidence score. High agreement means local execution. Low agreement means a fast fail and optional escalation, before any code runs.

### 1.1 Principal Findings

1. **Quantization collapse is real and measurable.** On an M4 Max, 2-bit (`UD-IQ2_XXS`, 6.77 GB) and 3-bit (`UD-IQ3_XXS`, 10.18 GB) quantizations of a 27B model fell into cyclic self-doubt loops inside `<think>` blocks. The loops consumed up to 1,477 of 1,500 reasoning tokens in the controlled evaluation, and all four candidates hit a 2,500-token ceiling in the swarm benchmark.
2. **System 2 is an evaluator, not a repair model.** It cannot judge code that was never produced (the Garbage In, Garbage Out law). Its value is detection and routing.
3. **Live drift detection works.** In the unconstrained-thinking Gemma run (Stage 2.3), candidate divergence inside extended scratchpads collapsed System 2 confidence from 98.7% to **41.8%**, well below the 88% certainty threshold, flagging epistemic drift before execution.
4. **End-to-end multi-agent verification succeeds when paired with direct completion (`reasoning_effort="none"`).** Suppressing extended `<think>` monologue on Google Gemma 4 12B (4-bit MLX, 6.77 GB), forwarding upstream module interfaces across stages, and normalizing markdown fence indentation reduced total 3-stage swarm time from 18.5 minutes to **2.1 minutes** (127.5s), achieved **99.5% to 99.9% consensus (5/5 heads)** across all three modules, and passed **4/4 automated pytest tests in 0.22s**.
5. **The code-track checkpoint is calibrated.** The guarded metric is 0.9010 (89.11% accuracy, 0.0891 Brier score, 0.0542 10-bin ECE) on held-out code traces.

### 1.2 Economic Paradigm

System 2 enables a third deployment option. Instead of paying for cloud frontier subscriptions or buying $4,000 to $7,500 workstations for 70B+ models, a team runs a 4-bit 7B to 14B generator plus the 143 MB harness on a standard 16 GB consumer laptop ($1,099). For a 10-developer team over three years (figures from the business plan):

| Approach | 3-year total cost |
| :--- | ---: |
| Cloud frontier subscriptions (10 seats at $200/month, plus overage) | $87,000 to $117,000 |
| Heavyweight workstations (10 x M4 Max 128 GB at $4,699) | $46,990 upfront capex |
| **System 2 edge stack (10 x 16 GB laptops at $1,099, plus ~$1,200/year escalation)** | **$14,590** |

Against the $87,000 cloud baseline, the System 2 stack saves about **$72,000+ (83%)**, with source code staying on-device. The estimate assumes 80% to 85% of tasks execute locally and only the 15% to 20% that drift escalate to a paid frontier endpoint.

---

## 2. The Mechanics of the 2-Bit / 3-Bit Quantization Collapse

### 2.1 Experimental Setting

The evaluation compared four quantizations of Qwen 3.8 27B on Apple Silicon (M4 Max, 128 GB Unified Memory). Table 1 reproduces the measured results from the Hugging Face evaluation report.

**Table 1. Quantization spectrum (Qwen 3.8 27B).**

| Benchmark / Metric | `UD-IQ2_XXS` (2.0-bit, 6.77 GB) | `UD-Q2_K_XL` (2.5-bit, 9.15 GB) | `UD-IQ3_XXS` (3.0-bit, 10.18 GB) | `Splash` (4.0-bit, 17.38 GB) |
| :--- | :--- | :--- | :--- | :--- |
| VRAM footprint | 6.77 GB (-61%) | 9.15 GB (-47%) | 10.18 GB (-41%) | 17.38 GB (baseline) |
| Test 1: OS concepts | 321 toks (16.3s, 19.7 t/s) | 313 toks (58.9s, 5.3 t/s) | 142 toks (5.3s, 26.9 t/s) | 446 toks (7.3s, 61.2 t/s) |
| Test 2: code debugging | FAILED (1,477 reasoning toks) | FAILED (1,473 reasoning toks) | PASSED (664 reasoning toks, 27.2s) | PASSED (420 reasoning toks, 17.0s) |
| Test 2 AST execution | 0 output tokens | 0 output tokens | PASSED (`[0, 2, 4]`) | PASSED (`[0, 2, 4]`) |
| Test 3: constraints | FAILED (1,461 reasoning toks) | FAILED (1,474 reasoning toks) | FAILED (1,472 reasoning toks) | PASSED (0 violations, 24.8s) |
| Total suite latency | 318.4s (5.3 min) | 485.3s (8.1 min) | 97.4s (1.6 min) | 49.1s |

### 2.2 Failure Mode: Loss of Termination Confidence

On simple prompts, `UD-IQ2_XXS` produces valid text. On reasoning-heavy prompts, quantization noise degrades the sharpness of the next-token distribution. Our working hypothesis, supported by the observed behavior and not by direct weight surgery, is the following:

1. The projection weights governing internal self-critique lose dynamic range at 2 to 3 bits.
2. The model therefore never reaches the probability threshold needed to emit the `</think>` stop token.
3. It re-evaluates the same sub-problem repeatedly. In Test 2 (the Python closure late-binding gotcha), the 2-bit model used **1,477 of 1,500 tokens** re-deriving the same comprehension mechanics, then hit `max_tokens` with an empty user-facing answer.

This is a cyclic self-doubt loop: high token spend, zero emitted code lines.

### 2.3 Contrast with 4-Bit Dense Execution

At 4-bit precision the attention weights and output distributions remain intact, and the loop disappears. The 4-bit Splash model solved the same Test 2 with 420 reasoning tokens in 17.0s and passed Test 3, which every lower-precision variant failed. Google Gemma 4 12B (4-bit MLX, 6.77 GB) shows the same health in the swarm benchmark. It occupies **the same 6.77 GB footprint as the 2-bit Qwen**, yet emits complete, diverse candidates in 2 to 4 seconds without runaway internal loops.

The practical rule is the **Golden Pairing Rule**: pair System 2 with generators whose quantization preserves output fluency, such as 4-bit 7B to 14B models, or 3-bit 27B models with thinking suppressed.

### 2.4 Time To Code Execution (TTCE)

For interactive agents, latency determines whether the tool is used. We define TTCE as wall-clock time from prompt to a code candidate that is safe to execute.

- **Anti-pattern (crushed 27B models).** Four parallel streams on consumer silicon drop throughput to roughly 4 tok/s per stream. With 1,500 to 3,000 tokens spent in `<think>` loops, the benchmarks measured **8 to 9 minutes per stage** (Path B: 480s per stage, 26.8 minutes total).
- **Optimal pattern (healthy 4-bit small models).** A 12B model at 50 to 70 tok/s finishes four candidates of 200 to 300 lines in 2 to 3 seconds. System 2 adds roughly 30 ms to 400 ms of deliberation on MPS in the business-plan measurements, and 580 ms to 1,107 ms average in the swarm runs (Table 3). This yields a **sub-4-second TTCE target**.
- **Fail-fast property.** If drift is detected, the pipeline fails in the same window and escalates, rather than wandering through an 8-minute hallucination loop.

> **Empirical validation of non-thinking TTCE.** In the initial Gemma swarm run, LM Studio defaulted `google/gemma-4-12b` to unconstrained `<think>` mode, causing stage generation to take 430s per stage (18.5 minutes total). When re-evaluated with `"reasoning_effort": "none"` (non-thinking direct completion), four parallel candidate streams finished Stage 2.1 in **12.7s to 13.9s** (352 to 401 tokens), Stage 2.2 in **48.4s to 55.7s** (1,173 to 1,482 tokens), and Stage 2.3 in **39.2s to 57.9s** (687 to 1,103 tokens), cutting total 3-stage swarm time from 18.5 minutes down to **2.1 minutes** (127.5s) and passing all 4 pytest tests.

---

## 3. System 2 Architectural Mechanics

### 3.1 Design Principles

1. **Strictly non-autoregressive.** One feedforward pass; no token-by-token decoding loop anywhere in the decision path.
2. **Frozen backbone.** ModernBERT-Large-Instruct (395M) weights remain frozen. Only adapters, heads and the Judger train.
3. **In-domain causal purity.** Checkpoints are domain-specific. External trivia datasets are not mixed into the shared adapter, because multi-domain mixing warps representations toward surface syntax.

### 3.2 Backbone and LoRA Subspace Adapters

Let the frozen encoder have 28 layers, indexed 0 to 27. Layers 0 to 19 are untouched (about 280M parameters). Layers 20 to 27 receive low-rank updates. For a frozen weight matrix $W_0 \in \mathbb{R}^{d \times k}$ in an adapted layer:

$$W = W_0 + \frac{\alpha}{r} B A, \qquad A \in \mathbb{R}^{r \times k}, \; B \in \mathbb{R}^{d \times r}, \; r = 32, \; \alpha = 32$$

so the scaling factor is $\alpha / r = 1$. This adds about 1.57M trainable parameters. Restricting adaptation to the top 8 layers is an empirical optimum. Deeper adaptation distorts lexical features, and higher rank without scheduled decay induces late-epoch overfitting.

**Table 2. Parameter budget (Scientific Specification v2.0).**

| Component | Layers | Parameters | Trainable |
| :--- | :---: | :---: | :---: |
| Encoder lexical layers | 0 to 19 | ~280M | No |
| Encoder LoRA subspace | 20 to 27 | ~1.57M | Yes |
| ThoughtHead encoders (5 heads, 2 layers each, d=512) | n/a | ~33.6M | Yes |
| Contrast MLPs (1024 to 256 to 1) | n/a | ~1.3M | Yes |
| Head calibrators (512 to 64 to 1) | n/a | ~165K | Yes |
| Judger cross-attention (8 heads) | 1 | ~1.05M | Yes |
| Judger decision scorer | n/a | ~33K | Yes |
| **Total** | n/a | **~432M** | **~37.7M (8.7%)** |

The distributed checkpoint `system2_code.pt` is **143 MB**. It contains the trained components (LoRA adapter, five heads and Judger). The 395M frozen backbone is loaded separately from the base ModernBERT release. The engine keeps one resident backbone and hot-swaps the small domain adapters in under 3 ms.

### 3.3 Candidate Encoding and the Five ThoughtHeads

Each candidate is serialized as `Question: {q} Answer: {c_i}` and encoded by the backbone. The implementation locates the `Answer:` boundary and builds two pooled vectors: $q$ (mean over question tokens) and $a$ (mean over answer tokens), plus a holistic mean $h$ over the full sequence. Five asymmetric span perspectives are then formed (`encoding.py`, default mode; `system2_runtime.py`):

| Head | Name | Span formulation | Role |
| :---: | :--- | :--- | :--- |
| 0 | Holistic Context | $v_0 = h$ | Global bidirectional premise-choice integration |
| 1 | Context-Anchored | $v_1 = 0.40\,q + 0.60\,a$ (40% Q / 60% A) | Question-weighted premise validation |
| 2 | Choice-Forward | $v_2 = 0.20\,q + 0.80\,a$ (20% Q / 80% A) | Direct consequence tracing |
| 3 | Symmetric Balance | $v_3 = 0.50\,q + 0.50\,a$ (50% Q / 50% A) | Equi-partitioned premise-action reasoning |
| 4 | Extreme Salience | $v_4 = 0.05\,q + 0.95\,a$ (5% Q / 95% A) | Fine-grained output boundary verification |

> **Note on specification drift.** The prose in `docs/SCIENTIFIC_SPECIFICATION.md` section 2.1 uses a different naming for heads 1 to 4. This paper follows the implemented weights in the code and the model card.

Each ThoughtHead projects the 1024-dimensional vector to 512 dimensions, applies a **2-layer** transformer encoder (8 attention heads, feedforward width 2048, GELU), and emits both candidate logits and a branch confidence. Two layers is a deliberate constraint. For a candidate sequence of length $L = 4$, deeper stacks create optimization bottlenecks and fail to train from scratch.

### 3.4 Arithmetic Centroid Contrast Invariance

For head $h$ and candidate $i \in \{0,1,2,3\}$, with post-attention features $x_i$:

$$\bar{x} = \frac{1}{4}\sum_{j=0}^{3} x_j, \qquad d_i = x_i - \bar{x}, \qquad v_i = [\,x_i \,\|\, d_i\,] \in \mathbb{R}^{1024}$$

$$\ell_{h,i} = W_2\,\mathrm{GELU}(W_1 v_i + b_1) + b_2$$

The reference centroid is the fixed arithmetic mean. A dynamic attention-based soft centroid is prohibited because it collapses contrast toward zero on salient options, exactly when contrast is most needed.

### 3.5 The Scale-Invariant Logit Law

Because the heads see asymmetric spans, their raw logits develop large variance disparities (observed standard deviations of 1.72 for one head against 0.00048 for another). Unnormalized summation lets high-variance heads drown the rest. Arbitration therefore standardizes per head before pooling:

$$z_{b,h,c} = \frac{\ell_{b,h,c} - \mu_{b,h}}{\sigma_{b,h} + \epsilon}, \qquad S_{b,c} = \sum_{h=0}^{4} w_h\, z_{b,h,c}$$

where $\mu_{b,h}$ and $\sigma_{b,h}$ are the mean and standard deviation over the candidate axis, and $w_h$ are normalized head or Judger weights. Every perspective contributes in equal statistical currency (standard-deviation units).

### 3.6 Judger and Calibrated Confidence

The Judger applies multi-head cross-attention over the five head embeddings $E \in \mathbb{R}^{B \times 5 \times 512}$ with residual pre-normalization:

$$\tilde{E} = E + \mathrm{MHA}(\mathrm{LN}(E)), \qquad s = W_{s2}\,\mathrm{GELU}(W_{s1}\tilde{E}), \qquad h^{*} = \arg\max_h s_h$$

The decision confidence is $\mathrm{conf}^{*} = \sigma\big((s_{h^*} - \bar{s}) / \tau\big)$ with learnable $\tau = \exp(\theta_\tau)$. The decision-level calibration loss $\mathcal{L}_{\text{judge\_cal}}$ trains this value directly against whether the selected candidate is correct, and its gradient is never detached or softened. A candidate is pruned when its branch score falls below 0.30.

Two training invariants preserve head diversity and calibration quality:

- **Cognitive Autonomy Invariant:** $\lambda_{\text{ens\_ce}} = 0$. Ensemble cross-entropy is never backpropagated into the ThoughtHeads, since a shared task loss collapses their asymmetric representations toward consensus.
- **Cosine annealing:** all training decays the learning rate to $1\times10^{-5}$ with 5% linear warmup. Optimization uses AdamW. Learning rates are $3\times10^{-4}$ (LoRA), $5\times10^{-4}$ (heads and Judger) and $1\times10^{-2}$ (calibration scale).

The composite objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_{\text{cal}}\mathcal{L}_{\text{cal}} + \lambda_{\text{judge}}\mathcal{L}_{\text{judge}} + \lambda_{\text{judge\_cal}}\mathcal{L}_{\text{judge\_cal}} + \lambda_{\text{div}}\mathcal{L}_{\text{div}} + \lambda_{\text{cand\_margin}}\mathcal{L}_{\text{cand\_margin}} + \lambda_{\text{balance}}\mathcal{L}_{\text{balance}}$$

### 3.7 Pairwise Cosine Cluster Consensus and Epistemic Drift

Beyond the head vote, System 2 reasons about the agreement of the four candidates themselves. Let $e_i$ be the pooled embedding of candidate $i$. The pairwise cosine matrix is:

$$C_{ij} = \frac{e_i \cdot e_j}{\|e_i\|\,\|e_j\|}$$

Candidates form a consensus cluster when pairwise similarity is at or above the **0.88** threshold, together with 4/5 or 5/5 head agreement. The documented states are:

| State | Condition | Action |
| :--- | :--- | :--- |
| **High certainty (factual consensus)** | Cluster agreement at or above 0.88, with 4/5 or 5/5 head agreement | Execute locally, zero API cost |
| **Epistemic drift** | Divergent candidates, low similarity, confidence below the 88% certainty threshold | Halt, fast-fail, escalate or request human review |

For reference, the paired benchmark in the Hugging Face report recorded a healthy case: 4/4 candidates aligned at 92.2% pairwise cosine similarity, 93.1% decision confidence, 5/5 head consensus and 497.68 ms deliberation latency.

> **Implementation note.** The shipped runtime classifies results with a three-tier rule over decision confidence and head agreement: *verified* at confidence of 75% or more with at least 60% head agreement; *uncertain* at 50% or more; and *drift* ("hallucination warning") below 50%. The 0.88 cosine-cluster threshold is the documented product-level contract. We did not find it hard-coded in the open Python runtime. Deployments should treat 41.8% as unambiguous drift under either rule.

The gatekeeper module (`sys2.guard`) applies the same principle to terminal commands, with thresholds above 0.70 for allow or block and below 0.40 for abstain. It maps to the Destructive Tool Gating use case in Section 5.

---

## 4. Empirical Multi-Agent Swarm Benchmark

### 4.1 Protocol

A Lead Architect (`qwen3.8-27b-splash`, 17.38 GB) decomposes a modular thread-safe `LRUTTLCache` into independent modules. For each module, four parallel zero-knowledge candidates are sampled from the path's generator, deliberated by System 2 (ModernBERT on MPS), saved to disk and verified with `pytest`. The Gemma and Path B runs used three modules (`cache_node.py`, `lru_ttl_cache.py`, `test_lru_ttl_cache.py`). Path A decomposed into four micro-files in the evaluation report and is tabulated on the three common modules.

### 4.2 Head-to-Head Results

**Table 3. Path A vs Path B vs Path Gemma (M4 Max, 128 GB).**

| Metric | Path A: `qwen-3.8-27b-smoll` | Path B: `qwen-3.8-27b-iq3xxs` | Path Gemma (Unconstrained `<think>`) | Path Gemma (Remediated: `reasoning_effort="none"`) |
| :--- | :--- | :--- | :--- | :--- |
| Model size / VRAM | 2.0-bit GGUF (6.77 GB) | 3.0-bit GGUF (10.18 GB) | 4.0-bit MLX (6.77 GB) | 4.0-bit MLX (6.77 GB) |
| Architect decomposition time | 24.1s (4 files) | 73.97s (3 files) | 27.04s (3 files) | 14.75s (3 files) |
| Module 1 (`cache_node.py`) | 99.7% conf (5/5 heads, 41 lines) | 81.6% conf (5/5 heads, 19 lines) | 76.6% conf (5/5 heads, 22 lines) | **99.6% conf (5/5 heads, 44 lines)** |
| Module 2 (`lru_ttl_cache.py`) | 94.2% conf (5/5 heads, 18 lines) | 75.2% conf (5/5 heads, 14 lines) | 98.7% conf (5/5 heads, 37 lines) | **99.9% conf (5/5 heads, 143 lines)** |
| Module 3 (`test_lru_ttl_cache.py`) | 99.6% conf (5/5 heads, 31 lines) | 67.8% conf (5/5 heads, 10 lines) | **41.8% conf (EPISTEMIC DRIFT)** | **99.5% conf (5/5 heads, 85 lines)** |
| Avg deliberation latency | 580 ms | 982 ms | 1,107 ms | **891 ms** |
| Total generation latency | ~4.5 minutes | 26.8 minutes (480s per stage) | 18.5 minutes (430s per stage) | **~2.1 minutes (127.5s total)** |
| Pytest result | FAILED (SyntaxError in cache.py) | FAILED (IndentationError, line 9) | HALTED / FAILED (Drift + Truncation) | **PASSED (4/4 tests in 0.22s)** |
| Primary root cause / outcome | CoT loop trap in 2-bit quant | CoT loop trap in 3-bit quant | Extended `<think>` debate caught by S2 | **Direct completion + context forwarding + AST guard** |

### 4.3 Interpretation

**Confidence is not correctness when all candidates fail identically.** Path A reports 94% to 99.7% confidence with unanimous 5/5 head agreement, yet its pytest run failed. When four candidates are all truncated or all malformed in the same way, they *agree*, and a similarity-based consensus signal reads that agreement as certainty. This is the clearest empirical statement of the GIGO law: System 2 correctly selects the least-bad candidate in a cluster, but it cannot synthesize code that was never emitted, and it cannot detect a shared failure that all candidates reproduce. Consensus measures agreement among hypotheses, not execution validity. Production pipelines must pair System 2 with a deterministic syntax and test gate (`compile()` + `pytest`).

**Path B (3-bit) confirms the collapse is not unique to 2 bits.** All four candidates hit the 2,500-token ceiling inside `<think>`, emitting only 10 to 19 line fragments after 8 to 9 minutes per stage. Confidence fell progressively (81.6%, 75.2%, 67.8%), consistent with increasingly fragmentary candidates.

**Path Gemma (Unconstrained `<think>`) vs Path Gemma (Remediated `reasoning_effort="none"`).** Comparing the two Gemma columns isolates the exact engineering conditions required for reliable edge code generation:
1. **Unconstrained `<think>` run (Column 3):** When LM Studio ran `google/gemma-4-12b` in default extended-thinking mode without upstream module context, subagents spent hundreds of seconds debating edge cases inside `<think>` scratchpads and truncating at token limits. In Stage 2.3, candidate divergence collapsed System 2 confidence to **41.8%**, triggering an immediate **Epistemic Drift** alert before execution.
2. **Remediated run (Column 4):** Applying three harness fixes (setting `reasoning_effort="none"` to suppress extended thinking, passing verified upstream modules as context into subsequent stages, and normalizing markdown indentation with `clean_python_indentation` + a `compile()` syntax guard) transformed the outcome on the exact same 6.77 GB model. Generation time fell from 18.5 minutes to **2.1 minutes**, System 2 consensus reached **99.6%, 99.9%, and 99.5%** across all 5 ThoughtHeads, and the generated `LRUTTLCache` implementation passed **4/4 pytest tests (`test_basic_set_get`, `test_lru_eviction`, `test_ttl_expiration`, `test_thread_safety`) in 0.22s**.

### 4.4 Live Validation Event: Gemma Stage 2.3 (Unconstrained `<think>`)

In the unconstrained Stage 2.3 run (generating `test_lru_ttl_cache.py`), the four candidate subagents entered an internal debate over **relative versus absolute TTL refresh** semantics. Their outputs diverged sharply:

| Candidate | Emitted lines | Character |
| :---: | :---: | :--- |
| 1 | 0 | Empty output |
| 2 | 78 | Partial implementation |
| 3 | 108 | Scratchpad debate embedded in the output |
| 4 | 7 | Fragment |

System 2 evaluated the set in a single feedforward pass. Decision confidence across the five ThoughtHeads fell from **98.7%** (on the preceding cache module) to **41.8%**, far below the 88% certainty threshold and below the 50% drift line of the shipped runtime. The harness raised **Epistemic Drift** and halted before test execution.

When the harness was subsequently re-run with `reasoning_effort="none"` and upstream interface context (Section 4.3, Column 4), all four subagents produced complete 85-to-119-line pytest suites in 39.2s to 57.9s, System 2 locked in at **99.5% confidence (5/5 heads)**, and all 4 unit and concurrency tests passed on the first run.

---

## 5. Commercial & Open-Core Model

Developers and security teams will not place a closed-source binary between their IDE and their models. System 2 therefore follows an open-core strategy: the engine is open, and the governance layer is commercial.

### 5.1 Open Source Core (Apache 2.0)

| Component | Description |
| :--- | :--- |
| **Python SDK** (`pip install system2`) | `load_system2`, `mean_pool_pairwise`, five ThoughtHeads, pairwise cosine consensus and epistemic drift detection. Auto-detects `mps`, `cuda` or `cpu`. |
| **Flagship checkpoint** (`system2_code.pt`) | 143 MB code-track weights on Hugging Face: https://huggingface.co/tjdoomer/system2-code. Covers polyglot code, syntax validation, algorithm traces and unit-test gating. |
| **Drop-in OpenAI-compatible proxy** | FastAPI server on `localhost:8000` (`python3 -m sys2.server --port 8000`). Point Cursor, Windsurf, Claude Code, Cline, Continue.dev, LangChain, LangGraph or AutoGen at `http://localhost:8000/v1`. |
| **Developer CLI** | `sys2` (deliberation, routing) and `sys2-guard` (command safety gate). |

Code is the first open domain because it is where small local models fail most visibly, as the quantization benchmarks show. The model-card quickstart illustrates the API:

```python
from sys2 import System2Engine

engine = System2Engine(device="mps")
result = engine.adjudicate(
    question="Select the correct, thread-safe implementation of an LRU cache eviction node:",
    choices=[candidate_a_code, candidate_b_code, candidate_c_code, candidate_d_code],
    domain="code",
)
print(result.confidence, result.head_consensus, result.deliberation_latency_ms)
```

### 5.2 Commercial Enterprise Tier

| Offering | Value |
| :--- | :--- |
| **Enterprise Epistemic Gateway** | Centralized drift telemetry and dashboards (failure rates, drift frequency per repository, dollar savings against cloud APIs); **automated PII masking**; **cloud auto-escalation** of drifted requests to authorized endpoints (frontier APIs, or corporate Bedrock and Azure OpenAI VPCs); immutable audit logging for SOC 2 and ISO 27001 evidence. |
| **Air-gapped and defense licensing** | Turnkey offline packages for SCIFs, classified networks, defense contractors and national labs. Fixed annual site licenses of $50,000 to $250,000 per year. |
| **Vertical checkpoints** | Proprietary domain adapters (medical, legal, finance, automotive, electrical) under enterprise license. |
| **Custom fine-tuning factory** | Bespoke five-head models trained on a customer's private codebases, internal APIs and engineering standards. |

Pilot pricing in the release plan is $20 per developer per month for the Gateway and $50,000 per year for air-gapped site licenses.

### 5.3 Target Markets

1. **Defense, intelligence and air-gapped systems** that cannot stream code to third-party clouds.
2. **Privacy-constrained enterprise** (fintech, medtech, legal) under PCI-DSS, HIPAA or client confidentiality, where 80% to 85% of queries stay on-device.
3. **Edge and field engineering** (industrial sites, hangars, remote mining) with no reliable broadband.

---

## 6. Limitations and Future Work

1. **Evaluator, not generator.** System 2 cannot recover output that was never produced. It must be paired with generators that preserve fluency.
2. **Shared-failure blind spot.** Agreeing candidates that share a defect yield high confidence (Path A). A deterministic syntax or test gate should follow consensus.
3. **Single swarm run per path.** The head-to-head table is one run per path on one machine. It illustrates failure modes and does not provide statistical estimates. Repeated trials with variance are future work.
4. **Cost model assumptions.** The 83% saving depends on the assumed 15% to 20% escalation rate and the cited subscription prices.
5. **Domain variance.** Guarded metrics differ widely by track (for example 0.9881 for automotive against 0.6700 for legal in the repository README). The code track (0.9010) is the flagship, and other tracks should be adopted according to their own calibration.
6. **Threshold documentation.** The 0.88 cosine-cluster contract should be made explicit in the open runtime to match the documentation.

---

## 7. Conclusion

Small local models will keep making stochastic errors, and extreme quantization makes some of those errors systematic, as the cyclic self-doubt loops of 2-bit and 3-bit models show. System 2 does not try to make a broken generator correct. It makes failure visible: five asymmetric perspectives, z-score standardized and arbitrated by a calibrated Judger, turn candidate disagreement into a quantitative certainty score in roughly half a second to a second on Apple Silicon. When subagents drifted inside unconstrained `<think>` scratchpads, System 2 dropped confidence to 41.8% and halted before broken code ran. When paired with a healthy 4-bit generator (`google/gemma-4-12b`, 6.77 GB) using direct completion (`reasoning_effort="none"`), upstream module context, and AST syntax verification, the swarm reached 99.5% to 99.9% consensus across all 5 ThoughtHeads and passed 4/4 pytest tests in 0.22s. On a $1,099 laptop, this is the gatekeeper that makes edge AI safe to act on.

---

## Appendix A. Source Provenance

All empirical figures are taken from: `SYSTEM2_BUSINESS_PLAN.md`, `OPEN_SOURCE_RELEASE_PLAN.md`, `MODEL_CARD.md`, `huggingface_eval_report.md`, `docs/SCIENTIFIC_SPECIFICATION.md`, `01-system-architecture.md`, the repository README, and the implementations in `encoding.py`, `heads.py`, `system.py`, `sys2/engine.py`, `sys2/server.py`, `sys2/guard.py` and `Sys2/engine/system2_runtime.py`.

## Appendix B. Reproducibility

```bash
pip install system2
python3 -m sys2.server --port 8000          # OpenAI-compatible proxy
python3 -m sys2 --interactive               # deliberation REPL
python3 -m sys2.guard "rm -rf /"            # command gatekeeper
pytest tests/                               # ML regression suite
```
