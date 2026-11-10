# System 2: Next-Generation Decision Intelligence & Fast Perceptual Arbitration

> **Architectural Status**: Production Explainer & Website Overview Page  
> **Target Audience**: AI Engineers, Enterprise Architects, Autonomous Agent Developers  
> **Reference Architecture**: Non-Autoregressive Feedforward Deliberation & Dynamic Subspace Routing  

---

## What is a Decision (or System 2) Model?

Over the past two years, the AI ecosystem has been dominated by **Large Language Models (LLMs)**. While LLMs excel at open-ended creative generation and conversational fluency, they are fundamentally ill-suited for the mission-critical "hot path" of autonomous software agents and automated enterprise workflows:

- **Autoregressive Latency Tax**: Standard LLMs generate text token-by-token. Even simple programmatic decisions (e.g. *"Should this support ticket go to Billing or Technical?"* or *"Is this terminal command safe to execute?"*) require unrolling dozens of tokens across 2,000 to 5,000 milliseconds.
- **Probabilistic Hallucination & Drift**: Autoregressive decoding is inherently stochastic. Prompts that require deterministic boolean gates or bounded enum selections frequently suffer from formatting drift, markdown preamble leakage, or unexpected refusal.
- **Uncalibrated Overconfidence**: Generative LLMs output incorrect assertions with the exact same high probability logits as verified facts.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TRADITIONAL GENERATIVE LLM (Slow, Autoregressive, 2,000 – 5,000 ms)                    │
│ Prompt ──► [ Token 1 ] ──► [ Token 2 ] ──► [ Token 3 ] ... ──► "The answer is Billing" │
└────────────────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM 2 DECISION ENGINE (Fast, Non-Autoregressive Feedforward, 15 – 35 ms)            │
│ State  ──► [ Bidirectional Feature Extraction ] ──► 5 ThoughtHeads ──► Typed Probabilities│
│ Schema ──► [ Options Evaluated in Parallel    ] ──► Judger Logits   ──► Calibrated Output │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

A **Decision Model** (pioneered by architectures like Typesafe's Jev, Cloudflare's Clef, and System 2) fundamentally re-engineers this paradigm. Instead of generating tokens, a decision model derives **bounded, strictly typed, calibrated probabilities directly from internal latent representations in a single feedforward pass**.

### Parallel Structured Decision Execution
Given an arbitrary operational payload (e.g., an enterprise support ticket, an agent bash command, or an API trace), System 2 evaluates multiple orthogonal questions and schema criteria simultaneously in tens of milliseconds:

```
INPUTS                                                     PARALLEL OUTPUTS
┌────────────────────────────────────────────────┐        ┌───────────────────────────────┐
│ SUPPORT PAYLOAD (Enterprise Customer)          │        │ DEPARTMENT ROUTING            │
│ "Our API gateway started returning 500 errors  │        │   Billing:          0.2%      │
│ 20 minutes ago. We can't process checkout      │        │   Technical:       99.6%  ████ │
│ transactions. Please help ASAP."               │        │   Sales:            0.2%      │
└───────────────────────┬────────────────────────┘        └───────────────────────────────┘
                        │
                        ▼                                 ┌───────────────────────────────┐
┌────────────────────────────────────────────────┐        │ URGENCY CLASSIFICATION        │
│ SCHEMA CRITERIA                                │───────►│   Urgent (Yes):    99.8%  ████ │
│ • Which team? [Billing / Technical / Sales]    │        │   Routine (No):     0.2%      │
│ • Is it urgent? [Yes / No]                     │        └───────────────────────────────┘
│ • Incident Severity? [Minor / Major / Critical]│
└────────────────────────────────────────────────┘        ┌───────────────────────────────┐
                                                          │ INCIDENT SEVERITY             │
                                                          │   Minor:            0.1%      │
                                                          │   Major:            4.2%      │
                                                          │   Critical:        95.7%  ████ │
                                                          └───────────────────────────────┘
```

By decoupling *decision-making* from *token generation*, agents can programmatically gather context, classify intent, verify security boundaries, and execute deterministic actions without human latency bottlenecks.

---

## How is System 2 Different from Other Decision Models?

While closed-source models (Typesafe AI's **Jev**) and cloud-bound offerings (Cloudflare's **Clef** and **Clef-flash**) have demonstrated the market demand for decision models, System 2 introduces three foundational architectural distinctions:

```
Decision Index 0.2.1 (Higher = Better)
  65 ┤
     │                                           ┌──────────────┐
  60 ┤                         Clef (27B) ●      │ -- Efficient │
     │                      (209ms / 61.2)       │    Frontier  │
  55 ┤      Clef-flash (9B) ●                    └──────────────┘      Jev (Closed) ●
     │       (38.8ms / 57.1)    Open Models Cluster                    (524ms / 57.9)
  50 ┤                          [26B - 31B Models]
     │                           (100 - 140ms)
  45 ┤
     │
  40 ┤      Kev-9B ●
     │   (51.4ms / 38.5)
  35 ┤
     └──────┬──────────────────┬─────────────────┬─────────────────┬──────────────────►
            0                 100               200               300           550 ms
                                        Median Latency (ms)
```

### 1. Ultra-Compact Parameter Scale: 395M vs. 9B–27B
- **Clef & Clef-flash** freeze large autoregressive backbones (Qwen3.5-9B and Qwen3.8-27B) and attach rank-256 LoRAs. This demands **18 GB to 54 GB of VRAM**, requiring multi-GPU server infrastructure or cloud workers.
- **System 2** is built on top of **ModernBERT-Large (395M)**. At full FP16/BF16 precision, the entire model occupies only **790 MB of memory**. It runs fully local on edge devices, entry-level Apple Silicon Macs (M2/M3/M4 with 16 GB unified memory), and standard CPU nodes in under 20–35 milliseconds.

### 2. Multi-Perspective Deliberation via 5 ThoughtHeads + Judger
Rather than using a single linear classification head over mean-pooled embeddings, System 2 implements a **5-Head Deliberation Topology**:
- **5 Asymmetric Cognitive Perspectives**: Inputs and candidate options are transformed across holistic, context-skewed, choice-forward, and boundary-weighted vector spans.
- **Arithmetic Centroid Contrast**: Each candidate vector $x_i$ is contrasted against the arithmetic mean of all candidate choices ($d_i = x_i - \frac{1}{K}\sum_{j=1}^K x_j$), feeding $[x_i \,\|\, d_i]$ to the scoring MLP to prevent contrast collapse.
- **Cross-Attending Judger Arbitration**: A dedicated Judger network attends across all 5 cognitive perspective vectors to arbitrate consensus, detect fractured agreement, and emit a final decision.

### 3. Non-Parametric Epistemic Grounding & Abstention
Standard classifiers and decision models suffer from uncalibrated overconfidence when exposed to out-of-distribution or adversarial inputs. System 2 extracts an intrinsic multi-agent consensus signal ($k/5$ heads). 
- **High Consensus (4/5 or 5/5)**: Verified decision confidence; proceed with autonomous execution.
- **Fractured Consensus ($\le 2/5$)**: System 2 automatically caps confidence at $\le 0.40$, signaling the agent orchestrator to **abstain, trigger human review, or escalate to deep deliberative compute**.

### 4. Deterministic Two-Stage Subspace Routing (< 0.1 ms)
Instead of forcing a single model to hold the entire world's knowledge, System 2 pairs a frozen 395M foundation with **13 specialized domain adapters** (`science`, `code`, `med`, `electrical`, `auto`, `culinary`, `math`, `legal`, `finance`, `physics`, `safety`, `eval`, `logic`). 
- A two-stage lexical-and-centroid router dispatches queries in **< 0.1 ms**.
- The resident engine hot-swaps lightweight 1.57MB LoRA weights in **< 3 ms** in memory without reloading the base backbone.

---

## How We Trained System 2

The training protocol of System 2 adheres to strict physical and mathematical invariants designed to maximize contrastive discernment while preventing catastrophic consensus collapse.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ SYSTEM 2 TRAINING PIPELINE                                                            │
│                                                                                        │
│  [ Frozen ModernBERT-Large Backbone (395M) ] (Layers 0 – 19 strictly frozen)           │
│                           │                                                            │
│                           ▼                                                            │
│  [ Subspace LoRA Adaptation ] (Layers 20 – 27 adapted with r=32, α=32, ~1.57M params)  │
│                           │                                                            │
│                           ▼                                                            │
│  [ Asymmetric Span Extraction ] (Holistic, Question-Skewed, Choice-Forward)            │
│                           │                                                            │
│                           ▼                                                            │
│  [ 5 Parallel ThoughtHeads ] (2 Transformer Layers per Head; arithmetic centroid)     │
│                           │                                                            │
│                           ▼                                                            │
│  [ Scale-Invariant Z-Score Logit Pooling ] (Neutralizing inter-head variance disparities)│
│                           │                                                            │
│                           ▼                                                            │
│  [ Coupled Calibrated Judger Arbitration ] (Trained via Cross-Entropy + Brier Loss)    │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

### The Architectural Invariants

1. **The Subspace Rank Boundary ($r=32, \alpha=32$):**  
   LoRA adaptation is strictly confined to the top 8 layers (layers 20–27). Empirical ablation proved that adapting deeper layers induces severe lexical distortion in base semantic representations, while unconstrained rank expansion triggers late-epoch overfitting.

2. **The Cognitive Autonomy Invariant ($\lambda_{\text{ens\_ce}} = 0$):**  
   We do not backpropagate the joint ensemble cross-entropy loss into the parallel ThoughtHeads. Forcing individual heads to minimize joint ensemble error causes gradient co-adaptation, collapsing diverse perspective representations into homogenous consensus. Each ThoughtHead is optimized on its own asymmetric span view.

3. **Scale-Invariant Z-Score Normalization:**  
   Because asymmetric span projections operate across differing vector norms, raw output logits exhibit severe inter-head variance disparities (e.g. standard deviations ranging from $1.72$ down to $0.0005$). System 2 applies z-score logit standardization per head:
   $$z_{b, h, c} = \frac{l_{b, h, c} - \mu_{b, h}}{\sigma_{b, h} + \epsilon}$$
   prior to arbitration, ensuring every cognitive perspective contributes in equal statistical currency.

4. **Coupled Decision-Level Calibration Gradient:**  
   The Judger network is directly penalized by a dual-loss objective combining label-smoothed cross-entropy and a **Brier Calibration Loss**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_{\text{cal}} \cdot \mathbb{E}\left[(p_{\text{selected}} - y_{\text{true}})^2\right]$$
   This guarantees that predicted output probabilities reflect empirical accuracy, penalizing overconfident incorrect selections.

---

## How Fine-Tuning Extends System 2 Capabilities

System 2 is not a monolithic static model; it is an extensible **subspace adaptation platform**. Enterprise organizations operate proprietary workflows, unique telemetry schemas, and strict domain invariants that general-purpose foundation models cannot capture zero-shot.

Fine-tuning extends System 2 along three practical dimensions:

### 1. Instant Custom Domain Tracks (< 1.57 MB per Adapter)
Organizations can train dedicated LoRA adapters for proprietary environments without touching the frozen 395M base model:
- **Financial Compliance & AML**: Real-time transaction monitoring, SAR filing triage, and sanctions screening.
- **Cybersecurity & SOC Triage**: Microsecond analysis of WAF logs, endpoint command execution, and SIEM alert de-duplication.
- **Autonomous Agent Gatekeeping**: Validating generated tool calls, shell scripts, and SQL mutations against organizational safety policies before execution.

Because each adapter is only **1.57 MB**, thousands of customer-specific models can reside simultaneously on edge nodes or hot-swap in memory in under 3 milliseconds.

### 2. Schema Permutation & Synthetic Generalization
By training adapters across permutations of field order, natural-language criteria descriptions, and variable candidate cardinalities ($K=2$ to $K=4$), fine-tuning enables System 2 to handle dynamic, user-defined runtime schemas with high transfer fidelity.

### 3. Reinforcement Learning for Calibrated Decisions (RLCD)
For workflows involving ordinal scales (e.g., triage severity: `[No Impact, Minor, Major, Critical]` or credit risk: `[AAA, AA, A, BBB, Default]`), System 2 can be fine-tuned using RLCD. Rather than treating all misclassifications identically, RLCD:
- Grants partial credit for adjacent ordinal tiers (e.g., predicting `Major` when truth is `Critical`).
- Heavily penalizes dangerous polarity reversals (e.g., predicting `No Impact` when truth is `Critical`).
- Applies reference policy penalties to prevent distribution drift.

---

## Quickstart: Running System 2

System 2 provides an intuitive, typed Python and CLI interface compatible with modern agent frameworks:

```python
from sys1 import System1Engine

# Initialize the 395M engine on Apple Silicon MPS, CUDA, or CPU
engine = System1Engine(device="mps")

# Fast Parallel Candidate Adjudication
result = engine.adjudicate(
    question="Customer ticket: 'Our payment webhook failed with a 401 signature verification error.'",
    choices=[
        "Billing: Invoices, card charges, and subscription management",
        "Technical: API keys, webhook signatures, and integration errors",
        "Sales: Enterprise plan sizing and custom contract renewals",
        "Security: Suspected credential breach and account lockdown",
    ]
)

print(f"Selected:   {result.selected_choice}")
print(f"Confidence: {result.confidence * 100:.1f}%")
print(f"Consensus:  {result.head_consensus} ThoughtHeads")
print(f"Latency:    {result.deliberation_latency_ms:.2f} ms")
```

### CLI Inspection:
```bash
# Evaluate candidate choices:
python3 -m sys1 "Which team should triage an unauthorized SSH key injection?" \
  --choices "Billing" "Technical Infrastructure" "Security Operations" "Sales"

# Sub-millisecond domain routing:
python3 -m sys1 --route "SELECT customer_id, count(*) FROM orders GROUP BY 1;"
# Output: [CODE] (Confidence: 99.4% | Latency: 0.082 ms)
```

---

*System 2 is open-source research developed to establish non-autoregressive certainty, ultra-low latency, and epistemic reliability for autonomous AI systems.*
