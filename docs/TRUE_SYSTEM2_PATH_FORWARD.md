# True System 2 Architecture: Technical Specification & Path Forward

**Author**: System Architecture Team  
**Date**: November 2026  
**Status**: Target Architecture Blueprint

---

## 1. Architectural Vision

While System 1 provides sub-millisecond domain routing and sub-20ms candidate screening over dense text embeddings, **True System 2** is an asynchronous deliberative reasoning and verification engine.

It operates on dual-process cognitive principles:
$$\text{Output} = \text{System2\_Verify}\left(\text{System1\_Filter}(\text{Generator}(\text{Prompt}))\right)$$

---

## 2. Core Pillars of True System 2

True System 2 replaces the static 28-layer feedforward projection with three decoupled verification and reasoning engines:

```
                          [ Hypotheses from System 1 ]
                                       │
                                       ▼
        ┌─────────────────────────────────────────────────────────────┐
        │                 TRUE SYSTEM 2 DELIBERATION                  │
        │                                                             │
        │   ┌─────────────────────────────────────────────────────┐   │
        │   │ 1. Autoregressive / Recurrent State Rollouts (MCTS) │   │
        │   │    - Dynamic search depth scaling with complexity   │   │
        │   │    - Intermediate hypothesis generation & backtracking│  │
        │   └──────────────────────────┬──────────────────────────┘   │
        │                              │                              │
        │   ┌──────────────────────────▼──────────────────────────┐   │
        │   │ 2. Structural Causal Models (SCMs) & do(X) Engine   │   │
        │   │    - Directed Acyclic Graphs (DAGs) of variables    │   │
        │   │    - Interventional invariance & counterfactual rollouts│
        │   └──────────────────────────┬──────────────────────────┘   │
        │                              │                              │
        │   ┌──────────────────────────▼──────────────────────────┐   │
        │   │ 3. External Symbolic Verification Engine            │   │
        │   │    - SymPy for algebraic and numeric exactness      │   │
        │   │    - Pint for physical units and dimensional bounds │   │
        │   │    - Z3 Theorem Prover for boolean/negation logic   │   │
        │   └─────────────────────────────────────────────────────┘   │
        └──────────────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                         [ Formal Verification Verdict ]
                         - Calibrated Confidence (0.0 to 1.0)
                         - Falsification Proof / Contradiction Trace
```

---

## 3. Pillar Breakdown

### Pillar 1: Autoregressive & Recurrent State Rollouts
- **The Problem in V1:** Fixed-depth $\mathrm{TC}^0$ encoder circuits cannot solve multi-step causal chains exceeding their static depth.
- **System 2 Solution:**
  - Implements Monte Carlo Tree Search (MCTS) or Chain-of-Thought beam rollouts over intermediate premises.
  - Compute time scales dynamically: easy entity lookups take 1 search step; multi-hop proofs take $N$ unrolled steps.
  - Explicit state transitions allow backtracking when a branch encounters an empirical or logical contradiction.

### Pillar 2: Structural Causal Models (SCMs) & Intervention Simulation
- **The Problem in V1:** Bidirectional attention is unmasked and undirected in its computational graph, destroying directed causal arrows and confusing correlation with causation. For example, it easily falls into false correlations like asserting wind causes tides, when physically:
  - **Tides** are driven by the quadrupole gravitational tidal tensor / gradient of celestial bodies ($\nabla \nabla \Phi \sim G M / d^3$, primarily the Moon and Sun).
  - **Surface currents and waves** are driven by atmospheric pressure gradients exerting boundary shear stress on the water surface ($\boldsymbol{\tau}_s = \rho_a C_d |\mathbf{U}| \mathbf{U}$).
  Without an explicit causal graph, associative encoders cannot decouple these distinct physical mechanisms.
- **System 2 Solution:**
  - Extracts and maintains an explicit Directed Acyclic Graph (DAG) $\mathcal{G} = (V, E)$ of causal variables.
  - Implements Pearl's interventional operator:
    $$P(Y \mid \mathrm{do}(X = x))$$
  - Tests whether candidate conclusions hold under surgical interventions (e.g., intervening with $\text{do}(\text{wind} = 0)$ confirms lunar tidal amplitude remains completely invariant).

### Pillar 3: External Symbolic Verification Engine
- **The Problem in V1:** Subword tokenizers chop numbers and units into arbitrary string fragments, causing catastrophic failures on numeric constants (e.g., guessing $c = 150,000\text{ km/s}$) and negation.
- **System 2 Solution:**
  - **Dimensional Analysis (Pint):** Enforces Fourier's principle of dimensional homogeneity; candidate answers with mismatched physical dimensions (e.g. attempting to equate energy $[M][L]^2[T]^{-2}$ with momentum $[M][L][T]^{-1}$) are rejected deterministically before scoring.
  - **Computer Algebra (SymPy):** Solves equations and arithmetic symbolically without neural approximation.
  - **SMT Solvers (Z3):** Evaluates formal logic, negation constraints, and contraindications. If a premise states $\neg A$, Z3 guarantees that any option entailing $A$ is assigned an empirical probability of $0$.

### The Semantic Parsing Bottleneck & Complexity Bounds
- **The Classical AI Bottleneck:** Translating unstructured, ambiguous natural language into formal First-Order Logic (FOL), SMT-LIB2 assertions, or SymPy ASTs is an AI-complete task. Zero-shot causal graph discovery from arbitrary text is mathematically underdetermined.
- **The Solution:** Constrain extraction to **bounded domain schemas** and typed slot-filling contracts (e.g., extracting known physical state variables, legal rule trees, or clinical contraindications) rather than unconstrained open-world formalization.
- **Complexity & Latency Bounds:** Unbounded First-Order Logic theorem proving is undecidable, and general SMT solving can be EXPTIME-complete. The target latency of **200–800 ms** strictly applies to **bounded constraint checking** (fixed-depth SCM rollouts, dimensional arithmetic, and decidable quantifier-free SMT theories). Deep theorem proving must run as an asynchronous background verification job.

---

## 4. The Intercept & Gatekeeper Protocol (Production Deployment)

True System 2 acts as an inline intercept for generative LLMs:

1. **Step 1: Draft Generation:** Generative LLM produces candidate response with cited facts and rationales.
2. **Step 2: System 1 Pre-Filter (< 30 ms):** Dispatches domain adapter, strips irrelevant distractors, checks basic formatting.
3. **Step 3: System 2 Deliberation (200–800 ms):**
   - Symbolic verifier validates all mathematical claims and physical units.
   - SCM engine checks causal consistency.
   - SMT solver validates negation constraints.
4. **Step 4: Calibrated Action:**
   - **Confidence $\ge 85\%$:** Response is approved and streamed to user/downstream systems.
   - **Confidence $< 65\%$ or Proof Failure:** Output is intercepted. System 2 emits a structured contradiction trace back to the generator for self-correction, or safely abstains with: *"Output cannot be verified against grounded constraints."*

---

## 5. Development Roadmap

| Phase | Milestone | Deliverable |
| :--- | :--- | :--- |
| **Phase 1** | **System 1 Consolidation** | Rebrand V1 as high-speed router/pre-filter; stabilize domain adapters; freeze API. |
| **Phase 2** | **Symbolic Verification Core** | Integrate Python symbolic runtime (SymPy, Pint, Z3) into deliberation pipeline. |
| **Phase 3** | **Causal DAG Extractor** | Implement premise-to-DAG parser extracting directed causal mechanisms. |
| **Phase 4** | **Adversarial AutoResearch** | Deploy co-evolutionary AutoResearch with paired negation/counterfactual probes. |
| **Phase 5** | **Unified Intercept Gateway** | Deploy production middleware intercepting LLM outputs with calibrated pass/fail gating. |

---

## 6. The Three Strategic Implementation & Training Paths

Following the initial stabilization of the dual-process core, three distinct paths define our implementation and training trajectory:

### Path A: The Semantic Parser / DSL Extractor (The Bridge to System 2)
* **What it is**: Currently, `sys2` uses structured pattern-matching to extract equations, units, and negation statements from prompts. The single biggest bottleneck in symbolic AI is translating raw natural language into formal logic / equation ASTs.
* **Architecture & Training Task**: Train a fast token-classification or slot-filling adapter on ModernBERT to extract:
  1. **Target Physical Dimensions (for Pint)**: Extract target concept (`"velocity"` $\to$ `[length]/[time]`) and claimed quantities (`"50 J"`, `"10 m/s"`).
  2. **Clean Algebraic Expressions (for SymPy)**: Parse natural equations (*"force equals mass times acceleration"*) into symbolic ASTs (`Eq(F, m*a)`).
  3. **Propositional Clauses (for Z3)**: Extract structured tuples `(Variable, Negated, Relation)` to evaluate non-trivial entailment without substring collisions.
  4. **Causal Edges (for SCM)**: Extract directed causal mechanisms `(Cause, Effect, Direction)` to construct DAGs dynamically.

### Path B: System 1 AutoResearch Expansion
* **What it is**: Run the autonomous `runner.sh` AutoResearch loop to train higher capacity, expand to new enterprise domains, and maximize calibrated held-out accuracy across the fast perception stack.
* **Target Enterprise Tracks**:
  - **Track N: Cyber Threat Intelligence (`cyber`)**: ATT&CK technique extraction, CVSS vulnerability severity, and command safety.
  - **Track O: Supply Chain & Operations Research (`logistics`)**: Network routing, multi-echelon inventory policies, and disruption shock prediction.
  - **Track P: Tax Compliance & Statutory Accounting (`tax`)**: Statutory deduction logic, IRC code interpretation, and regulatory thresholds.

### Path C: MCTS Intermediate Hypothesis Generator
* **What it is**: Train a compact 1B–3B generative model (or value function) to propose candidate intermediate reasoning steps (premises, lemmas, and derivation actions).
* **Role in Pipeline**: The generator proposes step-by-step reasoning branches, while the deterministic `sys2` verifiers check each step in real time, enabling test-time compute scaling without unconstrained hallucinations.

---

## 7. Curated Frontier Enterprise Dataset Inventory

To support **Path B (AutoResearch Expansion)** and ground the **Path A Semantic Parser**, the following frontier datasets are cataloged:

### 1. Cyber Threat Intelligence & Security
- **`RISys-Lab/Benchmarks_CyberSec_CTI-Bench`** (NeurIPS): MITRE ATT&CK technique extraction from unstructured incident reports.
- **`ismailtasdelen/mitre-attack-techniques-qa`**: Scenario-based QA mapped across enterprise ATT&CK tactics (Persistence, Defense Evasion, Exfiltration).
- **`zefang-liu/secqa`**: Multiple-choice computer systems security reasoning benchmark with graduated complexity tiers (v1 and v2).
- **`tuandunghcmut/combine-llm-security-benchmark`**: CVE-to-CWE mapping, CVSS vulnerability severity scoring, and attack technique classification.

### 2. Supply Chain, Logistics & Operations Research
- **`Cainiao-AI/LaDe`** (Alibaba): 10.6M real-world package delivery events across 21k couriers and multi-city hubs; spatio-temporal route optimization.
- **`Isomorph2026/isomorph-supply-chain-benchmark`**: Digital twin simulator for multi-echelon supply networks and disruption management; ideal for causal SCM intervention queries ($\mathrm{do}(\text{disruption} = \text{port\_strike})$).
- **`Jacoblian/RetailOpt-190`**: Operations research optimization instances testing linear programming and inventory constraint satisfaction.
- **`LightningRodLabs/supply-chain-predictions`**: Global trade flow disruption and macroeconomic shock forecasting.

### 3. Tax Compliance & Statutory Regulatory Reasoning
- **`intelligenceconsortium/TaxBench`** (Stanford / CMU): Comprehensive benchmark for personal, corporate, estate, international, and crypto tax compliance.
- **`cfosilvia/silvia-tax-bench`**: Verifies guidance reliability against primary tax authorities (IRC statutory sections, Treasury regulations, IRS publications).
- **`isaacus/australian-tax-guidance-retrieval`**: Real taxpayer queries paired with official Australian Taxation Office (ATO) rulings and guidance.
- **`nguha/legalbench`** (Tax subset): Statutory rule application and Canadian Tax Court case disposition reasoning.

