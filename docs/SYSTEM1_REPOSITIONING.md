# System 1 Repositioning & Adversarial Multi-Agent Audit

**Date**: November 2026  
**Status**: Official Architectural Transition

---

## 1. Context & Executive Summary

The feedforward architecture implemented in this repository (ModernBERT-Large 395M backbone + 1.57M LoRA adapters + 5-head transformer stack + Cross-Attending Judger) was originally conceived as a non-autoregressive "System 2" deliberation engine.

Following a rigorous multi-agent adversarial audit conducted by specialized agents across Machine Learning Architecture, Theoretical Physics, and Systems Safety Red-Teaming, the architecture has been **formally repositioned as System 1 (Fast Perception & Candidate Pre-Filtering)**.

While the engine demonstrates exceptional execution speed (< 20 ms on Apple Silicon GPU), high in-domain classification accuracy on benchmark distributions, and strict choice-order permutation invariance, it **does not perform causal deliberation**. Under computational complexity theory and causal inference (Judea Pearl's do-calculus), the architecture operates as a **static semantic similarity matcher in $\mathbb{R}^{1024}$**.

---

## 2. Why V1 is an Excellent System 1

V1 exhibits the defining properties of cognitive "System 1" (Kahneman dual-process theory): fast, autonomous, low-overhead perceptual screening and routing:

1. **Deterministic Sub-Millisecond Domain Routing (< 0.1 ms):**
   - Dispatches prompts to 13 domain tracks (`science`, `code`, `med`, `electrical`, `auto`, `culinary`, `math`, `legal`, `finance`, `physics`, `safety`, `eval`, `logic`) via lexical prefix matching and semantic centroid projection.
2. **Instant In-Memory Adapter Hot-Swapping (< 3 ms):**
   - Keeps the single 395M ModernBERT base resident in memory and hot-swaps domain-specialized weights instantly without reloading.
3. **High-Speed Roughness Filter (< 20 ms on GPU, < 300 ms on CPU):**
   - Evaluates 4 candidate options simultaneously in a single feedforward matrix pass.
   - Contrastive centroid differencing ($[x_i \,\|\, x_i - \bar{x}]$) rapidly eliminates distant or irrelevant distractors.
4. **Strict Permutation Invariance (0.00% Flip Rate):**
   - Cyclically shifting candidate choices ($A \to B \to C \to D$) produces identical logits and zero choice flips.

---

## 3. Why V1 Fails as True System 2

True System 2 deliberation requires computational depth that scales with problem hardness, state transitions, interventional causal modeling, and symbolic precision. V1 exhibits five fundamental theoretical and empirical ceilings:

1. **Circuit Complexity Ceiling ($\mathrm{TC}^0$):**
   - A fixed-depth 28-layer transformer encoder without autoregressive token generation or recurrent loops belongs to the uniform circuit complexity class $\mathrm{TC}^0$.
   - It cannot solve iterative reachability or multi-step causal chains ($A \to B \to C \to D$) when the chain length exceeds its static depth.
2. **Confined to Pearl's Layer 1 (Statistical Association $P(Y \mid X)$):**
   - Trained via Masked Language Modeling (MLM), ModernBERT models token co-occurrence probabilities.
   - While attention weights are row-normalized ($Q \neq K$), the underlying computational graph is unmasked and undirected across tokens, destroying the directed acyclic graph (DAG) topology required for structural causal intervention ($P(Y \mid \mathrm{do}(X))$).
   - Even causal autoregressive decoders with directed attention masks remain trapped in Pearl's Layer 1 ($L_1: P(Y \mid X)$) without an explicit structural causal model (SCM) defining background noise distributions and structural assignment equations $X_i = f_i(\mathrm{PA}_i, U_i)$.
   - Mean-pooling collapses relational binding: *"The drug caused the cure"* and *"The cure caused the drug"* occupy nearly identical positions in vector space.
3. **Coplanar ThoughtHeads:**
   - In `encoding.py`, the 5 "cognitive perspectives" are linear convex combinations:
     $$v_0 = \text{holistic}, \quad v_1 = 0.40 q + 0.60 a, \quad v_2 = 0.20 q + 0.80 a, \quad v_3 = 0.50 q + 0.50 a, \quad v_4 = 0.05 q + 0.95 a$$
   - These are strictly coplanar points along a 1D line segment connecting the question mean $q$ and answer mean $a$. The subsequent ThoughtHeads are 5 shallow 2-layer MLPs over sequence length $L=4$.
4. **Subword Numerical Blindness:**
   - Numbers are chopped into arbitrary discrete subwords (e.g. `299,792 km/s` $\to$ `['299', ',', '792', ...]`).
   - The model has no internal representation of magnitude ordering ($>$, $<$) or physical dimensional homogeneity ($[L][T]^{-1}$). It failed on *"What is the speed of light in a vacuum?"*, selecting `150,000 km/s` over `299,792 km/s`.
5. **Negation Blindness:**
   - Multiplying modal negation tokens by $3.0$ injects energy into the vector sum but cannot invert boolean truth.
   - When asked *"Which will NOT cause an iron nail to rust?"*, it selected **"Exposing it to humid air and rain"** with 5/5 head agreement.

---

## 4. The Path Forward: True System 2

Rather than attempting to force a feedforward encoder to perform tasks it is mathematically incapable of, the architecture is split into a coordinated dual-process pipeline:

- **System 1 (This Repository):** Fast perceptual routing, domain adapter hot-swapping, candidate pre-filtering, and coarse outlier elimination in $< 20\text{ ms}$.
- **True System 2 (Next-Gen Architecture):** Autoregressive/recurrent hypothesis rollouts (MCTS), Structural Causal Models (SCMs with $\mathrm{do}(X)$ operators), and external symbolic engines (SymPy, Pint, SMT solvers) for formal verification.
