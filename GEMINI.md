# System2 Architectural Invariants & AutoResearch Protocol

## 1. AutoResearch Protocol & Safety Guardrails
- **The Metric Invariant**: Never manually lower or overwrite `.best_metric`. All commits to `main` must strictly exceed the guarded metric on the full held-out evaluation set.
- **The Pre-Test Commit Law**: All architectural infrastructure code (e.g. new parameters, loss functions, head structures) must be committed to git BEFORE running `runner.sh`. This ensures that when `runner.sh` executes `git reset --hard HEAD` on a regression, the core capability is preserved while only the hyperparameter trial is reverted.
- **Strictly Non-Autoregressive**: The architecture must remain 100% feedforward and non-autoregressive. No token-by-token generation loops or sequential autoregressive decoders.

## 2. Encoder & Representation Learning
- **Frozen Base Backbone**: ModernBERT-large (395M) weights remain strictly frozen. Feature extraction produces dense bidirectional embeddings without autoregressive drift or alignment artifacts.
- **The Subspace Rank Sweet Spot**: LoRA adaptation is restricted to the top 8 layers (layers 20–27). The optimal capacity boundary is $r=32, \alpha=32$ (~1.57M parameters). Adapting deeper than 8 layers causes lexical distortion; higher rank without scheduled decay induces late-epoch overfitting.
- **The In-Domain Causal Purity Law**: Do not mix external factual trivia datasets (e.g. SciQ, OpenBookQA) into the shared LoRA backbone. Multi-domain mixing warps the shared representations toward surface syntax, severely harming in-domain causal reasoning. Expand data only using pure in-domain splits (ARC train + validation).

## 3. Deliberation & Hypothesis Generation
- **The 5-Head Deliberation Equilibrium**: 5 full dense parallel thought heads evaluate candidate choices simultaneously. Avoid sparse MoE gating (which starves the Judger from $5\times 5 \to 2\times 2$ interactions) and head oversaturation ($N=8$, which dilutes the supervised Judger selection target).
- **Arithmetic Centroid Contrast Invariance**: Always contrast candidate vectors against the fixed arithmetic mean: $d_i = x_i - \frac{1}{4}\sum_{j=1}^4 x_j$, feeding $[x_i \,\|\, d_i]$ to the scoring MLP. Never use dynamic attention SoftCentroid, which collapses contrast toward zero on salient options.
- **Intra-Choice Transformer Depth**: Use exactly 2 transformer self-attention layers per ThoughtHead. For candidate sequence length $L=4$, deeper stacks (3+ layers) induce optimization bottlenecks and fail to train from scratch without pretraining.

## 4. Training Dynamics & Loss Formulations
- **The Cosine Annealing Cooling Law**: All training must use a Cosine Annealing learning rate schedule decaying to $1\times 10^{-5}$ across total steps. Constant learning rates cause late-epoch oscillation around narrow minima.
- **Coupled Decision-Level Calibration Gradient**: The Judger must receive sharp, unattenuated gradients from whether its selected hypothesis is objectively correct ($\mathcal{L}_{\text{judge\_cal}}$). Never sever this gradient with detached calibrators or soften it with uncalibrated temperature dividers.
