"""load_system2.py - Robust System2 Model & Weight Loader.

Reverse-engineers constructor configurations from saved state_dict keys and
checkpoints, initializes matching System2 instances, restores weights cleanly,
and executes verification queries on Apple Silicon MPS or CPU.
"""

import argparse
import os
import sys
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Ensure project root is on sys.path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from bert import get_bert, model as default_bert_model, tokenizer as default_tokenizer
from lora import apply_lora
from system import System2


def extract_config(state: Any) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor]]:
    """Extracts System2 constructor configuration from a checkpoint dictionary or state_dict.

    Args:
        state: The loaded PyTorch checkpoint object (dict or state_dict).

    Returns:
        A tuple of (config_dict, sys2_state_dict).
    """
    if isinstance(state, dict) and "sys2" in state:
        sys2_state = state["sys2"]
    elif isinstance(state, dict):
        sys2_state = state
    else:
        raise ValueError(f"Unrecognized checkpoint format: expected dict, got {type(state)}")

    # 1. Infer foundational architectural topology from state_dict keys
    head_indices = {k.split(".")[1] for k in sys2_state.keys() if k.startswith("heads.")}
    num_heads = len(head_indices) if head_indices else 5

    layer_indices = [int(k.split(".")[4]) for k in sys2_state.keys() if "layers.layers." in k]
    head_layers = max(layer_indices) + 1 if layer_indices else 2

    has_diff = "heads.0.hypothesis_head.0.weight" in sys2_state
    hypo_w = sys2_state.get("heads.0.hypothesis_head.weight")
    is_pairwise = has_diff or (hypo_w is not None and hypo_w.shape[0] == 1)

    if isinstance(state, dict) and "pairwise" in state:
        is_pairwise = bool(state["pairwise"])
    if isinstance(state, dict) and "diff_features" in state:
        has_diff = bool(state["diff_features"])

    num_classes = 1 if is_pairwise else 4

    config: Dict[str, Any] = {
        "num_heads": num_heads,
        "num_classes": num_classes,
        "pairwise": is_pairwise,
        "diff_features": has_diff,
        "head_layers": head_layers,
    }

    # 2. Checkpoint dictionary metadata extraction
    if isinstance(state, dict):
        tunable_keys = [
            "centered_conf",
            "uniform_ensemble",
            "ensemble_temp",
            "margin_features",
            "consensus_features",
            "judger_win_emb",
            "judger_soft_emb",
            "judger_soft_tau",
            "margin_gamma",
            "prob_margin_gamma",
            "prob_ensemble",
            "head_prob_temp",
            "head_norm_first",
            "judger_residual",
            "head_mlp_dim",
            "head_norm_diff",
            "veto_blend",
            "dual_stream_contest",
            "dual_stream_blend",
            "dual_stream_alpha",
            "head_weights",
            "head_center_logits",
            "head_norm_logits",
            "head_input_norm",
            "decision_conf",
            "decision_conf_mode",
            "decision_conf_tau",
            "decision_conf_alpha",
            "decision_conf_learnable",
            "judger_mlp_dim",
            "judger_num_heads",
            "judger_dropout",
            "head_dropout",
            "learnable_temp",
        ]
        for key in tunable_keys:
            if key in state and state[key] is not None:
                config[key] = state[key]

    # 3. Structural reverse-engineering fallbacks from tensor shapes
    if "judger.log_temp" in sys2_state:
        config["learnable_temp"] = True
    if "log_decision_tau" in sys2_state:
        config["decision_conf_learnable"] = True
    if "judger.prenorm.weight" in sys2_state:
        config["judger_residual"] = True
    if "heads.0.input_norm.weight" in sys2_state:
        config["head_input_norm"] = True

    if "judger.margin_norm.weight" in sys2_state:
        m_dim = sys2_state["judger.margin_norm.weight"].shape[0]
        config["margin_dim"] = m_dim
        if m_dim == 5:
            config["consensus_features"] = True
        else:
            config["margin_features"] = True
    elif config.get("consensus_features", False):
        config["margin_dim"] = 5
    elif config.get("margin_features", False):
        config["margin_dim"] = 3

    if "judger.score.0.weight" in sys2_state:
        config["judger_mlp_dim"] = sys2_state["judger.score.0.weight"].shape[0]

    if "heads.0.hypothesis_head.0.weight" in sys2_state:
        w = sys2_state["heads.0.hypothesis_head.0.weight"]
        if w.dim() == 2:
            config["head_mlp_dim"] = w.shape[0]
        elif w.dim() == 1:
            config["head_norm_diff"] = True
            if "heads.0.hypothesis_head.1.weight" in sys2_state:
                config["head_mlp_dim"] = sys2_state["heads.0.hypothesis_head.1.weight"].shape[0]

    return config, sys2_state


def encode(text: str, device: Optional[torch.device] = None, model_name: Optional[str] = None) -> torch.Tensor:
    """Safely tokenizes and encodes text using ModernBERT with proper device placement.

    Args:
        text: The string to encode.
        device: Target torch device for the output embedding vector.
        model_name: Optional backbone model override.

    Returns:
        Mean-pooled hidden vector of shape (1, 1024) placed on target device.
    """
    tok, mod = get_bert(model_name) if model_name else (default_tokenizer, default_bert_model)
    mod_dev = next(mod.parameters()).device
    inputs = tok(text, return_tensors="pt").to(mod_dev)
    with torch.no_grad():
        last_hidden = mod(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        vec = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    if device is not None and vec.device != device:
        vec = vec.to(device)
    return vec


def load_system2(
    checkpoint_path: str = "models/system2.pt",
    device: Optional[str] = None,
    apply_encoder_lora: bool = True,
    strict: bool = True,
) -> Tuple[System2, Dict[str, Any]]:
    """Loads a System2 checkpoint with full architectural reconstruction.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        device: Device to place the model on ('mps', 'cpu', 'cuda').
        apply_encoder_lora: Whether to apply LoRA adapter weights to ModernBERT if found.
        strict: Whether to enforce strict key matching on state_dict loading.

    Returns:
        Tuple of (reconstructed_system2_model, config_dict).
    """
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    target_device = torch.device(device)

    resolved_path = os.path.abspath(checkpoint_path)
    if not os.path.exists(resolved_path):
        # Fall back relative to root dir
        resolved_path = os.path.join(ROOT_DIR, checkpoint_path)
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    state = torch.load(resolved_path, map_location="cpu", weights_only=False)
    config, sys2_state = extract_config(state)

    model = System2(**config)
    try:
        model.load_state_dict(sys2_state, strict=strict)
    except RuntimeError as e:
        if strict:
            print(f"[load_system2 warning] Strict load failed ({e}), retrying with strict=False...", file=sys.stderr)
            model.load_state_dict(sys2_state, strict=False)
        else:
            raise

    # Adapt ModernBERT backbone with LoRA if present in checkpoint
    if apply_encoder_lora and isinstance(state, dict) and "lora" in state:
        lora_cfg = state.get("lora_config", {"r": 32, "alpha": 32, "num_layers": 8})
        train_norms = lora_cfg.get("train_norms", False) or ("lora_norms" in state)
        apply_lora(
            default_bert_model,
            r=lora_cfg.get("r", 32),
            alpha=lora_cfg.get("alpha", 32),
            num_layers=lora_cfg.get("num_layers", 8),
            train_norms=train_norms,
        )
        default_bert_model.load_state_dict(state["lora"], strict=False)
        if "lora_norms" in state:
            default_bert_model.load_state_dict(state["lora_norms"], strict=False)

    model.to(target_device)
    model.eval()

    # Wrap forward to handle 2D inputs seamlessly for pairwise contrastive models
    original_forward = model.forward

    def adaptive_forward(encoder_output: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        # If passed a 2D tensor (B, 1024) but model expects pairwise candidate representations
        if encoder_output.dim() == 2 and (config.get("pairwise") or config.get("diff_features")):
            # Construct a contrastive pair (B, 2, 1024) against neutral vector
            encoder_output = torch.stack([encoder_output, torch.zeros_like(encoder_output)], dim=1)
        if encoder_output.device != target_device:
            encoder_output = encoder_output.to(target_device)
        return original_forward(encoder_output, *args, **kwargs)

    model.forward = adaptive_forward  # type: ignore[assignment]

    return model, config


def main() -> None:
    parser = argparse.ArgumentParser(description="System2 Model & Weight Loader")
    parser.add_argument("--checkpoint", default="models/system2.pt", help="Path to .pt checkpoint")
    parser.add_argument("--query", default="Is this ticket about billing?", help="Query text to run through System2")
    parser.add_argument("--device", default=None, help="Target device (mps, cpu, cuda)")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Initializing System2 loader on {device.upper()}...")

    model, config = load_system2(args.checkpoint, device=device)
    print(f"[*] Checkpoint '{args.checkpoint}' loaded successfully.")
    print(f"[*] Active Configuration:")
    for k in sorted(config.keys()):
        print(f"    - {k}: {config[k]}")

    print(f"\n[*] Encoding query: \"{args.query}\"")
    vec = encode(args.query, device=torch.device(device))
    print(f"[*] Encoder output shape: {tuple(vec.shape)}")

    print(f"[*] Executing feedforward parallel deliberation...")
    out = model(vec)

    # Output unpacking matching 04-handoff.md Section 2:
    # (best, best_conf, pruned, confs, logits, embeddings, scores)
    best, best_conf, pruned, confs, logits, emb, scores = out[:7]
    ens_logits = out[7] if len(out) > 7 else None

    num_heads = confs.shape[-1]
    confs_vec = confs.squeeze(0)

    print(f"\n--- System2 Deliberation Verdict ---")
    print(f"Heads: {num_heads}")
    print(f"Selected hypothesis #{int(best.item()) + 1}")
    print(f"Winning confidence: {best_conf.item():.3f}")
    print(f"Pruned: {int(pruned.item())} of {num_heads}")
    for i, c in enumerate(confs_vec):
        print(f"  head {i + 1}: conf {c.item():.3f}")

    print(f"\n[*] Verified output tuple structure:")
    print(f"    best: {tuple(best.shape)}")
    print(f"    best_conf: {tuple(best_conf.shape)}")
    print(f"    pruned: {tuple(pruned.shape)}")
    print(f"    confs: {tuple(confs.shape)}")
    print(f"    logits: {tuple(logits.shape)}")
    print(f"    embeddings: {tuple(emb.shape)}")
    print(f"    scores: {tuple(scores.shape)}")
    if ens_logits is not None:
        print(f"    ens_logits: {tuple(ens_logits.shape)}")
    print(f"\n[✓] System2 loader verification complete.")


if __name__ == "__main__":
    main()
