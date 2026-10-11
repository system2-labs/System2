#!/usr/bin/env python3
"""interact.py - Interactive Parallel Deliberation with System2.

Allows interactive querying, random test-set inspection, and full transparency
into the internal deliberation dynamics of System2's 5 ThoughtHeads and Judger.

Usage:
  # Interactive Prompt Mode:
  python3 interact.py

  # Test-Set Sample Mode (inspect random held-out scientific question):
  python3 interact.py --sample

  # One-shot CLI Mode:
  python3 interact.py "What causes ocean tides?" "Earth's rotation" "Moon's gravity" "Wind currents" "Sun's magnetism"
"""
import argparse
import sys
import time
import torch

from bert import tokenizer, model
from system import System2


def load_system2(checkpoint_path="./models/system2.pt", device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[*] Loading System2 checkpoint from {checkpoint_path} onto {device}...", flush=True)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    if isinstance(state, dict) and "sys2" in state:
        sys2_state = state["sys2"]
        if "lora" in state:
            from lora import apply_lora
            cfg = state.get("lora_config", {"r": 32, "alpha": 32, "num_layers": 8})
            train_norms = cfg.get("train_norms", False) or ("lora_norms" in state)
            apply_lora(model, r=cfg.get("r", 32), alpha=cfg.get("alpha", 32), num_layers=cfg.get("num_layers", 8), train_norms=train_norms)
            model.load_state_dict(state["lora"], strict=False)
            if "lora_norms" in state:
                model.load_state_dict(state["lora_norms"], strict=False)
    else:
        sys2_state = state

    model.to(device)
    model.eval()

    num_heads = len({k.split(".")[1] for k in sys2_state.keys() if k.startswith("heads.")})
    has_diff = "heads.0.hypothesis_head.0.weight" in sys2_state
    layer_indices = [int(k.split(".")[4]) for k in sys2_state.keys() if "layers.layers." in k]
    head_layers = max(layer_indices) + 1 if layer_indices else 2

    kwargs = {
        "num_heads": num_heads,
        "num_classes": 1 if has_diff else 4,
        "pairwise": True,
        "diff_features": has_diff,
        "head_layers": head_layers,
    }
    if isinstance(state, dict) and state.get("centered_conf", False):
        kwargs["centered_conf"] = True
    if isinstance(state, dict) and state.get("uniform_ensemble", False):
        kwargs["uniform_ensemble"] = True
    if isinstance(state, dict) and "ensemble_temp" in state:
        kwargs["ensemble_temp"] = state["ensemble_temp"]
    if (isinstance(state, dict) and state.get("margin_features", False)) or ("judger.margin_norm.weight" in sys2_state):
        kwargs["margin_features"] = True
    if isinstance(state, dict) and "margin_gamma" in state:
        kwargs["margin_gamma"] = state["margin_gamma"]
    if isinstance(state, dict) and "prob_ensemble" in state:
        kwargs["prob_ensemble"] = state["prob_ensemble"]
    if isinstance(state, dict) and "head_prob_temp" in state:
        kwargs["head_prob_temp"] = state["head_prob_temp"]
    if (isinstance(state, dict) and state.get("judger_residual", False)) or ("judger.prenorm.weight" in sys2_state):
        kwargs["judger_residual"] = True
    if isinstance(state, dict) and state.get("head_norm_first", False):
        kwargs["head_norm_first"] = True
    if isinstance(state, dict) and "head_mlp_dim" in state:
        kwargs["head_mlp_dim"] = state["head_mlp_dim"]
    elif "heads.0.hypothesis_head.0.weight" in sys2_state and sys2_state["heads.0.hypothesis_head.0.weight"].dim() == 2:
        kwargs["head_mlp_dim"] = sys2_state["heads.0.hypothesis_head.0.weight"].shape[0]

    if isinstance(state, dict) and state.get("head_norm_diff", False):
        kwargs["head_norm_diff"] = True
    elif "heads.0.hypothesis_head.0.weight" in sys2_state and sys2_state["heads.0.hypothesis_head.0.weight"].dim() == 1:
        kwargs["head_norm_diff"] = True
        if "heads.0.hypothesis_head.1.weight" in sys2_state:
            kwargs["head_mlp_dim"] = sys2_state["heads.0.hypothesis_head.1.weight"].shape[0]

    if isinstance(state, dict) and "veto_blend" in state:
        kwargs["veto_blend"] = state["veto_blend"]

    sys2 = System2(**kwargs)
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()

    meta = {
        "max_len": state.get("max_len", 128) if isinstance(state, dict) else 128,
        "ensemble_heads": state.get("ensemble_heads", True) if isinstance(state, dict) else True,
        "uniform_ensemble": state.get("uniform_ensemble", False) if isinstance(state, dict) else False,
        "span_pooling": state.get("span_pooling", False) if isinstance(state, dict) else False,
        "span_weight": state.get("span_weight", 0.5) if isinstance(state, dict) else 0.5,
        "span_mode": state.get("span_mode", "mean") if isinstance(state, dict) else "mean",
        "asym_spans": state.get("asym_spans", False) if isinstance(state, dict) else False,
        "device": device,
    }
    return sys2, meta


def deliberate(sys2, meta, question, choices, max_len=None, dynamic_len=True, model_ceiling=None):
    """Executes feedforward parallel deliberation for a single question and 4 choices."""
    device = meta["device"]
    flat_texts = [f"Question: {question} Answer: {c}" for c in choices]
    
    ceiling = model_ceiling or getattr(model.config, "max_position_embeddings", 8192)
    
    # Calculate exact un-truncated token counts per candidate sequence
    raw_token_counts = [len(tokenizer.encode(t, truncation=False)) for t in flat_texts]
    max_input_tokens = max(raw_token_counts)
    
    # Determine effective sequence length
    if max_len is not None:
        target_len = min(max_len, ceiling)
    elif dynamic_len:
        # Dynamically scale to the exact length of the input, capped at model ceiling
        target_len = min(max(max_input_tokens, 1), ceiling)
    else:
        target_len = min(meta.get("max_len", 512), ceiling)
    
    warnings = []
    if max_input_tokens >= ceiling:
        warn_msg = (
            f"⚠️  [CONTEXT CEILING REACHED] Input length ({max_input_tokens} tokens) meets or exceeds "
            f"the architectural ceiling ({ceiling} tokens). Trailing context, return statements, or assertions were truncated!"
        )
        warnings.append(warn_msg)
    elif max_input_tokens > 4096:
        notice_msg = (
            f"ℹ️  [HIGH CONTEXT NOTICE] Sequence length ({max_input_tokens} tokens) utilizes deep context (>4,096 tokens). "
            f"Rotary Position Embeddings (RoPE) extrapolation active up to {ceiling} ceiling."
        )
        warnings.append(notice_msg)
    elif max_input_tokens > target_len:
        warn_msg = (
            f"⚠️  [TRUNCATION WARNING] Input length ({max_input_tokens} tokens) exceeds active limit ({target_len}). "
            f"Dynamic context expansion recommended to prevent clipping."
        )
        warnings.append(warn_msg)
    
    t0 = time.time()
    inputs = tokenizer(flat_texts, return_tensors="pt", padding=True, truncation=True, max_length=target_len).to(device)
    actual_seq_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        last_hidden = model(**inputs).last_hidden_state
        if meta.get("asym_spans", False):
            import torch.nn.functional as F
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            B, T = input_ids.shape
            match = (input_ids[:, :-1] == 37741) & (input_ids[:, 1:] == 27)
            has_match = match.any(dim=1)
            ans_pos = match.int().argmax(dim=1)
            cols = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            valid_lens = attn_mask.sum(dim=1) - 1

            q_mask = (cols < ans_pos.unsqueeze(1)) & (attn_mask.bool())
            a_mask = (cols >= (ans_pos + 2).unsqueeze(1)) & (cols < valid_lens.unsqueeze(1)) & (attn_mask.bool())

            no_match = ~has_match.unsqueeze(1)
            q_mask = torch.where(no_match, attn_mask.bool(), q_mask)
            a_mask = torch.where(no_match, attn_mask.bool(), a_mask)

            q_mask_f = q_mask.unsqueeze(-1).float()
            a_mask_f = a_mask.unsqueeze(-1).float()
            q_vec = (last_hidden * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
            a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)

            mask = inputs["attention_mask"].unsqueeze(-1)
            holistic = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

            asym_span_mode = meta.get("asym_span_mode", "default")
            if asym_span_mode == "causal_attn":
                H_norm = F.normalize(last_hidden, p=2, dim=-1)
                q_norm = F.normalize(q_vec, p=2, dim=-1)
                cos_sim = (H_norm * q_norm.unsqueeze(1)).sum(dim=-1)
                scores = (cos_sim / 0.1).masked_fill(~a_mask, -1e9)
                attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
                a_attn = (last_hidden * attn_weights).sum(dim=1)

                v0 = holistic
                v1 = 0.35 * q_vec + 0.65 * a_vec
                v2 = 0.15 * q_vec + 0.85 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = 0.35 * q_vec + 0.65 * a_attn
            elif asym_span_mode == "choice_forward":
                v0 = holistic
                v1 = 0.40 * q_vec + 0.60 * a_vec
                v2 = 0.20 * q_vec + 0.80 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = 0.05 * q_vec + 0.95 * a_vec
            else:
                v0 = holistic
                v1 = 0.75 * q_vec + 0.25 * a_vec
                v2 = 0.25 * q_vec + 0.75 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = holistic

            multi_head_vecs = torch.stack([v0, v1, v2, v3, v4], dim=1)  # (4, 5, 1024)
            vecs = multi_head_vecs.view(1, len(choices), 5, -1).permute(0, 2, 1, 3)  # (1, 5, 4, 1024)
        elif meta.get("span_pooling", False):
            import torch.nn.functional as F
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            B, T = input_ids.shape
            match = (input_ids[:, :-1] == 37741) & (input_ids[:, 1:] == 27)
            has_match = match.any(dim=1)
            ans_pos = match.int().argmax(dim=1)
            cols = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            valid_lens = attn_mask.sum(dim=1) - 1

            q_mask = (cols < ans_pos.unsqueeze(1)) & (attn_mask.bool())
            a_mask = (cols >= (ans_pos + 2).unsqueeze(1)) & (cols < valid_lens.unsqueeze(1)) & (attn_mask.bool())

            no_match = ~has_match.unsqueeze(1)
            q_mask = torch.where(no_match, attn_mask.bool(), q_mask)
            a_mask = torch.where(no_match, attn_mask.bool(), a_mask)

            q_mask_f = q_mask.unsqueeze(-1).float()
            a_mask_f = a_mask.unsqueeze(-1).float()
            q_vec = (last_hidden * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
            span_mode = meta.get("span_mode", "mean")
            if span_mode == "attn":
                H_norm = F.normalize(last_hidden, p=2, dim=-1)
                q_norm = F.normalize(q_vec, p=2, dim=-1)
                cos_sim = (H_norm * q_norm.unsqueeze(1)).sum(dim=-1)
                scores = (cos_sim / 0.1).masked_fill(~a_mask, -1e9)
                attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
                a_vec = (last_hidden * attn_weights).sum(dim=1)
            else:
                a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)
            span_weight = meta.get("span_weight", 0.5)
            pooled = (1.0 - span_weight) * q_vec + span_weight * a_vec
            vecs = pooled.view(1, len(choices), -1)  # (1, 4, 1024)
        else:
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            vecs = pooled.view(1, len(choices), -1)  # (1, 4, 1024)

        out = sys2(vecs)
        best, conf, pruned, confs, logits, emb, scores = out[:7]
        ens_logits = out[7] if len(out) > 7 else torch.einsum('bn,bnc->bc', torch.softmax(scores, dim=-1), logits)
    latency_ms = (time.time() - t0) * 1000

    head_weights = torch.softmax(scores, dim=-1)[0].tolist()
    head_confs = confs[0].tolist()
    
    # Per-head rankings
    head_predictions = []
    for h in range(logits.size(1)):
        h_logits = logits[0, h].tolist()
        pred_idx = int(torch.argmax(logits[0, h]).item())
        head_predictions.append({
            "head": h + 1,
            "weight": head_weights[h],
            "conf": head_confs[h],
            "pred_idx": pred_idx,
            "pred_char": chr(65 + pred_idx),
            "logits": h_logits,
        })

    ens_pred_idx = int(torch.argmax(ens_logits[0]).item())
    
    return {
        "question": question,
        "choices": choices,
        "winning_choice_idx": ens_pred_idx,
        "winning_choice_char": chr(65 + ens_pred_idx),
        "winning_choice_text": choices[ens_pred_idx],
        "winning_conf": float(conf[0].item()),
        "head_predictions": head_predictions,
        "ens_logits": ens_logits[0].tolist(),
        "latency_ms": latency_ms,
        "raw_token_counts": raw_token_counts,
        "input_tokens": max_input_tokens,
        "effective_len": actual_seq_len,
        "model_ceiling": ceiling,
        "warnings": warnings,
    }


def print_deliberation(res, gold_char=None):
    chars = ["A", "B", "C", "D"]
    raw_counts = res.get("raw_token_counts")
    print("\n" + "=" * 70)
    print(f"QUESTION: {res['question']}")
    print("-" * 70)
    for idx, c in enumerate(res['choices']):
        marker = " "
        if chars[idx] == res['winning_choice_char']:
            marker = "➤"
        gold_tag = " (CORRECT)" if (gold_char and chars[idx] == gold_char) else ""
        tok_tag = f" [{raw_counts[idx]} tok]" if raw_counts else ""
        print(f" {marker} [{chars[idx]}]{tok_tag} {c}{gold_tag}")
    print("=" * 70)

    if res.get("warnings"):
        print("\n--- ⚠️  CONTEXT & ARCHITECTURAL BOUNDARY ALERTS ---")
        for w in res["warnings"]:
            print(f"  {w}")

    print("\n--- 🧠 PARALLEL DELIBERATION (5 THOUGHTHEADS) ---")
    for hp in res['head_predictions']:
        logits_str = ", ".join([f"{chars[i]}: {v:+.2f}" for i, v in enumerate(hp['logits'])])
        print(f"  Head #{hp['head']}: Selected [{hp['pred_char']}]  | Judger Weight: {hp['weight']*100:5.1f}% | Choice Logits: [{logits_str}]")

    print("\n--- ⚖️  JUDGER ARBITRATION & FINAL DECISION ---")
    ens_logits_str = ", ".join([f"{chars[i]}: {v:+.2f}" for i, v in enumerate(res['ens_logits'])])
    print(f"  Ensemble Logits:     [{ens_logits_str}]")
    print(f"  Winning Prediction:  [{res['winning_choice_char']}] {res['winning_choice_text']}")
    print(f"  Calibrated Conf:     {res['winning_conf'] * 100:.1f}%")
    eff_len = res.get('effective_len', 'N/A')
    in_tok = res.get('input_tokens', eff_len)
    ceil = res.get('model_ceiling', 8192)
    print(f"  Sequence Context:    {eff_len} tokens (input: {in_tok} tok, arch ceiling: {ceil} tok)")
    print(f"  Inference Latency:   {res['latency_ms']:.1f} ms (feedforward non-autoregressive)")

    if gold_char:
        status = "✅ CORRECT" if res['winning_choice_char'] == gold_char else "❌ INCORRECT"
        print(f"  Outcome:             {status} (Ground truth was [{gold_char}])")
    print("=" * 70 + "\n")


def sample_test_set(sys2, meta, n=1, data_dir="./data/arc", max_len=None, dynamic_len=True, model_ceiling=None):
    from datasets import load_from_disk
    import random
    ds = load_from_disk(data_dir)["test"]
    chars = ["A", "B", "C", "D"]
    
    indices = random.sample(range(len(ds)), n)
    for idx in indices:
        item = ds[idx]
        q = item["question"]
        choices = item["choices"]
        gold_char = chars[item["label"]]
        res = deliberate(sys2, meta, q, choices, max_len=max_len, dynamic_len=dynamic_len, model_ceiling=model_ceiling)
        print(f"[Test Sample #{idx + 1} of {len(ds)}]")
        print_deliberation(res, gold_char=gold_char)


def main():
    import os
    parser = argparse.ArgumentParser(description="Interact with System2 Deliberation Engine")
    parser.add_argument("args", nargs="*", help="Question followed by 4 choices")
    parser.add_argument("--ckpt", default=None, help="Path to checkpoint (defaults to ./models/system2.pt or ./models/system2_code.pt)")
    parser.add_argument("--code", action="store_true", help="Use Track B (System2-Code) checkpoint and dataset")
    parser.add_argument("--data", default=None, help="Path to dataset for sampling")
    parser.add_argument("--sample", action="store_true", help="Sample random question(s) from held-out test set")
    parser.add_argument("--n-samples", type=int, default=1, help="Number of test samples to evaluate")
    parser.add_argument("--max-len", type=int, default=None, help="Explicit maximum sequence length (overrides dynamic context)")
    parser.add_argument("--no-dynamic", action="store_true", help="Disable dynamic context sizing (fall back to checkpoint max_len)")
    parser.add_argument("--ceiling", type=int, default=8192, help="Architectural ceiling token limit (default: 8192)")
    args = parser.parse_args()

    # Determine default checkpoint and dataset paths
    if args.ckpt is None:
        if args.code:
            ckpt_path = "./models/system2_code.pt"
        elif os.path.exists("./models/system2_code.pt") and not os.path.exists("./models/system2.pt"):
            ckpt_path = "./models/system2_code.pt"
        else:
            ckpt_path = "./models/system2.pt"
    else:
        ckpt_path = args.ckpt

    is_code_track = args.code or ("code" in ckpt_path)
    if args.data is None:
        data_path = "./data/code_traces" if is_code_track else "./data/arc"
    else:
        data_path = args.data

    sys2, meta = load_system2(ckpt_path)
    dynamic_len = not args.no_dynamic

    if args.sample:
        sample_test_set(
            sys2, meta, n=args.n_samples, data_dir=data_path,
            max_len=args.max_len, dynamic_len=dynamic_len, model_ceiling=args.ceiling,
        )
        return

    if len(args.args) == 5:
        q = args.args[0]
        choices = args.args[1:5]
        res = deliberate(
            sys2, meta, q, choices,
            max_len=args.max_len, dynamic_len=dynamic_len, model_ceiling=args.ceiling,
        )
        print_deliberation(res)
        return

    if len(args.args) > 0 and len(args.args) != 5:
        print("Usage: python3 interact.py \"Question\" \"Choice A\" \"Choice B\" \"Choice C\" \"Choice D\"")
        sys.exit(1)

    print("\n" + "=" * 70)
    if is_code_track:
        print("WELCOME TO SYSTEM2-CODE INTERACTIVE DELIBERATION REPL (TRACK B)")
        print("Feedforward parallel deliberation over Python code execution traces.")
    else:
        print("WELCOME TO SYSTEM2 INTERACTIVE DELIBERATION REPL (TRACK A)")
        print("Feedforward parallel deliberation over scientific causal reasoning.")
    print(f"Dynamic Context: {'ENABLED (cap: ' + str(args.ceiling) + ' tokens)' if dynamic_len else 'DISABLED'}")
    print("Press Ctrl+C or type 'exit' to quit. Type 'sample' to draw from held-out test set.")
    print("=" * 70 + "\n")

    chars = ["A", "B", "C", "D"]
    while True:
        try:
            q = input("\nEnter Question / Code Snippet (or 'sample' / 'exit'): ").strip()
            if not q:
                continue
            if q.lower() in ["exit", "quit"]:
                break
            if q.lower() == "sample":
                sample_test_set(
                    sys2, meta, n=1, data_dir=data_path,
                    max_len=args.max_len, dynamic_len=dynamic_len, model_ceiling=args.ceiling,
                )
                continue

            choices = []
            for c_char in chars:
                c = input(f"  Choice [{c_char}]: ").strip()
                choices.append(c)

            res = deliberate(
                sys2, meta, q, choices,
                max_len=args.max_len, dynamic_len=dynamic_len, model_ceiling=args.ceiling,
            )
            print_deliberation(res)
        except KeyboardInterrupt:
            print("\nExiting System2 REPL.")
            break


if __name__ == "__main__":
    main()
