"""eval.py - JUDGE for the System2 AG-News experiment.

Read-only oracle (like prepare.py in test-training): loads a checkpoint,
runs the full test set, emits ONE JSON metric line on stdout.

  {"metric": <float>, "accuracy": <float>, "brier": <float>}

metric = 0.5 * accuracy + 0.5 * (1 - Brier)   # v1: Brier, not true ECE (TODO)

Why blend accuracy with calibration: accuracy alone can be gamed by
overconfident wrong guesses. Brier punishes it.
NOTE (TODO): the metric field is really the Brier score, not true ECE (which
needs confidence bins). Mirrors Laya's strict-scoring
spirit - honest calibration counts toward the score.

Exit codes:
  0 - valid, metric printed
  1 - evaluation failed
"""
import json
import sys

import torch
from datasets import load_from_disk

from bert import tokenizer, model
from system import System2


def mean_pool_texts(texts, device, max_len=192):
    """(N texts) -> (N, 1024) tensor via ModernBERT + attention-weighted mean."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        last_hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def mean_pool_pairwise(questions, choices_list, device, max_len=128, sources=None, task_prefixes=False, span_pooling=False, span_weight=0.5, span_mode="mean", asym_spans=False, asym_span_mode="default"):
    """Pairwise encoding: (b questions, 4 choices) -> (b, 4, 1024) or (b, 5, 4, 1024)."""
    b = len(questions)
    ceiling = getattr(model.config, "max_position_embeddings", 8192)
    if max_len > ceiling:
        print(f"[eval warning] Requested max_len={max_len} exceeds architectural ceiling ({ceiling}); clamped to {ceiling}.", file=sys.stderr)
        max_len = ceiling
    if task_prefixes:
        flat_texts = [
            f"{'[Task: Scientific Facts] ' if (sources and sources[idx] == 'openbookqa') else '[Task: Causal Reasoning] '}Question: {questions[idx]} Answer: {choices_list[idx][c]}"
            for idx in range(b)
            for c in range(4)
        ]
    else:
        flat_texts = [
            f"Question: {questions[idx]} Answer: {choices_list[idx][c]}"
            for idx in range(b)
            for c in range(4)
        ]
    inputs = tokenizer(flat_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        last_hidden = model(**inputs).last_hidden_state
        if asym_spans:
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
            elif asym_span_mode == "choice_forward_v2":
                v0 = holistic
                v1 = 0.40 * q_vec + 0.60 * a_vec
                v2 = 0.20 * q_vec + 0.80 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = 0.15 * q_vec + 0.85 * a_vec
            elif asym_span_mode == "balanced_spectrum":
                v0 = holistic
                v1 = 0.40 * q_vec + 0.60 * a_vec
                v2 = 0.20 * q_vec + 0.80 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = 0.35 * q_vec + 0.65 * a_vec
            elif asym_span_mode == "constraint_spectrum":
                v0 = holistic
                v1 = 0.70 * q_vec + 0.30 * a_vec
                v2 = 0.20 * q_vec + 0.80 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = 0.35 * q_vec + 0.65 * a_vec
            else:
                # 5 asymmetric ThoughtHead perspectives (default):
                v0 = holistic
                v1 = 0.75 * q_vec + 0.25 * a_vec
                v2 = 0.25 * q_vec + 0.75 * a_vec
                v3 = 0.50 * q_vec + 0.50 * a_vec
                v4 = holistic

            multi_head_vecs = torch.stack([v0, v1, v2, v3, v4], dim=1)  # (b*4, 5, 1024)
            multi_head_vecs = multi_head_vecs.view(b, 4, 5, -1).permute(0, 2, 1, 3)  # (b, 5, 4, 1024)
            return multi_head_vecs
        elif span_pooling:
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
            if span_mode == "attn":
                H_norm = F.normalize(last_hidden, p=2, dim=-1)
                q_norm = F.normalize(q_vec, p=2, dim=-1)
                cos_sim = (H_norm * q_norm.unsqueeze(1)).sum(dim=-1)
                scores = (cos_sim / 0.1).masked_fill(~a_mask, -1e9)
                attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
                a_vec = (last_hidden * attn_weights).sum(dim=1)
            else:
                a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)
            pooled = (1.0 - span_weight) * q_vec + span_weight * a_vec
        else:
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.view(b, 4, -1)


def evaluate(checkpoint_path, batch=32, max_samples=None, device=None, data_dir="./data/arc", ensemble=False, head_weights=None, ensemble_temp=None, head_norm_logits=None, veto_blend=None, margin_gamma=None, decision_conf=None, decision_conf_mode=None, decision_conf_tau=None, decision_conf_alpha=None, max_len=None, dynamic_len=False):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    bert_name = state.get("bert_model", "answerdotai/ModernBERT-large") if isinstance(state, dict) else "answerdotai/ModernBERT-large"
    from bert import get_bert
    global tokenizer, model
    tokenizer, model = get_bert(bert_name)
    model.to(device)
    ds = load_from_disk(data_dir)
    test = ds["test"]
    n = max_samples or len(test)
    cols = test[:n]
    labels = torch.tensor(cols["label"], device=device)
    has_choices = "choices" in cols and cols["choices"] is not None
    if isinstance(state, dict) and "sys2" in state:
        sys2_state = state["sys2"]
        if "lora" in state:
            from lora import apply_lora
            cfg = state.get("lora_config", {"r": 16, "alpha": 16, "num_layers": 4})
            train_norms = cfg.get("train_norms", False) or ("lora_norms" in state)
            apply_lora(model, r=cfg.get("r", 16), alpha=cfg.get("alpha", 16), num_layers=cfg.get("num_layers", 4), train_norms=train_norms)
            model.load_state_dict(state["lora"], strict=False)
            if "lora_norms" in state:
                model.load_state_dict(state["lora_norms"], strict=False)
        elif "encoder_top2" in state:
            model.load_state_dict(state["encoder_top2"], strict=False)
    else:
        sys2_state = state

    model.to(device)
    model.eval()

    num_heads = len({k.split(".")[1] for k in sys2_state.keys() if k.startswith("heads.")})
    has_diff = "heads.0.hypothesis_head.0.weight" in sys2_state
    has_recurrence = "feedback_proj.weight" in sys2_state
    if has_diff:
        is_pairwise = True
        num_classes = 1
    else:
        hypo_w = sys2_state.get("heads.0.hypothesis_head.weight")
        is_pairwise = (hypo_w is not None and hypo_w.shape[0] == 1)
        num_classes = 1 if is_pairwise else 4

    is_mote = "router.gate.weight" in sys2_state
    mote_mode = state.get("mote_mode", "supervised") if isinstance(state, dict) else "supervised"
    if is_mote:
        from mote import MoTESystem2
        has_soft = any("soft_centroids" in k for k in sys2_state.keys())
        sys2 = MoTESystem2(num_experts=num_heads, diff_features=has_diff, soft_centroid=has_soft, mote_mode=mote_mode)
    else:
        layer_indices = [int(k.split(".")[4]) for k in sys2_state.keys() if "layers.layers." in k]
        head_layers = max(layer_indices) + 1 if layer_indices else 2
        kwargs = {"num_heads": num_heads, "num_classes": num_classes, "pairwise": is_pairwise, "diff_features": has_diff, "head_layers": head_layers}
        if "judger.log_temp" in sys2_state:
            kwargs["learnable_temp"] = True
        if isinstance(state, dict) and state.get("centered_conf", False):
            kwargs["centered_conf"] = True
        if isinstance(state, dict) and state.get("uniform_ensemble", False):
            kwargs["uniform_ensemble"] = True
        if ensemble_temp is not None:
            kwargs["ensemble_temp"] = ensemble_temp
        elif isinstance(state, dict) and "ensemble_temp" in state:
            kwargs["ensemble_temp"] = state["ensemble_temp"]
        if "judger.margin_norm.weight" in sys2_state:
            m_dim = sys2_state["judger.margin_norm.weight"].shape[0]
            kwargs["margin_dim"] = m_dim
            if m_dim == 5 or (isinstance(state, dict) and state.get("consensus_features", False)):
                kwargs["consensus_features"] = True
            else:
                kwargs["margin_features"] = True
        elif isinstance(state, dict) and state.get("consensus_features", False):
            kwargs["consensus_features"] = True
            kwargs["margin_dim"] = 5
        elif isinstance(state, dict) and state.get("margin_features", False):
            kwargs["margin_features"] = True
            kwargs["margin_dim"] = 3
        if isinstance(state, dict) and state.get("judger_win_emb", False):
            kwargs["judger_win_emb"] = True
        if isinstance(state, dict) and state.get("judger_soft_emb", False):
            kwargs["judger_soft_emb"] = True
        if isinstance(state, dict) and "judger_soft_tau" in state:
            kwargs["judger_soft_tau"] = state["judger_soft_tau"]
        if margin_gamma is not None:
            kwargs["margin_gamma"] = margin_gamma
        elif isinstance(state, dict) and "margin_gamma" in state:
            kwargs["margin_gamma"] = state["margin_gamma"]
        if isinstance(state, dict) and "prob_margin_gamma" in state:
            kwargs["prob_margin_gamma"] = state["prob_margin_gamma"]
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
        if veto_blend is not None:
            kwargs["veto_blend"] = veto_blend
        elif isinstance(state, dict) and "veto_blend" in state:
            kwargs["veto_blend"] = state["veto_blend"]
        if isinstance(state, dict) and "dual_stream_contest" in state:
            kwargs["dual_stream_contest"] = state["dual_stream_contest"]
        if isinstance(state, dict) and "dual_stream_blend" in state:
            kwargs["dual_stream_blend"] = state["dual_stream_blend"]
        if isinstance(state, dict) and "dual_stream_alpha" in state:
            kwargs["dual_stream_alpha"] = state["dual_stream_alpha"]
        if head_weights is not None:
            kwargs["head_weights"] = head_weights
        elif isinstance(state, dict) and "head_weights" in state:
            kwargs["head_weights"] = state["head_weights"]
        if isinstance(state, dict) and state.get("head_center_logits", False):
            kwargs["head_center_logits"] = True
        if head_norm_logits is not None:
            kwargs["head_norm_logits"] = head_norm_logits
        elif isinstance(state, dict) and state.get("head_norm_logits", False):
            kwargs["head_norm_logits"] = True
        if (isinstance(state, dict) and state.get("head_input_norm", False)) or ("heads.0.input_norm.weight" in sys2_state):
            kwargs["head_input_norm"] = True
        if decision_conf is not None:
            kwargs["decision_conf"] = decision_conf
        elif isinstance(state, dict) and state.get("decision_conf", False):
            kwargs["decision_conf"] = True
        if decision_conf_mode is not None:
            kwargs["decision_conf_mode"] = decision_conf_mode
        elif isinstance(state, dict) and "decision_conf_mode" in state:
            kwargs["decision_conf_mode"] = state["decision_conf_mode"]
        if decision_conf_tau is not None:
            kwargs["decision_conf_tau"] = decision_conf_tau
        elif isinstance(state, dict) and "decision_conf_tau" in state:
            kwargs["decision_conf_tau"] = state["decision_conf_tau"]
        if decision_conf_alpha is not None:
            kwargs["decision_conf_alpha"] = decision_conf_alpha
        elif isinstance(state, dict) and "decision_conf_alpha" in state:
            kwargs["decision_conf_alpha"] = state["decision_conf_alpha"]
        if (isinstance(state, dict) and state.get("decision_conf_learnable", False)) or ("log_decision_tau" in sys2_state):
            kwargs["decision_conf_learnable"] = True
        if "judger.score.0.weight" in sys2_state:
            kwargs["judger_mlp_dim"] = sys2_state["judger.score.0.weight"].shape[0]
        elif isinstance(state, dict) and "judger_mlp_dim" in state:
            kwargs["judger_mlp_dim"] = state["judger_mlp_dim"]
        if isinstance(state, dict) and "judger_num_heads" in state:
            kwargs["judger_num_heads"] = state["judger_num_heads"]
        if isinstance(state, dict) and "judger_dropout" in state:
            kwargs["judger_dropout"] = state["judger_dropout"]
        import inspect
        if "recurrence" in inspect.signature(System2.__init__).parameters:
            kwargs["recurrence"] = has_recurrence
        sys2 = System2(**kwargs)
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()

    task_prefixes = isinstance(state, dict) and state.get("task_prefixes", False)
    ceiling = getattr(model.config, "max_position_embeddings", 8192)
    if max_len is not None:
        eval_max_len = min(max_len, ceiling)
    elif dynamic_len:
        eval_max_len = ceiling
    else:
        eval_max_len = min(state.get("max_len", 128) if isinstance(state, dict) else 128, ceiling)
    ensemble_heads = (isinstance(state, dict) and state.get("ensemble_heads", False)) or ensemble
    span_pooling = state.get("span_pooling", False) if isinstance(state, dict) else False
    span_weight = state.get("span_weight", 0.5) if isinstance(state, dict) else 0.5
    span_mode = state.get("span_mode", "mean") if isinstance(state, dict) else "mean"
    asym_spans = state.get("asym_spans", False) if isinstance(state, dict) else False
    asym_span_mode = state.get("asym_span_mode", "default") if isinstance(state, dict) else "default"
    sources = cols.get("source") if hasattr(cols, "get") and "source" in cols else None

    correct = 0
    ece_sum = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            if has_choices and is_pairwise:
                src_slice = sources[i : i + b] if sources else None
                vecs = mean_pool_pairwise(
                    cols["question"][i : i + b], cols["choices"][i : i + b],
                    device, max_len=eval_max_len, sources=src_slice,
                    task_prefixes=task_prefixes, span_pooling=span_pooling,
                    span_weight=span_weight, span_mode=span_mode,
                    asym_spans=asym_spans, asym_span_mode=asym_span_mode,
                )
            else:
                vecs = mean_pool_texts(cols["text"][i : i + b], device)

            if is_mote:
                domain_ids = torch.zeros(b, dtype=torch.long, device=device)
                if src_slice:
                    domain_ids = torch.tensor([1 if s == "openbookqa" else 0 for s in src_slice], device=device)
                best, conf, pruned, confs, logits, emb, masked_scores, dense_gates, aux_loss, ens_logits, active_mask = sys2(
                    vecs, domain_ids=domain_ids, mode=mote_mode
                )
                if mote_mode == "ensemble":
                    win_logits = ens_logits
                else:
                    win_logits = logits[torch.arange(b), best]
            else:
                out = sys2(vecs)
                best, conf, pruned, confs, logits, emb, scores = out[:7]
                if ensemble_heads:
                    win_logits = out[7] if len(out) > 7 else torch.einsum('bn,bnc->bc', torch.softmax(scores, dim=-1), logits)
                else:
                    win_logits = logits[torch.arange(b), best]  # (b, 4)

            preds = win_logits.max(dim=1).indices
            correct += (preds == labels[i : i + b]).sum().item()
            ece_sum += ((conf - (preds == labels[i : i + b]).float()) ** 2).sum().item()

    return {"accuracy": correct / n, "brier": ece_sum / n}


if __name__ == "__main__":
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("ckpt", nargs="?", default="./models/system2.pt")
        ap.add_argument("--max-samples", type=int, default=None)
        ap.add_argument("--data", default="./data/arc")
        ap.add_argument("--ensemble", action="store_true", help="Force ensemble evaluation over all heads")
        ap.add_argument("--head-weights", default=None, help="Comma-separated head weights e.g. 1.0,1.0,1.2,1.0,0.4")
        ap.add_argument("--ensemble-temp", type=float, default=None, help="Deliberation arbitration temperature tau")
        ap.add_argument("--head-norm-logits", action="store_true", default=None, help="Force per-head logit normalization")
        ap.add_argument("--veto-blend", type=float, default=None, help="Veto blend weight")
        ap.add_argument("--margin-gamma", type=float, default=None, help="Margin gamma weight")
        ap.add_argument("--decision-conf", action="store_true", default=None, help="Force decision-level confidence")
        ap.add_argument("--decision-conf-mode", default=None, help="Confidence mode: softmax, margin, hybrid")
        ap.add_argument("--decision-conf-tau", type=float, default=None, help="Decision confidence temperature")
        ap.add_argument("--decision-conf-alpha", type=float, default=None, help="Hybrid confidence alpha blend")
        ap.add_argument("--max-len", type=int, default=None, help="Override evaluation max_len")
        ap.add_argument("--dynamic-len", action="store_true", help="Dynamically scale context window up to architectural ceiling (8192)")
        args = ap.parse_args()
        ckpt = args.ckpt
        hw = [float(x) for x in args.head_weights.split(",")] if args.head_weights else None
        m = evaluate(ckpt, max_samples=args.max_samples, data_dir=args.data, ensemble=args.ensemble, head_weights=hw, ensemble_temp=args.ensemble_temp, head_norm_logits=args.head_norm_logits, veto_blend=args.veto_blend, margin_gamma=args.margin_gamma, decision_conf=args.decision_conf, decision_conf_mode=args.decision_conf_mode, decision_conf_tau=args.decision_conf_tau, decision_conf_alpha=args.decision_conf_alpha, max_len=args.max_len, dynamic_len=args.dynamic_len)
        metric = 0.5 * m["accuracy"] + 0.5 * (1 - m["brier"])
        print(json.dumps({"metric": round(metric, 4), "accuracy": round(m["accuracy"], 4), "brier": round(m["brier"], 4)}))
    except Exception as e:
        print(f"eval failed: {e}", file=sys.stderr)
        sys.exit(1)
