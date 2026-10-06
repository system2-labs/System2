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


def mean_pool_pairwise(questions, choices_list, device, max_len=128, sources=None, task_prefixes=False):
    """Pairwise encoding: (b questions, 4 choices) -> (b, 4, 1024)."""
    b = len(questions)
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
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    return pooled.view(b, 4, -1)


def evaluate(checkpoint_path, batch=32, max_samples=None, device=None, data_dir="./data/arc", ensemble=False):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    ds = load_from_disk(data_dir)
    test = ds["test"]
    n = max_samples or len(test)
    cols = test[:n]
    labels = torch.tensor(cols["label"], device=device)
    has_choices = "choices" in cols and cols["choices"] is not None

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
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
        import inspect
        if "recurrence" in inspect.signature(System2.__init__).parameters:
            kwargs["recurrence"] = has_recurrence
        sys2 = System2(**kwargs)
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()

    task_prefixes = isinstance(state, dict) and state.get("task_prefixes", False)
    eval_max_len = state.get("max_len", 128) if isinstance(state, dict) else 128
    ensemble_heads = (isinstance(state, dict) and state.get("ensemble_heads", False)) or ensemble
    sources = cols.get("source") if hasattr(cols, "get") and "source" in cols else None

    correct = 0
    ece_sum = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            if has_choices and is_pairwise:
                src_slice = sources[i : i + b] if sources else None
                vecs = mean_pool_pairwise(cols["question"][i : i + b], cols["choices"][i : i + b], device, max_len=eval_max_len, sources=src_slice, task_prefixes=task_prefixes)
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
        args = ap.parse_args()
        ckpt = args.ckpt
        m = evaluate(ckpt, max_samples=args.max_samples, data_dir=args.data, ensemble=args.ensemble)
        metric = 0.5 * m["accuracy"] + 0.5 * (1 - m["brier"])
        print(json.dumps({"metric": round(metric, 4), "accuracy": round(m["accuracy"], 4), "brier": round(m["brier"], 4)}))
    except Exception as e:
        print(f"eval failed: {e}", file=sys.stderr)
        sys.exit(1)
