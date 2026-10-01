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


def mean_pool_pairwise(questions, choices_list, device, max_len=128):
    """Pairwise encoding: (b questions, 4 choices) -> (b, 4, 1024)."""
    b = len(questions)
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


def evaluate(checkpoint_path, batch=32, max_samples=None, device=None, data_dir="./data/arc"):
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
        if "encoder_top2" in state:
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

    kwargs = {"num_heads": num_heads, "num_classes": num_classes, "pairwise": is_pairwise, "diff_features": has_diff}
    import inspect
    if "recurrence" in inspect.signature(System2.__init__).parameters:
        kwargs["recurrence"] = has_recurrence
    sys2 = System2(**kwargs)
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()

    correct = 0
    ece_sum = 0.0
    with torch.no_grad():
        for i in range(0, n, batch):
            b = min(batch, n - i)
            if has_choices and is_pairwise:
                vecs = mean_pool_pairwise(cols["question"][i : i + b], cols["choices"][i : i + b], device)
            else:
                vecs = mean_pool_texts(cols["text"][i : i + b], device)
            best, conf, pruned, confs, logits, emb, scores = sys2(vecs)
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
        args = ap.parse_args()
        ckpt = args.ckpt
        m = evaluate(ckpt, max_samples=args.max_samples, data_dir=args.data)
        metric = 0.5 * m["accuracy"] + 0.5 * (1 - m["brier"])
        print(json.dumps({"metric": round(metric, 4), "accuracy": round(m["accuracy"], 4), "brier": round(m["brier"], 4)}))
    except Exception as e:
        print(f"eval failed: {e}", file=sys.stderr)
        sys.exit(1)
