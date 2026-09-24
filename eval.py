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


def mean_pool_texts(texts, device):
    """(N texts) -> (N, 1024) tensor via ModernBERT + attention-weighted mean."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    last_hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def evaluate(checkpoint_path, batch=64, max_samples=None, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    ds = load_from_disk("./data/ag_news")
    test = ds["test"]
    n = max_samples or len(test)
    # NOTE: test[:n] on a HF Dataset returns a dict of COLUMNS, not a list of
    # rows. Iterating it yields column-name strings. So read from the columns.
    cols = test[:n]
    texts = cols["text"]
    labels = torch.tensor(cols["label"])

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # num_heads must match the checkpoint (the autoresearch loop varies it).
    # Recover it from the saved state: count distinct head indices.
    num_heads = len({k.split(".")[1] for k in state.keys() if k.startswith("heads.")})
    sys2 = System2(num_heads=num_heads, num_classes=4)
    sys2.load_state_dict(state)
    sys2.eval()

    correct = 0
    ece_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            vecs = mean_pool_texts(texts[i : i + b], device) if (b := len(texts[i : i + batch])) else None
            if vecs is None:
                continue
            best, conf, pruned, confs, logits, emb, scores = sys2(vecs)
            # Winner's predicted class = argmax of winning head's logits
            win_logits = logits[torch.arange(len(vecs)), best]  # (b, 4)
            preds = win_logits.max(dim=1).indices
            correct += (preds == labels[i : i + b]).sum().item()
            # Calibration: winning confidence vs correctness
            ece_sum += ((conf - (preds == labels[i : i + b]).float()) ** 2).sum().item()

    n = len(texts)
    return {"accuracy": correct / n, "brier": ece_sum / n}


if __name__ == "__main__":
    try:
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("ckpt", nargs="?", default="./models/system2.pt")
        ap.add_argument("--max-samples", type=int, default=None)
        args = ap.parse_args()
        ckpt = args.ckpt
        m = evaluate(ckpt, max_samples=args.max_samples)
        metric = 0.5 * m["accuracy"] + 0.5 * (1 - m["brier"])
        print(json.dumps({"metric": round(metric, 4), "accuracy": round(m["accuracy"], 4), "brier": round(m["brier"], 4)}))
    except Exception as e:
        print(f"eval failed: {e}", file=sys.stderr)
        sys.exit(1)
