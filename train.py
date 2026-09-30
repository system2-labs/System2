"""train.py - SUBJECT for the System2 AG-News experiment.

Supervised v1 training of heads + judger (encoder frozen).

Losses:
  1. CE on every head's class logits   - all heads learn to classify
  2. BCE on every head's confidence    - calibrated: conf -> 1[head is correct]
  3. Judger: push the score of the best-correct head above the rest
     - "best" = highest-confidence head that predicted the true label

Why confidence-BCE (v1 stand-in for RLCD): RLCD's REINFORCE is a
single-sample unbiased estimator of the same objective; BCE on
1[correct] is its supervised approximation and is far cheaper on CPU.
Swap in REINFORCE later if calibration quality demands it.

The autoresearch loop (runner.sh) treats this file as the subject:
edit HYPERPARAMS below, run, the judge decides commit/revert.
"""
import argparse
import json

import torch
from datasets import load_from_disk

from bert import tokenizer, model
from system import System2

# --- TUNABLES (autoresearch subject: the loop edits this block) ---
HYPERPARAMS = {
    "num_heads": 5,
    "lr_heads": 1e-3,
    "lr_judger": 3e-4,
    "batch_size": 16,
    "max_len": 128,
    "device": "mps",   # "mps" = Apple GPU (~8x faster than cpu)
    "epochs": 3,
    "train_size": 3500,
    "lambda_div": 0.01,
    "data_dir": "./data/arc",
    "pairwise": True,
}


def mean_pool_texts(texts, max_len, device):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    with torch.no_grad():
        last_hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def mean_pool_pairwise(questions, choices_list, max_len, device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", default="./models/system2.pt")
    # CLI overrides win over HYPERPARAMS (used by the smoke wrapper)
    ap.add_argument("--train-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--data", default=None)
    args = ap.parse_args()
    if args.train_size is not None:
        HYPERPARAMS["train_size"] = args.train_size
    if args.epochs is not None:
        HYPERPARAMS["epochs"] = args.epochs
    if args.batch_size is not None:
        HYPERPARAMS["batch_size"] = args.batch_size

    data_dir = args.data or HYPERPARAMS.get("data_dir", "./data/arc")
    ds = load_from_disk(data_dir)
    # Deterministic shuffle before slicing
    train = ds["train"].shuffle(seed=0)
    n_samples = min(HYPERPARAMS["train_size"], len(train))
    cols = train[: n_samples]
    labels = cols["label"]
    is_pairwise = HYPERPARAMS.get("pairwise", False) and "choices" in cols and cols["choices"] is not None

    torch.manual_seed(0)
    device = HYPERPARAMS["device"]
    model.to(device)          # encoder (frozen, lives on the GPU)
    sys2 = System2(num_heads=HYPERPARAMS["num_heads"], num_classes=1 if is_pairwise else 4, pairwise=is_pairwise)
    sys2.to(device)           # heads + judger must share the encoder's device
    sys2.train()

    # Encoder params frozen; optimise heads + judger only
    head_params = [p for h in sys2.heads for p in h.parameters()]
    judger_params = list(sys2.judger.parameters())
    for p in head_params + judger_params:
        assert p.requires_grad
    # One optimizer, two param groups (different LRs). Why a single optimizer:
    # all three losses share one autograd graph (the judger reads the heads'
    # embeddings), so a single backward() is the only way to keep it valid.
    optimizer = torch.optim.Adam([
        {"params": head_params, "lr": HYPERPARAMS["lr_heads"]},
        {"params": judger_params, "lr": HYPERPARAMS["lr_judger"]},
    ])

    bs = HYPERPARAMS["batch_size"]
    for epoch in range(HYPERPARAMS["epochs"]):
        total_ce = total_cal = total_judge = total_div = 0.0
        n_batches = 0
        for i in range(0, len(labels), bs):
            n_batches += 1
            # CRITICAL: slice the batch first, derive everything from the slice.
            # (Fixed 2026-09-24: a fixed-size target mismatched the real batch
            # size on the final partial batch -> ValueError in cross_entropy.)
            b = min(bs, len(labels) - i)
            if is_pairwise:
                vecs = mean_pool_pairwise(cols["question"][i : i + b], cols["choices"][i : i + b], HYPERPARAMS["max_len"], device)
            else:
                vecs = mean_pool_texts(cols["text"][i : i + b], HYPERPARAMS["max_len"], device)
            lab = torch.tensor(labels[i : i + b], device=device)  # (b,)

            best, conf, pruned, confs, logits, emb, scores = sys2(vecs)

            # --- losses (all computed, then ONE combined backward) ---

            # 1. Classification: every head learns the true label
            ce = torch.stack([torch.nn.functional.cross_entropy(h, lab) for h in logits.unbind(1)])
            ce_loss = ce.mean()

            # 2. Head Calibration: head's confidence -> 1[that head is correct]
            with torch.no_grad():
                head_correct = (logits.max(dim=2).indices == lab.unsqueeze(1)).float()  # (b, N)
            cal_loss = torch.nn.functional.binary_cross_entropy(confs, head_correct)

            # 3. Judger Selection: cross-entropy with label smoothing to prevent logit explosion
            target = confs.max(dim=1).indices  # (b,)
            judge_loss = torch.nn.functional.cross_entropy(scores, target, label_smoothing=0.15)

            # 4. Direct Judger Output Calibration: train winning conf against actual accuracy
            with torch.no_grad():
                win_logits = logits[torch.arange(len(vecs)), best]
                preds = win_logits.max(dim=1).indices
                is_correct = (preds == lab).float()
            judge_cal_loss = torch.nn.functional.binary_cross_entropy(conf, is_correct)

            # 5. Diversity: penalize pairwise cosine similarity between head embeddings
            N_heads = emb.size(1)
            if N_heads > 1 and HYPERPARAMS.get("lambda_div", 0) > 0:
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=2)
                sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))
                triu_idx = torch.triu_indices(N_heads, N_heads, offset=1, device=device)
                div_loss = sim_matrix[:, triu_idx[0], triu_idx[1]].mean()
            else:
                div_loss = torch.tensor(0.0, device=device)

            # Weighted single loss -> single backward keeps the shared graph valid.
            total_loss = ce_loss + 0.5 * cal_loss + 0.1 * judge_loss + 0.5 * judge_cal_loss + HYPERPARAMS.get("lambda_div", 0) * div_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_ce += ce_loss.item()
            total_cal += cal_loss.item()
            total_judge += judge_loss.item()
            total_div += div_loss.item()

        print(
            f"epoch {epoch + 1}/{HYPERPARAMS['epochs']}  "
            f"ce={total_ce / n_batches:.3f} "
            f"cal={total_cal / n_batches:.3f} "
            f"judge={total_judge / n_batches:.3f} "
            f"div={total_div / n_batches:.3f}",
            flush=True,
        )

    import os
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save(sys2.state_dict(), args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
