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
    "use_mote": False,
    "mote_mode": "arc_soft",  # "supervised" (Path 1), "ensemble" (Path 2), "arc_soft" (Path 3)
    "num_experts": 4,
    "mote_top_k": 2,
    "soft_centroid": True,
    "num_heads": 5,
    "head_layers": 2,
    "lr_heads": 0.001,
    "lr_router": 5e-4,
    "lr_soft_centroid": 1e-3,
    "lr_judger": 0.0003,
    "use_lora": True,
    "lora_r": 32,
    "lora_alpha": 32,
    "lora_layers": 8,  # adapt top 8 layers of ModernBERT via LoRA
    "lr_lora": 0.0002,
    "batch_size": 16,
    "max_len": 128,
    "device": "mps",   # "mps" = Apple GPU (~8x faster than cpu)
    "epochs": 5,
    "train_size": 4300,
    "lambda_div": 0.01,
    "lambda_judge": 0.1,
    "lambda_cal": 0.5,
    "lambda_judge_cal": 0.06,
    "weight_decay": 0.0,
    "warmup_ratio": 0.0,
    "oracle_judge_target": False,
    "detach_judger": True,
    "shuffle_per_epoch": False,
    "seed": 0,
    "label_smoothing": 0.15,
    "lambda_aux": 0.10,
    "data_dir": "./data/arc",
    "pairwise": True,
    "diff_features": True,
    "task_prefixes": False,
    "lora_train_norms": False,
    "lr_norm": 5e-5,
    "learnable_temp": False,
    "use_scheduler": True,
    "scheduler_min_lr": 1e-05,
    "soft_judge_targets": False,
    "soft_target_tau": 0.5,
    "ensemble_heads": True,
    "centered_conf": False,
    "judge_cal_smoothing": 0.3,
    "lambda_ens_ce": 0.0,
    "uniform_ensemble": False,
    "ensemble_temp": 1.8,
    "lambda_head_balance": 0.0,
    "margin_features": False,
    "span_pooling": False,
    "span_weight": 0.5,
    "span_mode": "mean",
    "lora_dropout": 0.0,
    "head_dropout": 0.0,
    "head_label_smoothing": 0.08,
    "margin_gamma": 0.0,
    "prob_ensemble": False,
    "head_prob_temp": 1.0,
    "head_norm_first": False,
    "judger_residual": False,
    "head_mlp_dim": 256,
    "head_norm_diff": False,
    "lambda_cand_margin": 0.0,
    "cand_margin": 1.0,
    "veto_blend": 0.0,
    "permute_choices": False,
    "asym_roles": False,
    "asym_spans": False,
}


def mean_pool_texts(texts, max_len, device, with_grad=False):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
    if with_grad:
        last_hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    else:
        with torch.no_grad():
            last_hidden = model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)


def mean_pool_pairwise(questions, choices_list, max_len, device, with_grad=False, sources=None, task_prefixes=False, span_pooling=False, span_weight=0.5, span_mode="mean", asym_spans=False):
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
    if with_grad:
        last_hidden = model(**inputs).last_hidden_state
    else:
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

        # 5 asymmetric ThoughtHead perspectives:
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


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


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
    diff_features = HYPERPARAMS.get("diff_features", False) and is_pairwise

    set_seed(HYPERPARAMS.get("seed", 42))
    device = HYPERPARAMS["device"]

    use_lora = HYPERPARAMS.get("use_lora", False)
    train_norms = HYPERPARAMS.get("lora_train_norms", False)
    unfreeze_layers = HYPERPARAMS.get("unfreeze_encoder_layers", 0)
    for p in model.parameters():
        p.requires_grad = False

    encoder_params = []
    norm_params = []
    if use_lora:
        from lora import apply_lora
        r = HYPERPARAMS.get("lora_r", 16)
        alpha = HYPERPARAMS.get("lora_alpha", 16)
        num_layers = HYPERPARAMS.get("lora_layers", 4)
        res = apply_lora(model, r=r, alpha=alpha, num_layers=num_layers, train_norms=train_norms, dropout=HYPERPARAMS.get("lora_dropout", 0.0))
        if train_norms:
            encoder_params, norm_params = res
        else:
            encoder_params = res
            norm_params = []
        model.train()
    elif unfreeze_layers > 0:
        for layer in model.layers[-unfreeze_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
                encoder_params.append(p)
        for p in model.final_norm.parameters():
            p.requires_grad = True
            encoder_params.append(p)
        model.train()
    else:
        model.eval()

    model.to(device)

    use_mote = HYPERPARAMS.get("use_mote", False)
    mote_mode = HYPERPARAMS.get("mote_mode", "supervised")
    if use_mote:
        from mote import MoTESystem2
        sys2 = MoTESystem2(
            num_experts=HYPERPARAMS.get("num_experts", 4),
            top_k=HYPERPARAMS.get("mote_top_k", 2),
            hidden_size=1024,
            head_dim=512,
            diff_features=diff_features,
            soft_centroid=HYPERPARAMS.get("soft_centroid", True),
            mote_mode=mote_mode,
        )
    else:
        sys2 = System2(
            num_heads=HYPERPARAMS["num_heads"],
            num_classes=1 if is_pairwise else 4,
            pairwise=is_pairwise,
            diff_features=diff_features,
            learnable_temp=HYPERPARAMS.get("learnable_temp", False),
            head_layers=HYPERPARAMS.get("head_layers", 2),
            centered_conf=HYPERPARAMS.get("centered_conf", False),
            uniform_ensemble=HYPERPARAMS.get("uniform_ensemble", False),
            ensemble_temp=HYPERPARAMS.get("ensemble_temp", 1.0),
            margin_features=HYPERPARAMS.get("margin_features", False),
            head_dropout=HYPERPARAMS.get("head_dropout", 0.0),
            margin_gamma=HYPERPARAMS.get("margin_gamma", 0.0),
            prob_ensemble=HYPERPARAMS.get("prob_ensemble", False),
            head_prob_temp=HYPERPARAMS.get("head_prob_temp", 1.0),
            head_norm_first=HYPERPARAMS.get("head_norm_first", False),
            judger_residual=HYPERPARAMS.get("judger_residual", False),
            head_mlp_dim=HYPERPARAMS.get("head_mlp_dim", 256),
            head_norm_diff=HYPERPARAMS.get("head_norm_diff", False),
            veto_blend=HYPERPARAMS.get("veto_blend", 0.0),
        )
    sys2.to(device)
    sys2.train()

    if use_mote:
        expert_params = [p for h in sys2.heads for p in h.parameters()]
        soft_params = [p for sc in sys2.soft_centroids for p in sc.parameters()] if sys2.soft_centroids else []
        router_params = list(sys2.router.parameters())
        judger_params = list(sys2.judger.parameters())
        for p in expert_params + router_params + soft_params + judger_params:
            assert p.requires_grad
        param_groups = [
            {"params": expert_params, "lr": HYPERPARAMS["lr_heads"]},
            {"params": router_params, "lr": HYPERPARAMS.get("lr_router", 5e-4)},
            {"params": judger_params, "lr": HYPERPARAMS["lr_judger"]},
        ]
        if soft_params:
            param_groups.append({"params": soft_params, "lr": HYPERPARAMS.get("lr_soft_centroid", 1e-3)})
    else:
        head_params = [p for h in sys2.heads for p in h.parameters()]
        judger_params = list(sys2.judger.parameters())
        for p in head_params + judger_params:
            assert p.requires_grad
        param_groups = [
            {"params": head_params, "lr": HYPERPARAMS["lr_heads"]},
            {"params": judger_params, "lr": HYPERPARAMS["lr_judger"]},
        ]

    if encoder_params:
        enc_lr = HYPERPARAMS.get("lr_lora", 2e-4) if use_lora else HYPERPARAMS.get("lr_encoder", 1e-5)
        param_groups.append({"params": encoder_params, "lr": enc_lr})
    if norm_params:
        norm_lr = HYPERPARAMS.get("lr_norm", 5e-5)
        param_groups.append({"params": norm_params, "lr": norm_lr})
    weight_decay = HYPERPARAMS.get("weight_decay", 0.0)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    use_scheduler = HYPERPARAMS.get("use_scheduler", False)
    if use_scheduler:
        total_steps = HYPERPARAMS["epochs"] * ((len(labels) + HYPERPARAMS["batch_size"] - 1) // HYPERPARAMS["batch_size"])
        warmup_ratio = HYPERPARAMS.get("warmup_ratio", 0.0)
        warmup_steps = int(total_steps * warmup_ratio)
        min_lr = HYPERPARAMS.get("scheduler_min_lr", 1e-5)
        if warmup_steps > 0:
            s1 = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
            )
            s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=min_lr
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[s1, s2], milestones=[warmup_steps]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=total_steps, eta_min=min_lr
            )
    else:
        scheduler = None

    task_prefixes = HYPERPARAMS.get("task_prefixes", False)
    sources = cols.get("source") if "source" in cols else None

    bs = HYPERPARAMS["batch_size"]
    for epoch in range(HYPERPARAMS["epochs"]):
        total_ce = total_cal = total_judge = total_div = 0.0
        n_batches = 0
        if HYPERPARAMS.get("shuffle_per_epoch", True):
            perm = torch.randperm(len(labels)).tolist()
        else:
            perm = list(range(len(labels)))

        for i in range(0, len(labels), bs):
            n_batches += 1
            batch_idx = perm[i : i + bs]
            b = len(batch_idx)
            with_grad = use_lora or (unfreeze_layers > 0)
            if is_pairwise:
                src_slice = [sources[idx] for idx in batch_idx] if sources else None
                q_slice = [cols["question"][idx] for idx in batch_idx]
                c_slice = [cols["choices"][idx] for idx in batch_idx]
                lab_list = [labels[idx] for idx in batch_idx]
                if HYPERPARAMS.get("permute_choices", False):
                    new_c_slice = []
                    new_lab_list = []
                    for idx_in_b in range(b):
                        orig_choices = c_slice[idx_in_b]
                        orig_lab = lab_list[idx_in_b]
                        p_order = torch.randperm(len(orig_choices)).tolist()
                        permuted_choices = [orig_choices[p] for p in p_order]
                        new_lab = p_order.index(orig_lab)
                        new_c_slice.append(permuted_choices)
                        new_lab_list.append(new_lab)
                    c_slice = new_c_slice
                    lab_list = new_lab_list
                vecs = mean_pool_pairwise(
                    q_slice, c_slice,
                    HYPERPARAMS["max_len"], device, with_grad=with_grad,
                    sources=src_slice, task_prefixes=task_prefixes,
                    span_pooling=HYPERPARAMS.get("span_pooling", False),
                    span_weight=HYPERPARAMS.get("span_weight", 0.5),
                    span_mode=HYPERPARAMS.get("span_mode", "mean"),
                    asym_spans=HYPERPARAMS.get("asym_spans", False),
                )
                lab = torch.tensor(lab_list, device=device)  # (b,)
            else:
                t_slice = [cols["text"][idx] for idx in batch_idx]
                vecs = mean_pool_texts(t_slice, HYPERPARAMS["max_len"], device, with_grad=with_grad)
                lab = torch.tensor([labels[idx] for idx in batch_idx], device=device)  # (b,)

            if use_mote:
                domain_ids = torch.zeros(b, dtype=torch.long, device=device)
                if src_slice:
                    domain_ids = torch.tensor([1 if s == "openbookqa" else 0 for s in src_slice], device=device)

                best, conf, pruned, confs, logits, emb, scores, dense_gates, aux_loss, ens_logits, active_mask = sys2(
                    vecs, domain_ids=domain_ids, mode=mote_mode, detach_judger=HYPERPARAMS.get("detach_judger", False)
                )

                if mote_mode == "ensemble":
                    ce_loss = torch.nn.functional.cross_entropy(ens_logits, lab)
                    win_logits = ens_logits
                elif mote_mode == "supervised":
                    ce_per_expert = torch.stack([
                        torch.nn.functional.cross_entropy(h, lab, reduction='none')
                        for h in logits.unbind(1)
                    ], dim=1)  # (b, E)
                    act_float = active_mask.float()
                    ce_loss = (ce_per_expert * (act_float / act_float.sum(dim=1, keepdim=True).clamp(min=1.0))).sum(dim=1).mean()
                    win_logits = logits[torch.arange(b), best]
                else:  # arc_soft
                    ce = torch.stack([torch.nn.functional.cross_entropy(h, lab) for h in logits.unbind(1)])
                    ce_loss = ce.mean()
                    win_logits = logits[torch.arange(b), best]
                task_loss = ce_loss
            else:
                out = sys2(
                    vecs, detach_judger=HYPERPARAMS.get("detach_judger", False)
                )
                best, conf, pruned, confs, logits, emb, scores = out[:7]
                if HYPERPARAMS.get("asym_roles", False):
                    ls_list = HYPERPARAMS.get("asym_ls", [0.10, 0.10, 0.04, 0.04, 0.08])
                    ce = torch.stack([
                        torch.nn.functional.cross_entropy(h, lab, label_smoothing=ls_list[idx])
                        for idx, h in enumerate(logits.unbind(1))
                    ])
                else:
                    h_ls = HYPERPARAMS.get("head_label_smoothing", 0.0)
                    ce = torch.stack([torch.nn.functional.cross_entropy(h, lab, label_smoothing=h_ls) for h in logits.unbind(1)])
                ce_loss = ce.mean()
                if HYPERPARAMS.get("ensemble_heads", False):
                    win_logits = out[7] if len(out) > 7 else torch.einsum('bn,bnc->bc', torch.softmax(scores, dim=-1), logits)
                    lambda_ens_ce = HYPERPARAMS.get("lambda_ens_ce", 0.0)
                    if lambda_ens_ce > 0.0:
                        ens_ce = torch.nn.functional.cross_entropy(win_logits, lab)
                        task_loss = (1.0 - lambda_ens_ce) * ce_loss + lambda_ens_ce * ens_ce
                    else:
                        task_loss = ce_loss
                else:
                    win_logits = logits[torch.arange(b), best]
                    task_loss = ce_loss
                aux_loss = torch.tensor(0.0, device=device)

            # 1b. Contrastive Candidate Margin Loss
            lambda_cand_margin = HYPERPARAMS.get("lambda_cand_margin", 0.0)
            if lambda_cand_margin > 0.0 and logits.dim() == 3:
                c_margin = HYPERPARAMS.get("cand_margin", 1.0)
                m_losses = []
                margin_weights = HYPERPARAMS.get("asym_margin_weights", [0.0, 0.0, 1.5, 1.5, 1.0]) if HYPERPARAMS.get("asym_roles", False) else [1.0] * logits.size(1)
                for idx, h in enumerate(logits.unbind(1)):
                    w_m = margin_weights[idx]
                    if w_m <= 0.0:
                        continue
                    correct_scores = h.gather(1, lab.unsqueeze(1))
                    diffs = correct_scores - h
                    loss_m = torch.clamp(c_margin - diffs, min=0.0)
                    mask = torch.ones_like(loss_m).scatter_(1, lab.unsqueeze(1), 0.0)
                    m_head = (loss_m * mask).sum(dim=1) / (loss_m.size(1) - 1)
                    m_losses.append(w_m * m_head)
                cand_margin_loss = torch.stack(m_losses).mean() if m_losses else torch.tensor(0.0, device=device)
            else:
                cand_margin_loss = torch.tensor(0.0, device=device)

            # --- losses (all computed, then ONE combined backward) ---

            # 2. Head Calibration: head's confidence -> 1[that head is correct]
            with torch.no_grad():
                head_correct = (logits.max(dim=2).indices == lab.unsqueeze(1)).float()  # (b, N)
            cal_per_head = torch.nn.functional.binary_cross_entropy(confs, head_correct, reduction='none')
            if use_mote and mote_mode == "supervised":
                cal_loss = (cal_per_head * active_mask.float()).sum() / active_mask.float().sum().clamp(min=1.0)
            else:
                cal_loss = cal_per_head.mean()

            # 3. Judger Selection: target is most confident among active heads or soft distillation
            if HYPERPARAMS.get("soft_judge_targets", False):
                with torch.no_grad():
                    head_preds = logits.max(dim=2).indices  # (b, N)
                    is_head_correct = (head_preds == lab.unsqueeze(1))  # (b, N) bool
                    ce_per_h = torch.stack([
                        torch.nn.functional.cross_entropy(h, lab, reduction='none')
                        for h in logits.unbind(1)
                    ], dim=1)  # (b, N)
                    tau = HYPERPARAMS.get("soft_target_tau", 0.5)
                    masked_scores = torch.where(is_head_correct, confs / tau, torch.full_like(confs, -1e9))
                    has_correct = is_head_correct.any(dim=1, keepdim=True)
                    fallback_scores = -ce_per_h / tau
                    target_logits = torch.where(has_correct, masked_scores, fallback_scores)
                    q = torch.softmax(target_logits, dim=-1)
                judge_loss = torch.nn.functional.cross_entropy(scores, q)
            elif use_mote:
                active_confs = confs.masked_fill(~active_mask, -1e9)
                target = active_confs.max(dim=1).indices
                judge_loss = torch.nn.functional.cross_entropy(scores, target)
            else:
                if HYPERPARAMS.get("oracle_judge_target", True):
                    with torch.no_grad():
                        # Mask non-correct heads: Judger must learn to select a head that got the question right!
                        masked_confs = torch.where(head_correct.bool(), confs, torch.full_like(confs, -1e9))
                        has_correct = head_correct.bool().any(dim=1)
                        target = torch.where(has_correct, masked_confs.max(dim=1).indices, confs.max(dim=1).indices)
                else:
                    target = confs.max(dim=1).indices
                ls = HYPERPARAMS.get("label_smoothing", 0.15)
                judge_loss = torch.nn.functional.cross_entropy(scores, target, label_smoothing=ls)

            # 4. Direct Judger Output Calibration: train winning conf against actual accuracy
            with torch.no_grad():
                preds = win_logits.max(dim=1).indices
                is_correct = (preds == lab).float()
                cal_ls = HYPERPARAMS.get("judge_cal_smoothing", 0.0)
                if cal_ls > 0.0:
                    target_cal = is_correct * (1.0 - 2 * cal_ls) + cal_ls
                else:
                    target_cal = is_correct
            judge_cal_loss = torch.nn.functional.binary_cross_entropy(conf, target_cal)

            # 5. Diversity: penalize pairwise cosine similarity between head embeddings
            N_heads = emb.size(1)
            if N_heads > 1 and HYPERPARAMS.get("lambda_div", 0) > 0:
                emb_norm = torch.nn.functional.normalize(emb, p=2, dim=2)
                sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))
                triu_idx = torch.triu_indices(N_heads, N_heads, offset=1, device=device)
                div_loss = sim_matrix[:, triu_idx[0], triu_idx[1]].mean()
            else:
                div_loss = torch.tensor(0.0, device=device)

            # 6. Judger Head Load-Balancing: penalize collapse into a single dominant head across batch
            lambda_balance = HYPERPARAMS.get("lambda_head_balance", 0.0)
            if lambda_balance > 0.0:
                w_judger = torch.softmax(scores, dim=-1)  # (B, N)
                w_mean = w_judger.mean(dim=0)             # (N,)
                balance_loss = (w_mean * torch.log(w_mean * N_heads + 1e-8)).sum()
            else:
                balance_loss = torch.tensor(0.0, device=device)

            # Weighted single loss -> single backward keeps the shared graph valid.
            aux_weight = HYPERPARAMS.get("lambda_aux", 0.10) if (use_mote and mote_mode in ["supervised", "ensemble"]) else 0.0
            lambda_cal = HYPERPARAMS.get("lambda_cal", 0.5)
            lambda_judge = HYPERPARAMS.get("lambda_judge", 0.1)
            lambda_judge_cal = HYPERPARAMS.get("lambda_judge_cal", 0.5)
            lambda_div = HYPERPARAMS.get("lambda_div", 0.01)
            total_loss = (
                task_loss
                + lambda_cal * cal_loss
                + lambda_judge * judge_loss
                + lambda_judge_cal * judge_cal_loss
                + lambda_div * div_loss
                + lambda_balance * balance_loss
                + lambda_cand_margin * cand_margin_loss
                + aux_weight * aux_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            total_ce += task_loss.item()
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
    save_max_len = HYPERPARAMS.get("max_len", 128)
    save_dict = {
        "sys2": sys2.state_dict(),
        "task_prefixes": task_prefixes,
        "use_mote": use_mote,
        "mote_mode": mote_mode if use_mote else "baseline",
        "max_len": save_max_len,
        "ensemble_heads": HYPERPARAMS.get("ensemble_heads", False),
        "centered_conf": HYPERPARAMS.get("centered_conf", False),
        "judge_cal_smoothing": HYPERPARAMS.get("judge_cal_smoothing", 0.0),
        "lambda_ens_ce": HYPERPARAMS.get("lambda_ens_ce", 0.0),
        "uniform_ensemble": HYPERPARAMS.get("uniform_ensemble", False),
        "ensemble_temp": HYPERPARAMS.get("ensemble_temp", 1.0),
        "lambda_head_balance": HYPERPARAMS.get("lambda_head_balance", 0.0),
        "margin_features": HYPERPARAMS.get("margin_features", False),
        "span_pooling": HYPERPARAMS.get("span_pooling", False),
        "span_weight": HYPERPARAMS.get("span_weight", 0.5),
        "span_mode": HYPERPARAMS.get("span_mode", "mean"),
        "lora_dropout": HYPERPARAMS.get("lora_dropout", 0.0),
        "head_dropout": HYPERPARAMS.get("head_dropout", 0.0),
        "head_label_smoothing": HYPERPARAMS.get("head_label_smoothing", 0.0),
        "margin_gamma": HYPERPARAMS.get("margin_gamma", 0.0),
        "prob_ensemble": HYPERPARAMS.get("prob_ensemble", False),
        "head_prob_temp": HYPERPARAMS.get("head_prob_temp", 1.0),
        "head_norm_first": HYPERPARAMS.get("head_norm_first", False),
        "judger_residual": HYPERPARAMS.get("judger_residual", False),
        "head_mlp_dim": HYPERPARAMS.get("head_mlp_dim", 256),
        "head_norm_diff": HYPERPARAMS.get("head_norm_diff", False),
        "veto_blend": HYPERPARAMS.get("veto_blend", 0.0),
        "asym_spans": HYPERPARAMS.get("asym_spans", False),
    }
    if use_lora:
        save_dict["lora"] = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        save_dict["lora_config"] = {
            "r": HYPERPARAMS.get("lora_r", 32),
            "alpha": HYPERPARAMS.get("lora_alpha", 32),
            "num_layers": HYPERPARAMS.get("lora_layers", 8),
            "train_norms": train_norms,
        }
        if train_norms:
            save_dict["lora_norms"] = {
                k: v.cpu() for k, v in model.state_dict().items()
                if ("attn_norm" in k or "mlp_norm" in k or "final_norm" in k)
            }
    elif unfreeze_layers > 0:
        top_keys = [f"layers.{28 - i - 1}" for i in range(unfreeze_layers)] + ["final_norm"]
        save_dict["encoder_top2"] = {
            k: v.cpu() for k, v in model.state_dict().items()
            if any(tk in k for tk in top_keys)
        }
    torch.save(save_dict, args.save)
    print(f"saved {args.save}")


if __name__ == "__main__":
    main()
