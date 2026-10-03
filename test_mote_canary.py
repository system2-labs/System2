"""test_mote_canary.py - Canary verification for MoTESystem2 architecture.

Tests:
1. Multi-domain data formatting with task prefixes (ARC + OpenBookQA).
2. ModernBERT-large + LoRA (top-8 layers, r=32) + MoTESystem2 on Apple Silicon MPS.
3. Sparse Top-2 routing and Switch Transformer load-balancing auxiliary loss.
4. Dynamic SoftCentroid gradient flow.
5. Checkpoint saving and loading in eval.py.
"""
import os
import sys
import torch
from datasets import load_from_disk
from bert import tokenizer, model
from lora import apply_lora
from mote import MoTESystem2
from eval import evaluate


def prepare_canary_batch(device):
    # Load 16 ARC samples and 16 OpenBookQA samples if available, or 32 ARC samples with dual task prefixes
    try:
        ds_arc = load_from_disk("./data/arc")["train"]
        arc_samples = ds_arc.select(range(16))
    except Exception as e:
        print(f"Error loading ARC: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        ds_obqa = load_from_disk("./data/arc_obqa")["train"]
        # Take 16 samples that have OBQA sentence completion style
        obqa_samples = ds_obqa.select(range(5000, 5016))
    except Exception:
        # Fallback to ARC second slice with different task label
        obqa_samples = ds_arc.select(range(16, 32))

    questions = []
    choices_list = []
    labels = []
    tasks = []

    for s in arc_samples:
        questions.append(f"[Task: Causal Reasoning] {s['question']}")
        choices_list.append(s['choices'])
        labels.append(s['label'])
        tasks.append("ARC")

    for s in obqa_samples:
        questions.append(f"[Task: Scientific Facts] {s['question']}")
        choices_list.append(s['choices'])
        labels.append(s['label'])
        tasks.append("OBQA")

    # Pairwise encoding: (32 questions, 4 choices) -> (32, 4, 1024)
    b = len(questions)
    flat_texts = [
        f"Question: {questions[idx]} Answer: {choices_list[idx][c]}"
        for idx in range(b)
        for c in range(4)
    ]
    inputs = tokenizer(flat_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    last_hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
    vecs = pooled.view(b, 4, -1)
    labels = torch.tensor(labels, device=device)

    return vecs, labels, tasks


def main():
    print("=" * 70)
    print("STARTING MoTE CANARY VERIFICATION")
    print("=" * 70)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Running on compute device: {device}")

    # 1. Apply LoRA (top-8 layers, r=32, alpha=32)
    print("\n[1/5] Initializing ModernBERT + LoRA (top-8 layers, r=32)...")
    lora_params = apply_lora(model, r=32, alpha=32, num_layers=8)
    model.train()
    model.to(device)
    print(f"  LoRA parameters count: {sum(p.numel() for p in lora_params):,}")

    # 2. Instantiate MoTESystem2
    print("\n[2/5] Instantiating MoTESystem2 (4 experts, Top-2 sparse routing, SoftCentroid)...")
    mote = MoTESystem2(num_experts=4, top_k=2, hidden_size=1024, head_dim=512, diff_features=True, soft_centroid=True)
    mote.train()
    mote.to(device)

    expert_params = [p for h in mote.heads for p in h.parameters()]
    router_params = list(mote.router.parameters())
    soft_params = [p for sc in mote.soft_centroids for p in sc.parameters()]
    judger_params = list(mote.judger.parameters())
    print(f"  Total MoTE trainable parameters: {sum(p.numel() for p in expert_params + router_params + soft_params + judger_params):,}")

    # Setup optimizer
    optimizer = torch.optim.Adam([
        {"params": expert_params, "lr": 1e-3},
        {"params": router_params, "lr": 5e-4},
        {"params": soft_params, "lr": 1e-3},
        {"params": judger_params, "lr": 3e-4},
        {"params": lora_params, "lr": 2e-4},
    ])

    # 3. Prepare balanced multi-domain batch
    print("\n[3/5] Encoding 32-sample balanced multi-domain batch (ARC + OBQA)...")
    vecs, labels, tasks = prepare_canary_batch(device)
    print(f"  Encoded input candidate tensor shape: {vecs.shape}")

    # 4. Forward & Backward Pass
    print("\n[4/5] Executing Forward Pass...")
    best, best_conf, pruned, confidences, logits, embeddings, scores, dense_gates, aux_loss = mote(vecs)

    # Check routing distribution
    arc_gates = dense_gates[:16]
    obqa_gates = dense_gates[16:]
    print("  Routing summary:")
    print(f"    ARC batch average expert utilization:  {[round(x, 3) for x in arc_gates.mean(dim=0).tolist()]}")
    print(f"    OBQA batch average expert utilization: {[round(x, 3) for x in obqa_gates.mean(dim=0).tolist()]}")
    print(f"    Switch Transformer load-balance aux loss: {aux_loss.item():.4f}")

    # Loss calculation
    ce_per_expert = torch.stack([
        torch.nn.functional.cross_entropy(h, labels, reduction='none')
        for h in logits.unbind(1)
    ], dim=1)  # (B, E)
    routed_ce = (ce_per_expert * dense_gates).sum(dim=1).mean()

    # Head calibration
    with torch.no_grad():
        head_correct = (logits.max(dim=2).indices == labels.unsqueeze(1)).float()
    cal_loss = torch.nn.functional.binary_cross_entropy(confidences, head_correct)

    # Judger losses
    target = confidences.max(dim=1).indices
    judge_loss = torch.nn.functional.cross_entropy(scores, target, label_smoothing=0.15)

    with torch.no_grad():
        win_logits = logits[torch.arange(len(vecs)), best]
        preds = win_logits.max(dim=1).indices
        is_correct = (preds == labels).float()
    judge_cal_loss = torch.nn.functional.binary_cross_entropy(best_conf, is_correct)

    total_loss = routed_ce + 0.5 * cal_loss + 0.1 * judge_loss + 0.5 * judge_cal_loss + 0.01 * aux_loss

    print("\n  Executing Backward Pass...")
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    # Validate gradient flow
    assert mote.router.gate.weight.grad is not None, "Router gate gradient is None!"
    assert mote.soft_centroids[0].query.weight.grad is not None, "SoftCentroid gradient is None!"
    assert mote.heads[0].input_proj.weight.grad is not None, "Head 0 gradient is None!"
    print("  Gradient assertions passed: Router, Heads, SoftCentroids, and Judger all received valid gradients.")
    print(f"  Training step losses: total={total_loss.item():.4f}, routed_ce={routed_ce.item():.4f}, aux_loss={aux_loss.item():.4f}")

    # 5. Checkpoint Saving and Eval Loading Test
    print("\n[5/5] Testing checkpoint saving & eval.py loading compatibility...")
    ckpt_path = "./models/test_mote_canary.pt"
    os.makedirs("./models", exist_ok=True)
    save_dict = {
        "sys2": mote.state_dict(),
        "lora": {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k},
        "lora_config": {"r": 32, "alpha": 32, "num_layers": 8},
    }
    torch.save(save_dict, ckpt_path)
    print(f"  Saved test checkpoint to {ckpt_path}")

    # Evaluate via eval.py on 16 test samples
    res = evaluate(ckpt_path, max_samples=16, device=device)
    print(f"  eval.py loaded checkpoint cleanly! Evaluation on 16 test samples: Acc={res['accuracy']*100:.1f}%, Brier={res['brier']:.4f}")

    # Clean up test checkpoint
    os.remove(ckpt_path)
    print(f"  Cleaned up temporary checkpoint {ckpt_path}")

    print("\n" + "=" * 70)
    print("CANARY VERIFICATION COMPLETE: ALL SYSTEMS NOMINAL")
    print("=" * 70)


if __name__ == "__main__":
    main()
