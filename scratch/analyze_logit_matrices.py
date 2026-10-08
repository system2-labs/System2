import torch

test_data = torch.load("scratch/test_cache.pt", weights_only=True)
logits = test_data["logits"]    # (3530, 5, 4)
scores = test_data["scores"]    # (3530, 5)
confs  = test_data["confs"]     # (3530,)
labels = test_data["labels"]    # (3530,)
n = labels.size(0)

# SOTA ensemble
hw = torch.tensor([1.0, 1.0, 1.20, 1.0, 0.38])
s = scores / 1.78 + torch.log(hw)
w = torch.softmax(s, dim=-1)
base_ens = torch.einsum('bn,bnc->bc', w, logits)
base_preds = base_ens.argmax(dim=-1)

# Correctness per head
head_preds = logits.argmax(dim=-1) # (3530, 5)
head_correct = (head_preds == labels.unsqueeze(-1)) # (3530, 5)

# Questions where at least one head was right, but ensemble got it wrong:
oracle_miss_mask = head_correct.any(dim=-1) & (base_preds != labels)
print(f"Total Oracle Misses (solvable by heads, missed by ensemble): {oracle_miss_mask.sum().item()}")

miss_indices = torch.where(oracle_miss_mask)[0]

# Inspect the first 10 misses:
print("\n--- Inspecting First 5 Missed Questions ---")
for idx in miss_indices[:5]:
    lbl = labels[idx].item()
    ens_p = base_preds[idx].item()
    print(f"\nQuestion {idx.item()}: True Label = {lbl}, Ensemble Picked = {ens_p}")
    print(f"Head Predictions: {[head_preds[idx, h].item() for h in range(5)]}")
    print(f"Head Correctness: {[head_correct[idx, h].item() for h in range(5)]}")
    print(f"Judger Scores: {[round(scores[idx, h].item(), 2) for h in range(5)]}")
    print(f"Arbitration Weights: {[round(w[idx, h].item(), 3) for h in range(5)]}")
    print("Logits Matrix (Heads x Choices):")
    for h in range(5):
        row_str = " ".join([f"{logits[idx, h, c].item():6.2f}" for c in range(4)])
        mark = " (RIGHT)" if head_correct[idx, h] else ""
        print(f"  Head {h}: [{row_str}]{mark}")
    ens_row = " ".join([f"{base_ens[idx, c].item():6.2f}" for c in range(4)])
    print(f"  Ensemble Logits: [{ens_row}]")
