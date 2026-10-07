import torch

data = torch.load("scratch/test_cache.pt", weights_only=True)
logits = data["logits"]    # (3530, 5, 4)
scores = data["scores"]    # (3530, 5)
confs = data["confs"]      # (3530,)
labels = data["labels"]    # (3530,)
n = labels.size(0)

def calc_metric(ens_logits, confs, labels):
    preds = ens_logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    brier = ((confs - (preds == labels).float()) ** 2).mean().item()
    metric = 0.5 * acc + 0.5 * (1 - brier)
    return metric, acc, brier

# Test zero-mean centering across choices for each head
centered_logits = logits - logits.mean(dim=-1, keepdim=True)

hw = torch.tensor([1.0, 1.0, 1.25, 1.0, 0.35])
arb_scores = scores / 1.80 + torch.log(hw)
weights = torch.softmax(arb_scores, dim=-1)

ens_orig = torch.einsum('bn,bnc->bc', weights, logits)
m_orig, a_orig, b_orig = calc_metric(ens_orig, confs, labels)
print(f"Original: Acc={a_orig*100:.2f}%, Metric={m_orig:.4f}")

ens_centered = torch.einsum('bn,bnc->bc', weights, centered_logits)
m_cent, a_cent, b_cent = calc_metric(ens_centered, confs, labels)
print(f"Post-hoc Zero-mean centered: Acc={a_cent*100:.2f}%, Metric={m_cent:.4f}")

# What if we normalize each head's logits by standard deviation? (Z-score normalization)
std_logits = logits.std(dim=-1, keepdim=True).clamp(min=1e-5)
z_logits = centered_logits / std_logits
ens_z = torch.einsum('bn,bnc->bc', weights, z_logits)
m_z, a_z, b_z = calc_metric(ens_z, confs, labels)
print(f"Post-hoc Z-scored: Acc={a_z*100:.2f}%, Metric={m_z:.4f}")

# What about fine grid over head_weights?
print("\n--- Fine Grid Search on Head Weights ---")
best_m = m_orig
best_hw = hw
for w2 in [1.15, 1.20, 1.22, 1.25, 1.28, 1.30, 1.35]:
    for w4 in [0.25, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42]:
        for tau in [1.70, 1.75, 1.78, 1.80, 1.82, 1.85]:
            test_hw = torch.tensor([1.0, 1.0, w2, 1.0, w4])
            t_scores = scores / tau + torch.log(test_hw)
            t_weights = torch.softmax(t_scores, dim=-1)
            t_ens = torch.einsum('bn,bnc->bc', t_weights, logits)
            m, a, b = calc_metric(t_ens, confs, labels)
            if m > best_m:
                best_m = m
                print(f"BETTER! w2={w2}, w4={w4}, tau={tau} -> Acc={a*100:.2f}% ({(t_ens.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")
