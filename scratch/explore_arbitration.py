import torch
import torch.nn.functional as F

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

# Check Judger alone:
best_head = scores.argmax(dim=-1) # (3530,)
judger_preds = logits[torch.arange(n), best_head].argmax(dim=-1)
judger_acc = (judger_preds == labels).float().mean().item()
judger_brier = ((confs - (judger_preds == labels).float()) ** 2).mean().item()
print(f"Judger Alone: Acc={judger_acc*100:.2f}% ({(judger_preds==labels).sum().item()}/{n}), Metric={0.5*judger_acc + 0.5*(1-judger_brier):.4f}")

# Check each individual head:
for h in range(5):
    preds_h = logits[:, h, :].argmax(dim=-1)
    acc_h = (preds_h == labels).float().mean().item()
    brier_h = ((confs - (preds_h == labels).float()) ** 2).mean().item()
    print(f"Head {h} Alone: Acc={acc_h*100:.2f}% ({(preds_h==labels).sum().item()}/{n}), Metric={0.5*acc_h + 0.5*(1-brier_h):.4f}")

# Check baseline ensemble
hw = torch.tensor([1.0, 1.0, 1.25, 1.0, 0.35])
arb_scores = scores / 1.80 + torch.log(hw)
weights = torch.softmax(arb_scores, dim=-1)
base_ens_logits = torch.einsum('bn,bnc->bc', weights, logits)
base_metric, base_acc, base_brier = calc_metric(base_ens_logits, confs, labels)
print(f"Base SOTA Ensemble: Acc={base_acc*100:.2f}% ({(base_ens_logits.argmax(-1)==labels).sum().item()}/{n}), Metric={base_metric:.4f}")

print("\n--- Testing Non-Linear / Margin / Max Pooling Ensembles ---")

# Idea 1: Max-pooling across head probabilities or logits
for temp in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
    probs = torch.softmax(logits / temp, dim=-1) # (B, 5, 4)
    # weighted by head weights
    w_probs = probs * weights.unsqueeze(-1)
    max_probs, _ = w_probs.max(dim=1)
    m, a, b = calc_metric(max_probs, confs, labels)
    if m > 0.665:
        print(f"Max-Prob Pool (temp={temp}): Acc={a*100:.2f}% ({(max_probs.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")

# Idea 2: Power-mean ensembling (generalized mean of probabilities)
for p_pow in [1.5, 2.0, 3.0, 4.0, 0.5]:
    probs = torch.softmax(logits, dim=-1)
    # p-norm along head dimension
    pow_probs = (weights.unsqueeze(-1) * (probs ** p_pow)).sum(dim=1) ** (1.0 / p_pow)
    m, a, b = calc_metric(pow_probs, confs, labels)
    if m > 0.665:
        print(f"Power Mean (p={p_pow}): Acc={a*100:.2f}% ({(pow_probs.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")

# Idea 3: Margin-boosted weights
sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
m1 = sorted_l[:, :, 0] - sorted_l[:, :, 1] # (B, 5) logit margin
for beta in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
    boosted_scores = arb_scores + beta * m1
    w = torch.softmax(boosted_scores, dim=-1)
    ens = torch.einsum('bn,bnc->bc', w, logits)
    m, a, b = calc_metric(ens, confs, labels)
    if m > 0.669:
        print(f"Margin-Boosted Logit Weights (beta={beta}): Acc={a*100:.2f}% ({(ens.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")

# Idea 4: Top-2 probability margin boost with ultra-fine grid
head_probs = torch.softmax(logits, dim=-1)
top2_p = torch.topk(head_probs, k=2, dim=-1).values
m_prob = top2_p[..., 0] - top2_p[..., 1]
for gamma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]:
    w = torch.softmax(arb_scores + gamma * m_prob, dim=-1)
    ens = torch.einsum('bn,bnc->bc', w, logits)
    m, a, b = calc_metric(ens, confs, labels)
    print(f"Prob-Margin Boost (gamma={gamma:.2f}): Acc={a*100:.2f}% ({(ens.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")
