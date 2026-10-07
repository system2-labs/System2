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

sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
margins = sorted_l[:, :, 0] - sorted_l[:, :, 1] # (B, 5)

# Notice: what if arbitration scores are modulated by:
# 1) Relative margin: each head's margin compared to mean margin across heads
# 2) Or pairwise head contrast?
mean_m = margins.mean(dim=-1, keepdim=True)
rel_margin = margins - mean_m # positive if this head is more confident than average head

for alpha in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]:
    for hw in [
        [1.0, 1.0, 1.20, 1.0, 0.38],
        [1.0, 0.95, 1.18, 1.0, 0.38],
        [1.0, 1.0, 1.25, 1.0, 0.35],
    ]:
        hw_t = torch.tensor(hw)
        for tau in [1.72, 1.74, 1.76, 1.78, 1.80]:
            s = scores / tau + torch.log(hw_t) + alpha * rel_margin
            w = torch.softmax(s, dim=-1)
            ens = torch.einsum('bn,bnc->bc', w, logits)
            m, a, b = calc_metric(ens, confs, labels)
            if m > 0.6711:
                print(f"BETTER: tau={tau}, hw={hw}, alpha={alpha:.2f} -> Acc={a*100:.2f}% ({(ens.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")
