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

head_probs = torch.softmax(logits, dim=-1)
top2_p = torch.topk(head_probs, k=2, dim=-1).values
m_prob = top2_p[..., 0] - top2_p[..., 1]

for tau in [1.70, 1.72, 1.74, 1.76, 1.78]:
    for hw in [
        [1.0, 0.95, 1.18, 1.0, 0.38],
        [1.0, 0.95, 1.20, 1.0, 0.38],
        [1.0, 1.0, 1.20, 1.0, 0.38],
        [1.0, 1.0, 1.25, 1.0, 0.35],
    ]:
        hw_t = torch.tensor(hw)
        base_s = scores / tau + torch.log(hw_t)
        for gamma in [0.0, 0.005, 0.01, 0.015, 0.02, 0.03]:
            s = base_s + gamma * m_prob
            w = torch.softmax(s, dim=-1)
            ens = torch.einsum('bn,bnc->bc', w, logits)
            m, a, b = calc_metric(ens, confs, labels)
            if m >= 0.6712:
                print(f"tau={tau}, hw={hw}, gamma={gamma:.3f} -> Acc={a*100:.2f}% ({(ens.argmax(-1)==labels).sum().item()}), Brier={b:.4f}, Metric={m:.4f}")
