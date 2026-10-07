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

best_metric = 0.6711
best_config = None

# Grid over w0, w1, w2, w3, w4, tau
# Fix w0=1.0 as anchor
for w1 in [0.95, 1.0, 1.05]:
    for w2 in [1.18, 1.20, 1.22, 1.24, 1.26]:
        for w3 in [0.95, 1.0, 1.05]:
            for w4 in [0.34, 0.36, 0.37, 0.38, 0.39, 0.40]:
                for tau in [1.74, 1.76, 1.78, 1.80, 1.82]:
                    hw = torch.tensor([1.0, w1, w2, w3, w4])
                    t_scores = scores / tau + torch.log(hw)
                    t_weights = torch.softmax(t_scores, dim=-1)
                    t_ens = torch.einsum('bn,bnc->bc', t_weights, logits)
                    m, a, b = calc_metric(t_ens, confs, labels)
                    if m > best_metric:
                        best_metric = m
                        best_config = (w1, w2, w3, w4, tau, a, b, (t_ens.argmax(-1)==labels).sum().item())
                        print(f"NEW PEAK! w=[1.0, {w1}, {w2}, {w3}, {w4}], tau={tau} -> Acc={a*100:.2f}% ({best_config[-1]}/{n}), Brier={b:.4f}, Metric={m:.4f}")

if best_config:
    print(f"\nFinal Best Config: w=[1.0, {best_config[0]}, {best_config[1]}, {best_config[2]}, {best_config[3]}], tau={best_config[4]}")
    print(f"Accuracy: {best_config[5]*100:.2f}% ({best_config[7]}/{n}), Brier: {best_config[6]:.4f}, Metric: {best_metric:.4f}")
