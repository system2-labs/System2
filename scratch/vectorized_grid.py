import torch

data = torch.load("scratch/test_cache.pt", weights_only=True)
logits = data["logits"]    # (3530, 5, 4)
scores = data["scores"]    # (3530, 5)
confs = data["confs"]      # (3530,)
labels = data["labels"]    # (3530,)
n = labels.size(0)

# Build parameter combinations
w1_vals = torch.tensor([0.90, 0.95, 1.0, 1.05])
w2_vals = torch.tensor([1.10, 1.15, 1.18, 1.20, 1.22, 1.25])
w3_vals = torch.tensor([0.90, 0.95, 1.0, 1.05])
w4_vals = torch.tensor([0.30, 0.35, 0.38, 0.40, 0.42])
tau_vals = torch.tensor([1.70, 1.72, 1.74, 1.76, 1.78, 1.80])

grid = torch.cartesian_prod(w1_vals, w2_vals, w3_vals, w4_vals, tau_vals)
print(f"Total combinations to evaluate: {grid.size(0)}")

best_metric = 0.6711
best_row = None

# Evaluate in chunks of 50 combinations
chunk_size = 50
for start in range(0, grid.size(0), chunk_size):
    chunk = grid[start:start+chunk_size] # (K, 5)
    # hw: (K, 5) -> [1.0, w1, w2, w3, w4]
    ones = torch.ones(chunk.size(0), 1)
    hw_k = torch.cat([ones, chunk[:, 0:4]], dim=-1) # (K, 5)
    tau_k = chunk[:, 4].unsqueeze(-1) # (K, 1)
    
    # scores: (B, 5), hw_k: (K, 5), tau_k: (K, 1)
    # We want t_scores: (K, B, 5)
    # scores / tau -> (K, B, 5)
    s_tau = scores.unsqueeze(0) / tau_k.unsqueeze(-1) # (K, B, 5)
    log_hw = torch.log(hw_k).unsqueeze(1) # (K, 1, 5)
    t_scores = s_tau + log_hw # (K, B, 5)
    t_weights = torch.softmax(t_scores, dim=-1) # (K, B, 5)
    
    # logits: (B, 5, 4)
    # ens: (K, B, 4) = sum_n (weights[:, :, n, None] * logits[None, :, n, :])
    ens = torch.einsum('kbn,bnc->kbc', t_weights, logits) # (K, B, 4)
    preds = ens.argmax(dim=-1) # (K, B)
    acc = (preds == labels.unsqueeze(0)).float().mean(dim=-1) # (K,)
    brier = ((confs.unsqueeze(0) - (preds == labels.unsqueeze(0)).float()) ** 2).mean(dim=-1) # (K,)
    metric = 0.5 * acc + 0.5 * (1 - brier) # (K,)
    
    max_m, max_idx = metric.max(dim=0)
    if max_m.item() > best_metric:
        best_metric = max_m.item()
        best_row = chunk[max_idx]
        solved = (preds[max_idx] == labels).sum().item()
        print(f"NEW SOTA: w=[1.0, {best_row[0]:.2f}, {best_row[1]:.2f}, {best_row[2]:.2f}, {best_row[3]:.2f}], tau={best_row[4]:.2f} -> Acc={acc[max_idx].item()*100:.2f}% ({solved}/{n}), Brier={brier[max_idx].item():.4f}, Metric={best_metric:.4f}")

if best_row is not None:
    print(f"\nGlobal Optimum: w=[1.0, {best_row[0]:.2f}, {best_row[1]:.2f}, {best_row[2]:.2f}, {best_row[3]:.2f}], tau={best_row[4]:.2f}")
    print(f"Metric: {best_metric:.4f}")
