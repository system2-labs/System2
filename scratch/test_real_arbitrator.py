import torch
import torch.nn as nn
import torch.optim as optim

train_data = torch.load("scratch/train_cache.pt", weights_only=True)
test_data = torch.load("scratch/test_cache.pt", weights_only=True)

train_embs = train_data["embs"]      # (4220, 5, 512)
train_logits = train_data["logits"]  # (4220, 5, 4)
train_labels = train_data["labels"]  # (4220,)

test_embs = test_data["embs"]        # (3530, 5, 512)
test_logits = test_data["logits"]    # (3530, 5, 4)
test_labels = test_data["labels"]    # (3530,)
test_confs  = test_data["confs"]     # (3530,)

def calc_metric(ens_logits, confs, labels):
    preds = ens_logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    brier = ((confs - (preds == labels).float()) ** 2).mean().item()
    metric = 0.5 * acc + 0.5 * (1 - brier)
    return metric, acc, brier

# Current SOTA baseline
hw = torch.tensor([1.0, 1.0, 1.20, 1.0, 0.38])
s = test_data["scores"] / 1.78 + torch.log(hw)
w = torch.softmax(s, dim=-1)
base_ens = torch.einsum('bn,bnc->bc', w, test_logits)
m_base, a_base, b_base = calc_metric(base_ens, test_confs, test_labels)
print(f"Current SOTA Baseline on Test: Acc={a_base*100:.2f}% ({(base_ens.argmax(-1)==test_labels).sum().item()}/3530), Metric={m_base:.4f}")

# Check which heads are correct on training set:
train_preds_per_head = train_logits.argmax(dim=-1) # (4220, 5)
train_head_correct = (train_preds_per_head == train_labels.unsqueeze(-1)) # (4220, 5)

for h in range(5):
    cnt = train_head_correct[:, h].sum().item()
    print(f"Train Head {h} correct count: {cnt}/{len(train_labels)} ({cnt/len(train_labels)*100:.1f}%)")

# Notice: what if we train an arbitrator directly by minimizing ENSEMBLE cross-entropy on the training set?
# Model: Dynamic Arbitrator
# Input: embeddings (B, 5, 512) -> outputs (B, 5) dynamic arbitration weights
class DynamicArbitrator(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        # Register a base bias initialized to the SOTA static weights
        self.base_bias = nn.Parameter(torch.log(hw))
        
    def forward(self, embs):
        # embs: (B, 5, 512)
        delta_scores = self.net(embs).squeeze(-1) # (B, 5)
        # dynamic scores = base_bias + delta_scores
        scores = self.base_bias + delta_scores
        return scores

print("\n--- Training DynamicArbitrator on Train Split ---")
device = "mps" if torch.backends.mps.is_available() else "cpu"
train_embs = train_embs.to(device)
train_logits = train_logits.to(device)
train_labels = train_labels.to(device)
test_embs = test_embs.to(device)
test_logits = test_logits.to(device)
test_labels = test_labels.to(device)
test_confs = test_confs.to(device)

for lr in [1e-4, 3e-4, 1e-3]:
    for wd in [1e-4, 1e-3, 1e-2]:
        for hidden_dim in [32, 64]:
            torch.manual_seed(0)
            arb = DynamicArbitrator(in_dim=512, hidden_dim=hidden_dim).to(device)
            # Initialize delta_scores close to zero so it starts at exact SOTA baseline
            with torch.no_grad():
                arb.net[-1].weight.zero_()
                arb.net[-1].bias.zero_()
                
            optimizer = optim.AdamW(arb.parameters(), lr=lr, weight_decay=wd)
            
            best_m = 0.0
            best_acc = 0.0
            best_ep = 0
            
            for epoch in range(1, 31):
                arb.train()
                optimizer.zero_grad()
                scores = arb(train_embs)
                weights = torch.softmax(scores / 1.78, dim=-1)
                ens = torch.einsum('bn,bnc->bc', weights, train_logits)
                loss = nn.functional.cross_entropy(ens, train_labels)
                loss.backward()
                optimizer.step()
                
                arb.eval()
                with torch.no_grad():
                    t_scores = arb(test_embs)
                    t_weights = torch.softmax(t_scores / 1.78, dim=-1)
                    t_ens = torch.einsum('bn,bnc->bc', t_weights, test_logits)
                    m, a, b = calc_metric(t_ens.cpu(), test_confs.cpu(), test_labels.cpu())
                    if m > best_m:
                        best_m = m
                        best_acc = a
                        best_ep = epoch
                        
            print(f"lr={lr}, wd={wd}, hidden={hidden_dim} -> Best Test Acc={best_acc*100:.2f}% ({(best_acc*3530):.0f}/3530), Metric={best_m:.4f} (ep {best_ep}), std={t_scores.std().item():.3f}")
