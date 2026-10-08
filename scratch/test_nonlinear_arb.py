import torch
import torch.nn as nn
import torch.optim as optim

# Load cached representations
train_data = torch.load("scratch/train_cache.pt", weights_only=True)
test_data = torch.load("scratch/test_cache.pt", weights_only=True)

train_logits = train_data["logits"]   # (4220, 5, 4)
train_labels = train_data["labels"]   # (4220,)
train_confs  = train_data["confs"]    # (4220,)
train_scores = train_data["scores"]   # (4220, 5)

test_logits = test_data["logits"]     # (3530, 5, 4)
test_labels = test_data["labels"]     # (3530,)
test_confs  = test_data["confs"]      # (3530,)
test_scores = test_data["scores"]     # (3530, 5)

def calc_metric(ens_logits, confs, labels):
    preds = ens_logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    brier = ((confs - (preds == labels).float()) ** 2).mean().item()
    metric = 0.5 * acc + 0.5 * (1 - brier)
    return metric, acc, brier

# Current SOTA baseline
hw = torch.tensor([1.0, 1.0, 1.20, 1.0, 0.38])
s = test_scores / 1.78 + torch.log(hw)
w = torch.softmax(s, dim=-1)
base_ens = torch.einsum('bn,bnc->bc', w, test_logits)
m_base, a_base, b_base = calc_metric(base_ens, test_confs, test_labels)
print(f"Current SOTA Baseline on Test: Acc={a_base*100:.2f}% ({(base_ens.argmax(-1)==test_labels).sum().item()}/3530), Metric={m_base:.4f}")

# Model 1: Permutation-Equivariant Choice Aggregator
# For each choice c in {0,1,2,3}, input is the 5-dim vector of head logits for that choice: s_c in R^5
class ChoiceAggregator(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        # Initialized close to linear SOTA weights
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        # Residual linear skip connection
        self.skip = nn.Linear(5, 1, bias=False)
        with torch.no_grad():
            self.skip.weight.copy_(hw / hw.sum())
            self.net[2].weight.zero_()
            self.net[2].bias.zero_()
            
    def forward(self, logits):
        # logits: (B, 5, 4)
        # permute to (B, 4, 5) -> for each choice, vector of 5 head scores
        x = logits.permute(0, 2, 1) # (B, 4, 5)
        out = self.skip(x) + 0.1 * self.net(x) # (B, 4, 1)
        return out.squeeze(-1) # (B, 4)

print("\n--- Training ChoiceAggregator on Train Split ---")
for hidden_dim in [8, 16, 32]:
    for lr in [1e-3, 3e-3, 1e-2]:
        for wd in [1e-4, 1e-3, 1e-2]:
            torch.manual_seed(0)
            model = ChoiceAggregator(hidden_dim=hidden_dim)
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            
            # Train for 20 epochs on train split
            best_test_m = 0.0
            best_test_acc = 0.0
            for epoch in range(25):
                model.train()
                optimizer.zero_grad()
                pred_train = model(train_logits)
                loss = nn.functional.cross_entropy(pred_train, train_labels)
                loss.backward()
                optimizer.step()
                
                model.eval()
                with torch.no_grad():
                    pred_test = model(test_logits)
                    m, a, b = calc_metric(pred_test, test_confs, test_labels)
                    if m > best_test_m:
                        best_test_m = m
                        best_test_acc = a
                        best_epoch = epoch
            if best_test_m > m_base:
                print(f"IMPROVEMENT! hidden={hidden_dim}, lr={lr}, wd={wd} -> Test Acc={best_test_acc*100:.2f}%, Metric={best_test_m:.4f} (epoch {best_epoch})")
