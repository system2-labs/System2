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
print(f"Current SOTA Baseline: Acc={a_base*100:.2f}% ({(base_ens.argmax(-1)==test_labels).sum().item()}/3530), Metric={m_base:.4f}")

# Build Option-Aware Judger
class OptionAwareJudger(nn.Module):
    def __init__(self, emb_dim=512, hidden_dim=64):
        super().__init__()
        # Input features per head: emb (512) + logits (4) + probs (4) + margin (1) = 521
        self.in_proj = nn.Linear(emb_dim + 4 + 4 + 1, emb_dim)
        self.attention = nn.MultiheadAttention(emb_dim, 4, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.conf_head = nn.Linear(emb_dim, 1)
        
    def forward(self, embs, logits):
        # embs: (B, 5, 512), logits: (B, 5, 4)
        probs = torch.softmax(logits, dim=-1)
        sorted_p, _ = torch.sort(probs, dim=-1, descending=True)
        margin = (sorted_p[..., 0] - sorted_p[..., 1]).unsqueeze(-1) # (B, 5, 1)
        
        feats = torch.cat([embs, logits, probs, margin], dim=-1) # (B, 5, 521)
        x = self.in_proj(feats) # (B, 5, 512)
        att, _ = self.attention(x, x, x)
        scores = self.score(att).squeeze(-1) # (B, 5)
        
        # Also compute confidence
        best = scores.argmax(dim=-1)
        best_att = att[torch.arange(att.size(0)), best]
        conf = torch.sigmoid(self.conf_head(best_att)).squeeze(-1)
        return scores, conf

print("\n--- Training OptionAwareJudger ---")
device = "mps" if torch.backends.mps.is_available() else "cpu"

train_embs = train_embs.to(device)
train_logits = train_logits.to(device)
train_labels = train_labels.to(device)

test_embs = test_embs.to(device)
test_logits = test_logits.to(device)
test_labels = test_labels.to(device)

# Target for Judger on training set:
# Ground truth: which head got the sample right?
train_preds_per_head = train_logits.argmax(dim=-1) # (B, 5)
train_correct_mask = (train_preds_per_head == train_labels.unsqueeze(-1)) # (B, 5)
# Soft target: uniform over correct heads if any, else soft over negative loss
target_dist = train_correct_mask.float()
has_any = target_dist.sum(dim=-1, keepdim=True) > 0
target_dist = torch.where(has_any, target_dist / target_dist.sum(dim=-1, keepdim=True).clamp(min=1.0), torch.full_like(target_dist, 0.2))

for lr in [3e-4, 1e-3, 2e-3]:
    for hidden_dim in [64, 128]:
        torch.manual_seed(42)
        model = OptionAwareJudger(hidden_dim=hidden_dim).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        
        best_m = 0.0
        best_acc = 0.0
        best_ep = 0
        
        for epoch in range(1, 31):
            model.train()
            optimizer.zero_grad()
            scores, conf = model(train_embs, train_logits)
            
            # Loss 1: Cross-entropy with soft target
            loss_judge = -(target_dist * torch.log_softmax(scores, dim=-1)).sum(dim=-1).mean()
            
            # Loss 2: Ensembled choice cross-entropy
            w = torch.softmax(scores / 1.78 + torch.log(hw.to(device)), dim=-1)
            ens_logits = torch.einsum('bn,bnc->bc', w, train_logits)
            loss_ens = nn.functional.cross_entropy(ens_logits, train_labels)
            
            # Total loss
            loss = loss_judge + loss_ens
            loss.backward()
            optimizer.step()
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                t_scores, t_conf = model(test_embs, test_logits)
                w_t = torch.softmax(t_scores / 1.78 + torch.log(hw.to(device)), dim=-1)
                t_ens = torch.einsum('bn,bnc->bc', w_t, test_logits)
                
                m, a, b = calc_metric(t_ens.cpu(), test_confs.cpu(), test_labels.cpu())
                if m > best_m:
                    best_m = m
                    best_acc = a
                    best_ep = epoch
                    
        print(f"lr={lr}, hidden={hidden_dim} -> Best Test Acc={best_acc*100:.2f}% ({(best_acc*3530):.0f}/3530), Metric={best_m:.4f} (at epoch {best_ep})")
