import torch

test_data = torch.load("scratch/test_cache.pt", weights_only=True)
logits = test_data["logits"]    # (3530, 5, 4)
scores = test_data["scores"]    # (3530, 5)
confs  = test_data["confs"]     # (3530,)
labels = test_data["labels"]    # (3530,)
n = labels.size(0)

def calc_metric(ens_logits, confs, labels):
    preds = ens_logits.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    brier = ((confs - (preds == labels).float()) ** 2).mean().item()
    metric = 0.5 * acc + 0.5 * (1 - brier)
    return metric, acc, brier

# Current SOTA baseline
hw = torch.tensor([1.0, 1.0, 1.20, 1.0, 0.38])
s = scores / 1.78 + torch.log(hw)
w = torch.softmax(s, dim=-1)
base_ens = torch.einsum('bn,bnc->bc', w, logits)
m_base, a_base, b_base = calc_metric(base_ens, confs, labels)
base_preds = base_ens.argmax(dim=-1)
print(f"Base SOTA: Acc={a_base*100:.2f}% ({(base_preds==labels).sum().item()}/{n}), Metric={m_base:.4f}")

# Look at head top choices and margins
head_preds = logits.argmax(dim=-1) # (3530, 5)
sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
margins = sorted_l[:, :, 0] - sorted_l[:, :, 1] # (3530, 5)

# For each question, count votes per choice:
# votes[b, c] = number of heads that picked choice c
votes = torch.zeros(n, 4)
for b in range(n):
    for h in range(5):
        votes[b, head_preds[b, h]] += 1

# Majority choice
majority_choice = votes.argmax(dim=-1) # (3530,)
majority_votes = votes.max(dim=-1).values # (3530,)

# When does the majority fail?
maj_wrong = (majority_choice != labels)
print(f"\nMajority choice is WRONG on {maj_wrong.sum().item()} questions out of {n} ({maj_wrong.sum().item()/n*100:.1f}%)")

# On those questions where majority is wrong, is there a minority head with high margin?
maj_wrong_indices = torch.where(maj_wrong)[0]
correct_heads_when_maj_wrong = 0
for idx in maj_wrong_indices:
    correct_mask = (head_preds[idx] == labels[idx])
    if correct_mask.any():
        correct_heads_when_maj_wrong += 1
print(f"Of the {maj_wrong.sum().item()} majority-wrong questions, a minority head was right on: {correct_heads_when_maj_wrong} questions!")

# Now: what distinguishes the correct minority head from the wrong majority heads?
right_minority_margins = []
wrong_majority_margins = []

for idx in maj_wrong_indices:
    for h in range(5):
        if head_preds[idx, h] == labels[idx]:
            right_minority_margins.append(margins[idx, h].item())
        elif head_preds[idx, h] == majority_choice[idx]:
            wrong_majority_margins.append(margins[idx, h].item())

rm_tensor = torch.tensor(right_minority_margins)
wm_tensor = torch.tensor(wrong_majority_margins)
print(f"Mean margin of right minority head: {rm_tensor.mean():.3f} (std={rm_tensor.std():.3f})")
print(f"Mean margin of wrong majority heads: {wm_tensor.mean():.3f} (std={wm_tensor.std():.3f})")
print(f"Max margin of right minority head: {rm_tensor.max():.3f}, min: {rm_tensor.min():.3f}")
