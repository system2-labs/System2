import torch

data = torch.load("scratch/test_cache.pt", weights_only=True)
logits = data["logits"]    # (3530, 5, 4)
scores = data["scores"]    # (3530, 5)
confs = data["confs"]      # (3530,)
labels = data["labels"]    # (3530,)
n = labels.size(0)

hw = torch.tensor([1.0, 1.0, 1.25, 1.0, 0.35])
s = scores / 1.80 + torch.log(hw)
w = torch.softmax(s, dim=-1)
ens = torch.einsum('bn,bnc->bc', w, logits)
ens_preds = ens.argmax(dim=-1)

preds_per_head = logits.argmax(dim=-1) # (B, 5)
correct_matrix = (preds_per_head == labels.unsqueeze(-1)) # (B, 5)
num_correct = correct_matrix.sum(dim=-1) # (B,)

# Questions where exactly 1 head was correct, but ensemble got it wrong:
mask_1 = (num_correct == 1) & (ens_preds != labels)
print(f"Total cases where exactly 1 head was right and ensemble failed: {mask_1.sum().item()}")

# Which head was the lone correct head in these cases?
indices_1 = torch.where(mask_1)[0]
correct_heads_count = torch.zeros(5)
for idx in indices_1:
    head_idx = torch.where(correct_matrix[idx])[0].item()
    correct_heads_count[head_idx] += 1

for h in range(5):
    print(f"Head {h} was the lone hero in {int(correct_heads_count[h].item())} failed questions")

# In these cases, what were the Judger scores for that lone hero head?
hero_scores = []
other_scores = []
hero_margins = []
other_margins = []

sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
margins = sorted_l[:, :, 0] - sorted_l[:, :, 1] # (B, 5)

for idx in indices_1:
    h = torch.where(correct_matrix[idx])[0].item()
    hero_scores.append(scores[idx, h].item())
    hero_margins.append(margins[idx, h].item())
    for o in range(5):
        if o != h:
            other_scores.append(scores[idx, o].item())
            other_margins.append(margins[idx, o].item())

hero_scores = torch.tensor(hero_scores)
other_scores = torch.tensor(other_scores)
hero_margins = torch.tensor(hero_margins)
other_margins = torch.tensor(other_margins)

print(f"\nHero head mean Judger score: {hero_scores.mean():.3f} vs other heads: {other_scores.mean():.3f}")
print(f"Hero head mean logit margin: {hero_margins.mean():.3f} vs other heads: {other_margins.mean():.3f}")
