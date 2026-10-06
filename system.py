import torch
import torch.nn as nn
from bert import encode
from heads import ThoughtHead


class Judger(nn.Module):
    """Picks the best hypothesis from N candidates.

    Hypotheses attend to each other (they can compare/compete).
    The best one is the one the encoder context supports most.
    """

    def __init__(self, head_dim=512, learnable_temp=False, centered_conf=False):
        super().__init__()
        self.learnable_temp = learnable_temp
        self.centered_conf = centered_conf
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        self.attention = nn.MultiheadAttention(
            embed_dim=head_dim, num_heads=8, batch_first=True)
        self.score = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings):
        # embeddings: (B, N, 512) - one row per hypothesis
        attended, _ = self.attention(embeddings, embeddings, embeddings)
        scores = self.score(attended).squeeze(-1)  # (B, N)
        best = scores.max(dim=-1).indices        # (B,)
        best_score = scores.gather(1, best.unsqueeze(1)).squeeze(1)
        if self.centered_conf:
            # Mean-centered margin: removes extreme-value positive bias across N heads
            margin = best_score - scores.mean(dim=-1)
            best_conf = torch.sigmoid(margin)
        elif hasattr(self, "log_temp") and self.learnable_temp:
            temp = torch.exp(self.log_temp).clamp(min=0.1, max=10.0)
            best_conf = torch.sigmoid(best_score / temp)
        else:
            best_conf = torch.sigmoid(best_score)
        pruned = (scores < 0.3).sum(dim=-1)
        return best, best_conf, pruned, scores


class System2(nn.Module):
    """Encoder + N parallel thought heads + judger.

    Supports both:
    1. Single-vector classification: (B, 1024) -> N hypotheses -> 1 selected
    2. Pairwise candidate reasoning:  (B, num_choices, 1024) -> N heads evaluate candidates with cross-choice attention -> judger selects best head
    """

    def __init__(self, num_heads=5, num_classes=None, pairwise=False, diff_features=False, learnable_temp=False, head_layers=2, centered_conf=False):
        super().__init__()
        self.pairwise = pairwise
        self.diff_features = diff_features
        self.num_classes = num_classes
        head_classes = 1 if pairwise else num_classes
        self.heads = nn.ModuleList([
            ThoughtHead(num_classes=head_classes, diff_features=diff_features, num_layers=head_layers)
            for _ in range(num_heads)
        ])
        self.judger = Judger(learnable_temp=learnable_temp, centered_conf=centered_conf)

    def forward(self, encoder_output, detach_judger=False):
        if encoder_output.dim() == 3:
            # Pairwise candidate reasoning: (B, num_choices, 1024)
            all_logits, confidences, embeddings = [], [], []
            for head in self.heads:
                x = head.input_proj(encoder_output)  # (B, num_choices, 512)
                x = head.layers(x)                   # (B, num_choices, 512) - cross-candidate self-attention
                if self.diff_features:
                    # Centroid difference: contrast candidate against the group mean
                    diff = x - x.mean(dim=1, keepdim=True)
                    feat = torch.cat([x, diff], dim=-1)  # (B, num_choices, 1024)
                    cand_scores = head.hypothesis_head(feat).squeeze(-1)
                else:
                    cand_scores = head.hypothesis_head(x).squeeze(-1)  # (B, num_choices)
                head_emb = x.mean(dim=1)             # (B, 512)
                conf = torch.sigmoid(head.confidence_head(head_emb)).squeeze(-1)  # (B,)
                all_logits.append(cand_scores)
                confidences.append(conf)
                embeddings.append(head_emb)
            embeddings = torch.stack(embeddings, dim=1)    # (B, N, 512)
            confidences = torch.stack(confidences, dim=1)  # (B, N)
            logits = torch.stack(all_logits, dim=1)        # (B, N, num_choices)
            judger_in = embeddings.detach() if detach_judger else embeddings
            best, best_conf, pruned, scores = self.judger(judger_in)
            weights = torch.softmax(scores, dim=-1)
            ens_logits = torch.einsum('bn,bnc->bc', weights, logits)
            return best, best_conf, pruned, confidences, logits, embeddings, scores, ens_logits
        else:
            # Single-vector classification: (B, 1024)
            embeddings, confidences = [], []
            for head in self.heads:
                x = head.input_proj(encoder_output).unsqueeze(1)
                x = head.layers(x).squeeze(1)  # (B, 512)
                embeddings.append(x)
                confidences.append(torch.sigmoid(head.confidence_head(x)).squeeze(-1))
            embeddings = torch.stack(embeddings, dim=1)  # (B, N, 512)
            confidences = torch.stack(confidences, dim=1)  # (B, N)
            head_logits = [self.heads[n].hypothesis_head(embeddings[:, n, :]) for n in range(embeddings.size(1))]
            logits = torch.stack(head_logits, dim=1)  # (B, N, C)
            judger_in = embeddings.detach() if detach_judger else embeddings
            best, best_conf, pruned, scores = self.judger(judger_in)
            weights = torch.softmax(scores, dim=-1)
            ens_logits = torch.einsum('bn,bnc->bc', weights, logits)
            return best, best_conf, pruned, confidences, logits, embeddings, scores, ens_logits


if __name__ == '__main__':
    torch.manual_seed(0)
    sys2 = System2(num_heads=5)
    vec = encode('Is this ticket about billing?')
    best, best_conf, pruned, confs, logits, emb, scores, ens_logits = sys2(vec)
    print('Heads:', len(confs))  # demo still works: tuple unpacked below
    print('Selected hypothesis #', int(best.item()) + 1)
    print('Winning confidence: %.3f' % best_conf.item())
    print('Pruned: %d of %d' % (int(pruned.item()), len(confs)))
    for i, c in enumerate(confs):
        print('  head %d: conf %.3f' % (i + 1, c.item()))
