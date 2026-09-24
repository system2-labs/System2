import torch
import torch.nn as nn
from bert import encode
from heads import ThoughtHead


class Judger(nn.Module):
    """Picks the best hypothesis from N candidates.

    Hypotheses attend to each other (they can compare/compete).
    The best one is the one the encoder context supports most.
    """

    def __init__(self, head_dim=512):
        super().__init__()
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
        best_conf = torch.sigmoid(scores.gather(1, best.unsqueeze(1))).squeeze(1)
        pruned = (scores < 0.3).sum(dim=-1)
        return best, best_conf, pruned, scores


class System2(nn.Module):
    """Encoder + N parallel thought heads + judger.

    One forward pass: text -> vector -> N hypotheses -> 1 selected.
    No autoregressive generation. No hidden alignment layer.
    """

    def __init__(self, num_heads=5, num_classes=None):
        super().__init__()
        self.heads = nn.ModuleList([ThoughtHead(num_classes=num_classes) for _ in range(num_heads)])
        self.num_classes = num_classes
        self.judger = Judger()

    def forward(self, encoder_output):
        embeddings, confidences = [], []
        for head in self.heads:
            x = head.input_proj(encoder_output).unsqueeze(1)
            x = head.layers(x).squeeze(1)  # (B, 512)
            embeddings.append(x)
            confidences.append(torch.sigmoid(head.confidence_head(x)).squeeze(-1))
        # Stack per-head outputs: (B, N, 512) embeddings, (B, N) confidences.
        # Stacking confidences here (not at the call site) so every consumer
        # gets a tensor, not a Python list.
        embeddings = torch.stack(embeddings, dim=1)  # (B, N, 512)
        confidences = torch.stack(confidences, dim=1)  # (B, N)
        # (B, N, C) - index along the N (head) dim, never the batch dim.
        # zipping heads with `embeddings` would iterate over dim 0 (batch) and
        # collapse every head to a fixed (N, C) output - a batch-size bug that
        # only surfaces when batch != num_heads.
        head_logits = [self.heads[n].hypothesis_head(embeddings[:, n, :]) for n in range(embeddings.size(1))]
        logits = torch.stack(head_logits, dim=1)  # (B, N, C)
        best, best_conf, pruned, scores = self.judger(embeddings)
        return best, best_conf, pruned, confidences, logits, embeddings, scores


if __name__ == '__main__':
    torch.manual_seed(0)
    sys2 = System2(num_heads=5)
    vec = encode('Is this ticket about billing?')
    best, best_conf, pruned, confs, logits, emb, scores = sys2(vec)
    print('Heads:', len(confs))  # demo still works: tuple unpacked below
    print('Selected hypothesis #', int(best.item()) + 1)
    print('Winning confidence: %.3f' % best_conf.item())
    print('Pruned: %d of %d' % (int(pruned.item()), len(confs)))
    for i, c in enumerate(confs):
        print('  head %d: conf %.3f' % (i + 1, c.item()))
