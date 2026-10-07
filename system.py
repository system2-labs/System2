import math
import torch
import torch.nn as nn
from bert import encode
from heads import ThoughtHead


class Judger(nn.Module):
    """Picks the best hypothesis from N candidates.

    Hypotheses attend to each other (they can compare/compete).
    The best one is the one the encoder context supports most.
    """

    def __init__(self, head_dim=512, learnable_temp=False, centered_conf=False, margin_features=False, margin_dim=3, residual=False):
        super().__init__()
        self.learnable_temp = learnable_temp
        self.centered_conf = centered_conf
        self.margin_features = margin_features
        self.residual = residual
        if residual:
            self.prenorm = nn.LayerNorm(head_dim)
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        self.attention = nn.MultiheadAttention(
            embed_dim=head_dim, num_heads=8, batch_first=True)
        in_dim = head_dim + (margin_dim if margin_features else 0)
        if margin_features:
            self.margin_norm = nn.LayerNorm(margin_dim)
        self.score = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings, margin_feats=None):
        # embeddings: (B, N, 512) - one row per hypothesis
        if getattr(self, "residual", False):
            normed = self.prenorm(embeddings)
            att_out, _ = self.attention(normed, normed, normed)
            attended = embeddings + att_out
        else:
            attended, _ = self.attention(embeddings, embeddings, embeddings)
        if self.margin_features:
            if margin_feats is None:
                margin_feats = torch.zeros(embeddings.size(0), embeddings.size(1), 3, device=embeddings.device, dtype=embeddings.dtype)
            normed_margins = self.margin_norm(margin_feats)
            score_in = torch.cat([attended, normed_margins], dim=-1)
        else:
            score_in = attended
        scores = self.score(score_in).squeeze(-1)  # (B, N)
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

    def __init__(self, num_heads=5, num_classes=None, pairwise=False, diff_features=False, learnable_temp=False, head_layers=2, centered_conf=False, uniform_ensemble=False, ensemble_temp=1.0, margin_features=False, head_dropout=0.0, margin_gamma=0.0, prob_ensemble=False, head_prob_temp=1.0, head_norm_first=False, judger_residual=False, head_mlp_dim=256, head_norm_diff=False, veto_blend=0.0, head_weights=None, prob_margin_gamma=0.0):
        super().__init__()
        self.pairwise = pairwise
        self.diff_features = diff_features
        self.num_classes = num_classes
        self.uniform_ensemble = uniform_ensemble
        self.ensemble_temp = ensemble_temp
        self.margin_features = margin_features
        self.margin_gamma = margin_gamma
        self.prob_margin_gamma = prob_margin_gamma
        self.prob_ensemble = prob_ensemble
        self.head_prob_temp = head_prob_temp
        self.head_norm_first = head_norm_first
        self.judger_residual = judger_residual
        self.head_mlp_dim = head_mlp_dim
        self.head_norm_diff = head_norm_diff
        self.veto_blend = veto_blend
        self.head_weights = head_weights
        head_classes = 1 if pairwise else num_classes
        self.heads = nn.ModuleList([
            ThoughtHead(num_classes=head_classes, diff_features=diff_features, num_layers=head_layers, dropout=head_dropout, norm_first=head_norm_first, mlp_dim=head_mlp_dim, norm_diff=head_norm_diff)
            for _ in range(num_heads)
        ])
        self.judger = Judger(learnable_temp=learnable_temp, centered_conf=centered_conf, margin_features=margin_features, residual=judger_residual)

    def forward(self, encoder_output, detach_judger=False):
        if encoder_output.dim() >= 3:
            # Pairwise candidate reasoning: (B, num_choices, 1024) or (B, N, num_choices, 1024)
            all_logits, confidences, embeddings = [], [], []
            for n, head in enumerate(self.heads):
                head_in = encoder_output[:, n, :, :] if encoder_output.dim() == 4 else encoder_output
                x = head.input_proj(head_in)  # (B, num_choices, 512)
                x = head.layers(x)            # (B, num_choices, 512) - cross-candidate self-attention
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

            sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
            m1 = sorted_l[:, :, 0] - sorted_l[:, :, 1]
            if self.margin_features:
                m_mean = sorted_l[:, :, 0] - logits.mean(dim=-1)
                p = torch.softmax(logits, dim=-1)
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1) / 1.38629
                margin_feats = torch.stack([m1, m_mean, ent], dim=-1)
                if detach_judger:
                    margin_feats = margin_feats.detach()
            else:
                margin_feats = None

            judger_in = embeddings.detach() if detach_judger else embeddings
            best, best_conf, pruned, scores = self.judger(judger_in, margin_feats=margin_feats)
            if self.uniform_ensemble:
                ens_logits = logits.mean(dim=1)
            else:
                tau = getattr(self, "ensemble_temp", 1.0)
                gamma = getattr(self, "margin_gamma", 0.0)
                prob_gamma = getattr(self, "prob_margin_gamma", 0.0)
                arb_scores = scores / tau
                if getattr(self, "head_weights", None) is not None:
                    hw = torch.as_tensor(self.head_weights, device=scores.device, dtype=scores.dtype)
                    arb_scores = arb_scores + torch.log(hw.clamp(min=1e-5))
                if prob_gamma > 0.0:
                    head_probs = torch.softmax(logits, dim=-1)
                    top2_p = torch.topk(head_probs, k=2, dim=-1).values
                    m_prob = top2_p[..., 0] - top2_p[..., 1]
                    arb_scores = arb_scores + prob_gamma * m_prob
                if gamma > 0.0:
                    weights = torch.softmax(arb_scores + gamma * m1, dim=-1)
                else:
                    weights = torch.softmax(arb_scores, dim=-1)

                if getattr(self, "prob_ensemble", False):
                    head_tau = getattr(self, "head_prob_temp", 1.0)
                    probs = torch.softmax(logits / head_tau, dim=-1)
                    ens_probs = torch.einsum('bn,bnc->bc', weights, probs)
                    ens_logits = torch.log(ens_probs.clamp(min=1e-8))
                else:
                    ens_logits = torch.einsum('bn,bnc->bc', weights, logits)
            if getattr(self, "veto_blend", 0.0) > 0.0:
                vb = getattr(self, "veto_blend", 0.0)
                peak_logits = logits[torch.arange(logits.size(0), device=logits.device), best]
                ens_logits = (1.0 - vb) * ens_logits + vb * peak_logits
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

            sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
            m1 = sorted_l[:, :, 0] - (sorted_l[:, :, 1] if sorted_l.size(-1) > 1 else sorted_l[:, :, 0])
            if self.margin_features:
                m_mean = sorted_l[:, :, 0] - logits.mean(dim=-1)
                p = torch.softmax(logits, dim=-1)
                denom = math.log(float(logits.size(-1))) if logits.size(-1) > 1 else 1.0
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1) / denom
                margin_feats = torch.stack([m1, m_mean, ent], dim=-1)
                if detach_judger:
                    margin_feats = margin_feats.detach()
            else:
                margin_feats = None

            judger_in = embeddings.detach() if detach_judger else embeddings
            best, best_conf, pruned, scores = self.judger(judger_in, margin_feats=margin_feats)
            if self.uniform_ensemble:
                ens_logits = logits.mean(dim=1)
            else:
                tau = getattr(self, "ensemble_temp", 1.0)
                gamma = getattr(self, "margin_gamma", 0.0)
                prob_gamma = getattr(self, "prob_margin_gamma", 0.0)
                arb_scores = scores / tau
                if getattr(self, "head_weights", None) is not None:
                    hw = torch.as_tensor(self.head_weights, device=scores.device, dtype=scores.dtype)
                    arb_scores = arb_scores + torch.log(hw.clamp(min=1e-5))
                if prob_gamma > 0.0:
                    head_probs = torch.softmax(logits, dim=-1)
                    top2_p = torch.topk(head_probs, k=2, dim=-1).values
                    m_prob = top2_p[..., 0] - top2_p[..., 1]
                    arb_scores = arb_scores + prob_gamma * m_prob
                if gamma > 0.0:
                    weights = torch.softmax(arb_scores + gamma * m1, dim=-1)
                else:
                    weights = torch.softmax(arb_scores, dim=-1)

                if getattr(self, "prob_ensemble", False):
                    head_tau = getattr(self, "head_prob_temp", 1.0)
                    probs = torch.softmax(logits / head_tau, dim=-1)
                    ens_probs = torch.einsum('bn,bnc->bc', weights, probs)
                    ens_logits = torch.log(ens_probs.clamp(min=1e-8))
                else:
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
