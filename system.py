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

    def __init__(self, head_dim=512, learnable_temp=False, centered_conf=False, margin_features=False, margin_dim=3, residual=False, judger_mlp_dim=64, judger_num_heads=8, judger_dropout=0.0):
        super().__init__()
        self.learnable_temp = learnable_temp
        self.centered_conf = centered_conf
        self.margin_features = margin_features
        self.residual = residual
        self.judger_mlp_dim = judger_mlp_dim
        self.judger_num_heads = judger_num_heads
        self.judger_dropout = judger_dropout
        if residual:
            self.prenorm = nn.LayerNorm(head_dim)
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.zeros(1))
        self.attention = nn.MultiheadAttention(
            embed_dim=head_dim, num_heads=judger_num_heads, dropout=judger_dropout, batch_first=True)
        in_dim = head_dim + (margin_dim if margin_features else 0)
        if margin_features:
            self.margin_norm = nn.LayerNorm(margin_dim)
        self.score = nn.Sequential(
            nn.Linear(in_dim, judger_mlp_dim),
            nn.GELU(),
            nn.Linear(judger_mlp_dim, 1),
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

    def __init__(self, num_heads=5, num_classes=None, pairwise=False, diff_features=False, learnable_temp=False, head_layers=2, centered_conf=False, uniform_ensemble=False, ensemble_temp=1.0, margin_features=False, head_dropout=0.0, margin_gamma=0.0, prob_ensemble=False, head_prob_temp=1.0, head_norm_first=False, judger_residual=False, head_mlp_dim=256, head_norm_diff=False, veto_blend=0.0, head_weights=None, prob_margin_gamma=0.0, head_center_logits=False, head_norm_logits=False, head_input_norm=False, consensus_features=False, judger_win_emb=False, margin_dim=3, judger_soft_emb=False, judger_soft_tau=1.0, dual_stream_contest=False, dual_stream_blend=0.0, dual_stream_alpha=0.5, decision_conf=False, decision_conf_mode="softmax", decision_conf_tau=0.75, decision_conf_learnable=False, decision_conf_alpha=0.3, judger_mlp_dim=64, judger_num_heads=8, judger_dropout=0.0):
        super().__init__()
        self.pairwise = pairwise
        self.diff_features = diff_features
        self.num_classes = num_classes
        self.uniform_ensemble = uniform_ensemble
        self.ensemble_temp = ensemble_temp
        self.margin_features = margin_features
        self.consensus_features = consensus_features
        self.judger_win_emb = judger_win_emb
        self.judger_soft_emb = judger_soft_emb
        self.judger_soft_tau = judger_soft_tau
        self.margin_gamma = margin_gamma
        self.prob_margin_gamma = prob_margin_gamma
        self.prob_ensemble = prob_ensemble
        self.head_prob_temp = head_prob_temp
        self.head_norm_first = head_norm_first
        self.judger_residual = judger_residual
        self.head_mlp_dim = head_mlp_dim
        self.head_norm_diff = head_norm_diff
        self.veto_blend = veto_blend
        self.dual_stream_contest = dual_stream_contest
        self.dual_stream_blend = dual_stream_blend
        self.dual_stream_alpha = dual_stream_alpha
        self.head_weights = head_weights
        self.head_center_logits = head_center_logits
        self.head_norm_logits = head_norm_logits
        self.head_input_norm = head_input_norm
        self.decision_conf = decision_conf
        self.decision_conf_mode = decision_conf_mode
        self.decision_conf_tau = decision_conf_tau
        self.decision_conf_alpha = decision_conf_alpha
        self.decision_conf_learnable = decision_conf_learnable
        self.judger_mlp_dim = judger_mlp_dim
        self.judger_num_heads = judger_num_heads
        self.judger_dropout = judger_dropout
        if decision_conf_learnable:
            self.log_decision_tau = nn.Parameter(torch.tensor(math.log(max(decision_conf_tau, 0.01))))
        effective_margin_dim = 5 if consensus_features else (margin_dim if (margin_features or consensus_features) else 0)
        self.margin_dim = effective_margin_dim
        head_classes = 1 if pairwise else num_classes
        self.heads = nn.ModuleList([
            ThoughtHead(num_classes=head_classes, diff_features=diff_features, num_layers=head_layers, dropout=head_dropout, norm_first=head_norm_first, mlp_dim=head_mlp_dim, norm_diff=head_norm_diff, input_norm=head_input_norm)
            for _ in range(num_heads)
        ])
        self.judger = Judger(learnable_temp=learnable_temp, centered_conf=centered_conf, margin_features=(margin_features or consensus_features), margin_dim=effective_margin_dim, residual=judger_residual, judger_mlp_dim=judger_mlp_dim, judger_num_heads=judger_num_heads, judger_dropout=judger_dropout)

    def forward(self, encoder_output, detach_judger=False):
        if encoder_output.dim() >= 3:
            # Pairwise candidate reasoning: (B, num_choices, 1024) or (B, N, num_choices, 1024)
            all_logits, confidences, embeddings = [], [], []
            for n, head in enumerate(self.heads):
                head_in = encoder_output[:, n, :, :] if encoder_output.dim() == 4 else encoder_output
                x = head.input_proj(head_in)  # (B, num_choices, 512)
                if getattr(head, "input_norm", None) is not None:
                    x = head.input_norm(x)
                x = head.layers(x)            # (B, num_choices, 512) - cross-candidate self-attention
                if self.diff_features:
                    # Centroid difference: contrast candidate against the group mean
                    diff = x - x.mean(dim=1, keepdim=True)
                    feat = torch.cat([x, diff], dim=-1)  # (B, num_choices, 1024)
                    cand_scores = head.hypothesis_head(feat).squeeze(-1)
                else:
                    cand_scores = head.hypothesis_head(x).squeeze(-1)  # (B, num_choices)
                if getattr(self, "judger_soft_emb", False):
                    soft_tau = getattr(self, "judger_soft_tau", 1.0)
                    p_c = torch.softmax(cand_scores / soft_tau, dim=-1)
                    soft_x = torch.einsum('bc,bcd->bd', p_c, x)
                    head_emb = 0.5 * x.mean(dim=1) + 0.5 * soft_x
                elif getattr(self, "judger_win_emb", False):
                    win_c = cand_scores.argmax(dim=-1)
                    win_x = x[torch.arange(x.size(0), device=x.device), win_c]
                    head_emb = 0.5 * x.mean(dim=1) + 0.5 * win_x
                else:
                    head_emb = x.mean(dim=1)             # (B, 512)
                conf = torch.sigmoid(head.confidence_head(head_emb)).squeeze(-1)  # (B,)
                all_logits.append(cand_scores)
                confidences.append(conf)
                embeddings.append(head_emb)
            embeddings = torch.stack(embeddings, dim=1)    # (B, N, 512)
            confidences = torch.stack(confidences, dim=1)  # (B, N)
            logits = torch.stack(all_logits, dim=1)        # (B, N, num_choices)
            if getattr(self, "head_center_logits", False):
                logits = logits - logits.mean(dim=-1, keepdim=True)

            sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
            m1 = sorted_l[:, :, 0] - sorted_l[:, :, 1]
            if getattr(self, "consensus_features", False):
                preds = logits.argmax(dim=-1)  # (B, N)
                agreement = (preds.unsqueeze(-1) == preds.unsqueeze(-2)).float()  # (B, N, N)
                agree_ratio = agreement.mean(dim=-1)  # (B, N)
                p = torch.softmax(logits, dim=-1)  # (B, N, C)
                mean_probs = p.mean(dim=1)  # (B, C)
                chosen_consensus_prob = mean_probs.gather(1, preds)  # (B, N)
                m_mean = sorted_l[:, :, 0] - logits.mean(dim=-1)
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1) / 1.38629
                margin_feats = torch.stack([m1, m_mean, ent, agree_ratio, chosen_consensus_prob], dim=-1)
                if detach_judger:
                    margin_feats = margin_feats.detach()
            elif self.margin_features:
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

                if getattr(self, "head_norm_logits", False):
                    logits_mean = logits.mean(dim=-1, keepdim=True)
                    logits_std = logits.std(dim=-1, keepdim=True).clamp(min=1e-5)
                    normed_logits = (logits - logits_mean) / logits_std
                elif getattr(self, "head_center_logits", False):
                    normed_logits = logits - logits.mean(dim=-1, keepdim=True)
                else:
                    normed_logits = logits

                if getattr(self, "prob_ensemble", False):
                    head_tau = getattr(self, "head_prob_temp", 1.0)
                    probs = torch.softmax(logits / head_tau, dim=-1)
                    ens_probs = torch.einsum('bn,bnc->bc', weights, probs)
                    ens_logits = torch.log(ens_probs.clamp(min=1e-8))
                else:
                    ens_logits = torch.einsum('bn,bnc->bc', weights, normed_logits)
            if getattr(self, "veto_blend", 0.0) > 0.0:
                vb = getattr(self, "veto_blend", 0.0)
                active_l = normed_logits if (getattr(self, "head_norm_logits", False) or getattr(self, "head_center_logits", False)) else logits
                peak_logits = active_l[torch.arange(logits.size(0), device=logits.device), best]
                ens_logits = (1.0 - vb) * ens_logits + vb * peak_logits
            if getattr(self, "dual_stream_contest", False):
                # Option A.2: Déjà vu Dual-Stream 1-on-1 Duel between top 2 hypotheses
                sorted_heads = scores.argsort(dim=-1, descending=True)
                h1 = sorted_heads[:, 0]
                h2 = sorted_heads[:, 1]
                B_curr = logits.size(0)
                l1 = logits[torch.arange(B_curr, device=logits.device), h1]
                l2 = logits[torch.arange(B_curr, device=logits.device), h2]
                c1 = l1.argmax(dim=-1)
                c2 = l2.argmax(dim=-1)
                conflict = (c1 != c2)
                if conflict.any():
                    d1 = l1[torch.arange(B_curr, device=logits.device), c1] - l1[torch.arange(B_curr, device=logits.device), c2]
                    d2 = l2[torch.arange(B_curr, device=logits.device), c2] - l2[torch.arange(B_curr, device=logits.device), c1]
                    s1 = scores[torch.arange(B_curr, device=scores.device), h1]
                    s2 = scores[torch.arange(B_curr, device=scores.device), h2]
                    alpha = getattr(self, "dual_stream_alpha", 0.5)
                    duel_diff = (s1 - s2) + alpha * (d1 - d2)
                    winner = torch.where(duel_diff > 0, c1, c2)
                    ds_blend = getattr(self, "dual_stream_blend", 0.3)
                    duel_boost = torch.zeros_like(ens_logits).scatter_(1, winner.unsqueeze(1), ds_blend)
            if getattr(self, "decision_conf", False):
                conf_tau = torch.exp(self.log_decision_tau).clamp(min=0.1, max=10.0) if (getattr(self, "decision_conf_learnable", False) and hasattr(self, "log_decision_tau")) else getattr(self, "decision_conf_tau", 0.75)
                conf_mode = getattr(self, "decision_conf_mode", "softmax")
                ens_for_conf = ens_logits.detach()
                if conf_mode == "softmax":
                    probs = torch.softmax(ens_for_conf / conf_tau, dim=-1)
                    win_conf = probs.max(dim=-1).values
                elif conf_mode == "margin":
                    sorted_e, _ = torch.sort(ens_for_conf, dim=-1, descending=True)
                    m = sorted_e[:, 0] - sorted_e[:, 1]
                    win_conf = torch.sigmoid(m / conf_tau)
                elif conf_mode == "hybrid":
                    alpha = getattr(self, "decision_conf_alpha", 0.3)
                    sorted_e, _ = torch.sort(ens_for_conf, dim=-1, descending=True)
                    m = sorted_e[:, 0] - sorted_e[:, 1]
                    m_conf = torch.sigmoid(m / conf_tau)
                    win_conf = alpha * best_conf + (1.0 - alpha) * m_conf
                else:
                    win_conf = best_conf
            else:
                win_conf = best_conf
            return best, win_conf, pruned, confidences, logits, embeddings, scores, ens_logits
        else:
            # Single-vector classification: (B, 1024)
            embeddings, confidences = [], []
            for head in self.heads:
                x = head.input_proj(encoder_output).unsqueeze(1)
                if getattr(head, "input_norm", None) is not None:
                    x = head.input_norm(x)
                x = head.layers(x).squeeze(1)  # (B, 512)
                embeddings.append(x)
                confidences.append(torch.sigmoid(head.confidence_head(x)).squeeze(-1))
            embeddings = torch.stack(embeddings, dim=1)  # (B, N, 512)
            confidences = torch.stack(confidences, dim=1)  # (B, N)
            head_logits = [self.heads[n].hypothesis_head(embeddings[:, n, :]) for n in range(embeddings.size(1))]
            logits = torch.stack(head_logits, dim=1)  # (B, N, C)

            sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
            m1 = sorted_l[:, :, 0] - (sorted_l[:, :, 1] if sorted_l.size(-1) > 1 else sorted_l[:, :, 0])
            if getattr(self, "consensus_features", False):
                preds = logits.argmax(dim=-1)  # (B, N)
                agreement = (preds.unsqueeze(-1) == preds.unsqueeze(-2)).float()  # (B, N, N)
                agree_ratio = agreement.mean(dim=-1)  # (B, N)
                p = torch.softmax(logits, dim=-1)  # (B, N, C)
                mean_probs = p.mean(dim=1)  # (B, C)
                chosen_consensus_prob = mean_probs.gather(1, preds)  # (B, N)
                m_mean = sorted_l[:, :, 0] - logits.mean(dim=-1)
                denom = math.log(float(logits.size(-1))) if logits.size(-1) > 1 else 1.0
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1) / denom
                margin_feats = torch.stack([m1, m_mean, ent, agree_ratio, chosen_consensus_prob], dim=-1)
                if detach_judger:
                    margin_feats = margin_feats.detach()
            elif self.margin_features:
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

                if getattr(self, "head_norm_logits", False):
                    logits_mean = logits.mean(dim=-1, keepdim=True)
                    logits_std = logits.std(dim=-1, keepdim=True).clamp(min=1e-5)
                    normed_logits = (logits - logits_mean) / logits_std
                elif getattr(self, "head_center_logits", False):
                    normed_logits = logits - logits.mean(dim=-1, keepdim=True)
                else:
                    normed_logits = logits

                if getattr(self, "prob_ensemble", False):
                    head_tau = getattr(self, "head_prob_temp", 1.0)
                    probs = torch.softmax(logits / head_tau, dim=-1)
                    ens_probs = torch.einsum('bn,bnc->bc', weights, probs)
                    ens_logits = torch.log(ens_probs.clamp(min=1e-8))
                else:
                    ens_logits = torch.einsum('bn,bnc->bc', weights, normed_logits)
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
