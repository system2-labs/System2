import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from heads import ThoughtHead


# NOTE: [iterate] Helper-extracted masked reductions vs inline branches: shared
# helpers eliminate 4x duplicated mean/std/margin masking while preserving the
# exact unmasked PyTorch calls (bit-for-bit identical) when mask is None.
def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """Compute keepdim mean over `dim`, excluding False entries when `mask` is provided."""
    if mask is None:
        return x.mean(dim=dim, keepdim=True)
    # torch.where avoids 0.0 * (-1e9 or inf) edge cases on padded slots.
    valid = torch.where(mask, x, x.new_zeros(()))
    count = mask.to(dtype=x.dtype).sum(dim=dim, keepdim=True).clamp(min=1.0)
    return valid.sum(dim=dim, keepdim=True) / count


def _masked_mean_std(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-row mean and Bessel-corrected std over valid slots; safe for valid_count == 1."""
    if mask is None and x.size(dim) > 1:
        return x.mean(dim=dim, keepdim=True), x.std(dim=dim, keepdim=True).clamp(min=1e-5)
    if mask is None:
        # C == 1 with no mask: ddof=1 variance is undefined; clamp std=1e-5 so logits center to 0.0 without NaN.
        mean = x.mean(dim=dim, keepdim=True)
        return mean, torch.full_like(mean, 1e-5)
    mean = _masked_mean(x, mask, dim=dim)
    sq_diff = torch.where(mask, (x - mean) ** 2, x.new_zeros(()))
    count = mask.to(dtype=x.dtype).sum(dim=dim, keepdim=True)
    # Use (count - 1).clamp(min=1) to match PyTorch's unbiased std for count >= 2
    # while yielding 0 variance (clamped to 1e-5 std) without NaN when count == 1.
    denom = (count - 1.0).clamp(min=1.0)
    # Clamp variance BEFORE sqrt so zero-variance / valid_count==1 rows do not evaluate
    # d/dx sqrt(0) = inf in backward() and poison gradients with 0 * inf = NaN.
    std = torch.sqrt((sq_diff.sum(dim=dim, keepdim=True) / denom).clamp(min=1e-10))
    return mean, std


def _top2_margin(
    sorted_vals: torch.Tensor,
    valid_count: Optional[torch.Tensor],
) -> torch.Tensor:
    """Return top-1 minus top-2 margin across valid entries; 0.0 when fewer than 2 valid entries exist."""
    if sorted_vals.size(-1) <= 1:
        return sorted_vals.new_zeros(sorted_vals.shape[:-1])
    raw_margin = sorted_vals[..., 0] - sorted_vals[..., 1]
    if valid_count is None:
        return raw_margin
    # Reshape (B,) -> (B, 1, ...) so (B, C) margins never broadcast to (B, B).
    vc = valid_count.reshape(sorted_vals.size(0), *([1] * (raw_margin.dim() - 1)))
    return torch.where(vc >= 2, raw_margin, raw_margin.new_zeros(()))


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

    def forward(self, encoder_output, detach_judger=False, choice_mask: Optional[torch.Tensor] = None):
        if encoder_output.dim() >= 3:
            # Pairwise candidate reasoning: (B, num_choices, 1024) or (B, N, num_choices, 1024)
            B_in, C_in = encoder_output.size(0), encoder_output.size(-2)
            if choice_mask is not None:
                choice_mask = choice_mask.to(device=encoder_output.device, dtype=torch.bool)
                if choice_mask.shape != (B_in, C_in):
                    raise ValueError(
                        f"choice_mask shape {tuple(choice_mask.shape)} does not match (B, C)=({B_in}, {C_in})."
                    )
                if not choice_mask.any(dim=-1).all():
                    raise ValueError("choice_mask must contain at least one True (valid) choice per batch item.")
                # Collapse all-True masks to None so 4+ choice paths execute the exact unmasked graph.
                if choice_mask.all():
                    choice_mask = None

            mask_bc1 = choice_mask.unsqueeze(-1) if choice_mask is not None else None   # (B, C, 1)
            mask_b1c = choice_mask.unsqueeze(1) if choice_mask is not None else None    # (B, 1, C)
            valid_count_b = choice_mask.sum(dim=-1) if choice_mask is not None else None  # (B,)
            pad_mask = ~choice_mask if choice_mask is not None else None

            all_logits, confidences, embeddings = [], [], []
            for n, head in enumerate(self.heads):
                head_in = encoder_output[:, n, :, :] if encoder_output.dim() == 4 else encoder_output
                x = head.input_proj(head_in)  # (B, num_choices, 512)
                if getattr(head, "input_norm", None) is not None:
                    x = head.input_norm(x)
                # TransformerEncoder src_key_padding_mask uses True = ignored position.
                x = head.layers(x, src_key_padding_mask=pad_mask) if pad_mask is not None else head.layers(x)
                if mask_bc1 is not None:
                    # Zero padded token activations after self-attention so residual/FFN outputs
                    # on dummy slots never leak into centroid contrast or head embeddings.
                    x = torch.where(mask_bc1, x, x.new_zeros(()))

                x_mean = _masked_mean(x, mask_bc1, dim=1)  # (B, 1, 512)
                if self.diff_features:
                    # Centroid difference: contrast valid candidates against the valid group mean
                    diff = x - x_mean
                    if mask_bc1 is not None:
                        diff = torch.where(mask_bc1, diff, diff.new_zeros(()))
                    feat = torch.cat([x, diff], dim=-1)  # (B, num_choices, 1024)
                    cand_scores = head.hypothesis_head(feat).squeeze(-1)
                else:
                    cand_scores = head.hypothesis_head(x).squeeze(-1)  # (B, num_choices)

                if pad_mask is not None:
                    cand_scores = cand_scores.masked_fill(pad_mask, -1e9)

                if getattr(self, "judger_soft_emb", False):
                    soft_tau = getattr(self, "judger_soft_tau", 1.0)
                    p_c = torch.softmax(cand_scores / soft_tau, dim=-1)
                    soft_x = torch.einsum('bc,bcd->bd', p_c, x)
                    head_emb = 0.5 * x_mean.squeeze(1) + 0.5 * soft_x
                elif getattr(self, "judger_win_emb", False):
                    win_c = cand_scores.argmax(dim=-1)
                    win_x = x[torch.arange(x.size(0), device=x.device), win_c]
                    head_emb = 0.5 * x_mean.squeeze(1) + 0.5 * win_x
                else:
                    head_emb = x_mean.squeeze(1)             # (B, 512)
                conf = torch.sigmoid(head.confidence_head(head_emb)).squeeze(-1)  # (B,)
                all_logits.append(cand_scores)
                confidences.append(conf)
                embeddings.append(head_emb)
            embeddings = torch.stack(embeddings, dim=1)    # (B, N, 512)
            confidences = torch.stack(confidences, dim=1)  # (B, N)
            logits = torch.stack(all_logits, dim=1)        # (B, N, num_choices)
            if getattr(self, "head_center_logits", False):
                logits = logits - _masked_mean(logits, mask_b1c, dim=-1)
                if mask_b1c is not None:
                    logits = logits.masked_fill(~mask_b1c, -1e9)

            sorted_l, _ = torch.sort(logits, dim=-1, descending=True)
            m1 = _top2_margin(sorted_l, valid_count_b)
            if getattr(self, "consensus_features", False):
                preds = logits.argmax(dim=-1)  # (B, N)
                agreement = (preds.unsqueeze(-1) == preds.unsqueeze(-2)).float()  # (B, N, N)
                agree_ratio = agreement.mean(dim=-1)  # (B, N)
                p = torch.softmax(logits, dim=-1)  # (B, N, C)
                mean_probs = p.mean(dim=1)  # (B, C)
                chosen_consensus_prob = mean_probs.gather(1, preds)  # (B, N)
                m_mean = sorted_l[:, :, 0] - _masked_mean(logits, mask_b1c, dim=-1).squeeze(-1)
                ent = -(p * torch.log(p + 1e-8)).sum(dim=-1) / 1.38629
                margin_feats = torch.stack([m1, m_mean, ent, agree_ratio, chosen_consensus_prob], dim=-1)
                if detach_judger:
                    margin_feats = margin_feats.detach()
            elif self.margin_features:
                m_mean = sorted_l[:, :, 0] - _masked_mean(logits, mask_b1c, dim=-1).squeeze(-1)
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
                    sorted_p = torch.sort(head_probs, dim=-1, descending=True).values
                    m_prob = _top2_margin(sorted_p, valid_count_b)
                    arb_scores = arb_scores + prob_gamma * m_prob
                if gamma > 0.0:
                    weights = torch.softmax(arb_scores + gamma * m1, dim=-1)
                else:
                    weights = torch.softmax(arb_scores, dim=-1)

                if getattr(self, "head_norm_logits", False):
                    logits_mean, logits_std = _masked_mean_std(logits, mask_b1c, dim=-1)
                    normed_logits = (logits - logits_mean) / logits_std
                    if mask_b1c is not None:
                        normed_logits = normed_logits.masked_fill(~mask_b1c, -1e9)
                elif getattr(self, "head_center_logits", False):
                    normed_logits = logits - _masked_mean(logits, mask_b1c, dim=-1)
                    if mask_b1c is not None:
                        normed_logits = normed_logits.masked_fill(~mask_b1c, -1e9)
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
            if getattr(self, "dual_stream_contest", False) and scores.size(-1) >= 2:
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
                    # Mask by per-row conflict so agreeing rows in a mixed batch (B > 1) are untouched,
                    # and apply before decision_conf so win_conf reflects the post-duel logits.
                    ens_logits = ens_logits + duel_boost * conflict.unsqueeze(1).to(ens_logits.dtype)
            if pad_mask is not None:
                # Re-assert -1e9 on padded slots after prob_ensemble/veto_blend/dual_stream_contest so
                # padded slots never carry finite log(1e-8) values and cannot win argmax.
                ens_logits = ens_logits.masked_fill(pad_mask, -1e9)
            if getattr(self, "decision_conf", False):
                conf_tau = torch.exp(self.log_decision_tau).clamp(min=0.1, max=10.0) if (getattr(self, "decision_conf_learnable", False) and hasattr(self, "log_decision_tau")) else getattr(self, "decision_conf_tau", 0.75)
                conf_mode = getattr(self, "decision_conf_mode", "softmax")
                ens_for_conf = ens_logits.detach()
                if conf_mode == "softmax":
                    probs = torch.softmax(ens_for_conf / conf_tau, dim=-1)
                    win_conf = probs.max(dim=-1).values
                elif conf_mode == "margin":
                    sorted_e, _ = torch.sort(ens_for_conf, dim=-1, descending=True)
                    m = _top2_margin(sorted_e, valid_count_b)
                    win_conf = torch.sigmoid(m / conf_tau)
                elif conf_mode == "hybrid":
                    alpha = getattr(self, "decision_conf_alpha", 0.3)
                    sorted_e, _ = torch.sort(ens_for_conf, dim=-1, descending=True)
                    m = _top2_margin(sorted_e, valid_count_b)
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
    # Demo-only import: keeps `import system` free of any ModernBERT load.
    from bert import encode

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
