"""mote.py - Sparse Mixture of Thought Experts (MoTE) for System2.

Components:
1. SoftCentroid: Attention-weighted dynamic group centroid for contrastive candidate features.
2. ThoughtRouter: Multi-mode router supporting:
   - "supervised": Domain-conditioned routing (ARC -> Experts 0 & 1, OBQA -> Experts 2 & 3).
   - "ensemble": Noisy top-2 dispatch with load-balancing loss for logit ensembling.
   - "arc_soft": In-domain all-expert activation for pure ARC reasoning.
3. MoTESystem2: Modular reasoning architecture with masked Judger arbitration.
"""
import torch
import torch.nn as nn
from heads import ThoughtHead
from system import Judger


class SoftCentroid(nn.Module):
    """Attention-weighted dynamic centroid.

    Replaces fixed arithmetic mean (1/4 sum x_j) with learned attention:
    alpha = softmax(w_c^T x)
    x_bar = sum(alpha_j * x_j)

    Zero-initialized so at step 0 it is mathematically identical to uniform arithmetic mean.
    """

    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(dim, 1, bias=False)
        nn.init.zeros_(self.query.weight)

    def forward(self, x):
        # x: (B, num_choices, dim)
        attn_logits = self.query(x)  # (B, num_choices, 1)
        weights = torch.softmax(attn_logits, dim=1)
        soft_mean = (x * weights).sum(dim=1, keepdim=True)  # (B, 1, dim)
        return soft_mean, weights


class ThoughtRouter(nn.Module):
    """Dispatches inputs to specialized thought experts."""

    def __init__(self, d_in=1024, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(d_in, num_experts)
        self.domain_clf = nn.Linear(d_in, 2)  # 2-class domain classifier (0: ARC, 1: OBQA)

    def forward(self, x_ctx, training=True, mode="supervised", domain_ids=None):
        B = x_ctx.size(0)
        device = x_ctx.device

        if mode == "supervised":
            domain_logits = self.domain_clf(x_ctx)  # (B, 2)
            if domain_ids is None:
                pred_domain = domain_logits.argmax(dim=-1)
            else:
                pred_domain = domain_ids

            active_mask = torch.zeros(B, self.num_experts, dtype=torch.bool, device=device)
            dense_gates = torch.zeros(B, self.num_experts, device=device)
            for b in range(B):
                d = pred_domain[b].item()
                if d == 0:  # ARC -> Experts 0 & 1
                    active_mask[b, 0] = active_mask[b, 1] = True
                    dense_gates[b, 0] = dense_gates[b, 1] = 0.5
                else:       # OpenBookQA -> Experts 2 & 3
                    active_mask[b, 2] = active_mask[b, 3] = True
                    dense_gates[b, 2] = dense_gates[b, 3] = 0.5

            if domain_ids is not None:
                domain_loss = nn.functional.cross_entropy(domain_logits, domain_ids)
            else:
                domain_loss = torch.tensor(0.0, device=device)

            return active_mask, dense_gates, dense_gates, domain_loss

        elif mode == "ensemble":
            logits = self.gate(x_ctx)  # (B, E)
            if training:
                noise = torch.randn_like(logits) * (1.0 / (self.num_experts ** 2))
                noisy_logits = logits + noise
            else:
                noisy_logits = logits

            full_probs = torch.softmax(logits, dim=-1)  # (B, E)
            topk_vals, topk_indices = torch.topk(noisy_logits, self.top_k, dim=-1)  # (B, K)
            topk_gates = torch.softmax(topk_vals, dim=-1)  # (B, K)

            dense_gates = torch.zeros_like(full_probs).scatter_(1, topk_indices, topk_gates)
            active_mask = torch.zeros_like(full_probs, dtype=torch.bool).scatter_(1, topk_indices, True)

            # Switch Transformer load balancing auxiliary loss
            mask_float = torch.zeros_like(full_probs).scatter_(1, topk_indices, 1.0)
            f = mask_float.mean(dim=0)
            P = full_probs.mean(dim=0)
            aux_loss = self.num_experts * torch.sum(f * P)

            return active_mask, dense_gates, full_probs, aux_loss

        else:  # "arc_soft" / all experts active
            active_mask = torch.ones(B, self.num_experts, dtype=torch.bool, device=device)
            dense_gates = torch.full((B, self.num_experts), 1.0 / self.num_experts, device=device)
            aux_loss = torch.tensor(0.0, device=device)
            return active_mask, dense_gates, dense_gates, aux_loss


class MoTESystem2(nn.Module):
    """Sparse Mixture of Thought Experts System2 Architecture."""

    def __init__(
        self,
        num_experts=4,
        top_k=2,
        hidden_size=1024,
        head_dim=512,
        diff_features=True,
        soft_centroid=True,
        mote_mode="supervised",
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.diff_features = diff_features
        self.soft_centroid = soft_centroid
        self.mote_mode = mote_mode

        self.router = ThoughtRouter(d_in=hidden_size, num_experts=num_experts, top_k=top_k)
        self.heads = nn.ModuleList([
            ThoughtHead(hidden_size=hidden_size, head_dim=head_dim, num_classes=1, diff_features=diff_features)
            for _ in range(num_experts)
        ])
        if soft_centroid:
            self.soft_centroids = nn.ModuleList([
                SoftCentroid(dim=head_dim) for _ in range(num_experts)
            ])
        else:
            self.soft_centroids = None

        self.judger = Judger(head_dim=head_dim)

    def forward(self, encoder_output, domain_ids=None, mode=None, detach_judger=False):
        mode = mode or self.mote_mode
        B = encoder_output.size(0)
        x_ctx = encoder_output.mean(dim=1)  # (B, 1024)
        active_mask, dense_gates, full_probs, aux_loss = self.router(
            x_ctx, training=self.training, mode=mode, domain_ids=domain_ids
        )

        all_logits, confidences, embeddings = [], [], []
        for i, head in enumerate(self.heads):
            x = head.input_proj(encoder_output)  # (B, 4, 512)
            x = head.layers(x)                   # (B, 4, 512)
            if self.diff_features:
                if self.soft_centroids is not None:
                    soft_mean, _ = self.soft_centroids[i](x)
                    diff = x - soft_mean
                else:
                    diff = x - x.mean(dim=1, keepdim=True)
                feat = torch.cat([x, diff], dim=-1)
                cand_scores = head.hypothesis_head(feat).squeeze(-1)  # (B, 4)
            else:
                cand_scores = head.hypothesis_head(x).squeeze(-1)
            head_emb = x.mean(dim=1)  # (B, 512)
            conf = torch.sigmoid(head.confidence_head(head_emb)).squeeze(-1)  # (B,)

            all_logits.append(cand_scores)
            confidences.append(conf)
            embeddings.append(head_emb)

        embeddings = torch.stack(embeddings, dim=1)    # (B, E, 512)
        confidences = torch.stack(confidences, dim=1)  # (B, E)
        logits = torch.stack(all_logits, dim=1)        # (B, E, 4)

        # Cross-attending Judger
        judger_in = embeddings.detach() if detach_judger else embeddings
        raw_best, _, pruned, scores = self.judger(judger_in)

        # Mask Judger scores by active_mask
        masked_scores = scores.masked_fill(~active_mask, -1e9)
        best = masked_scores.max(dim=-1).indices
        best_conf = torch.sigmoid(scores.gather(1, best.unsqueeze(1))).squeeze(1)

        # Logit ensemble
        ens_logits = (logits * dense_gates.unsqueeze(-1)).sum(dim=1)  # (B, 4)

        return best, best_conf, pruned, confidences, logits, embeddings, masked_scores, dense_gates, aux_loss, ens_logits, active_mask
