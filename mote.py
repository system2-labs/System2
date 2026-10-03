"""mote.py - Sparse Mixture of Thought Experts (MoTE) for System2.

Components:
1. SoftCentroid: Attention-weighted dynamic group centroid for contrastive candidate features.
2. ThoughtRouter: Top-K noisy gating router with Switch Transformer load-balancing auxiliary loss.
3. MoTESystem2: Multi-expert architecture routing diverse question styles to specialized ThoughtHeads.
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
    """Dispatches inputs to top-K thought experts with load balancing."""

    def __init__(self, d_in=1024, num_experts=4, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate = nn.Linear(d_in, num_experts)

    def forward(self, x_ctx, training=True):
        logits = self.gate(x_ctx)  # (B, E)
        if training:
            noise = torch.randn_like(logits) * (1.0 / (self.num_experts ** 2))
            noisy_logits = logits + noise
        else:
            noisy_logits = logits

        full_probs = torch.softmax(logits, dim=-1)  # (B, E)
        topk_vals, topk_indices = torch.topk(noisy_logits, self.top_k, dim=-1)  # (B, K)
        topk_gates = torch.softmax(topk_vals, dim=-1)  # (B, K)

        # Dense gate weights (B, E) - strictly non-zero only for active top-K
        dense_gates = torch.zeros_like(full_probs).scatter_(1, topk_indices, topk_gates)

        # Switch Transformer load balancing auxiliary loss
        mask = torch.zeros_like(full_probs).scatter_(1, topk_indices, 1.0)
        f = mask.mean(dim=0)      # (E,) fraction of samples assigned to expert e
        P = full_probs.mean(dim=0) # (E,) average routing probability for expert e
        aux_loss = self.num_experts * torch.sum(f * P)

        return topk_indices, dense_gates, full_probs, aux_loss


class MoTESystem2(nn.Module):
    """Sparse Mixture of Thought Experts System2 Architecture.

    1. ThoughtRouter dispatches each question context to top-K specialized heads.
    2. Active heads compute candidates with learned SoftCentroid contrast.
    3. Judger cross-attends across all expert embeddings to select winning hypothesis.
    """

    def __init__(
        self,
        num_experts=4,
        top_k=2,
        hidden_size=1024,
        head_dim=512,
        diff_features=True,
        soft_centroid=True,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.diff_features = diff_features
        self.soft_centroid = soft_centroid

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

    def forward(self, encoder_output):
        # encoder_output: (B, 4, 1024)
        B = encoder_output.size(0)
        x_ctx = encoder_output.mean(dim=1)  # (B, 1024)
        topk_indices, dense_gates, full_probs, aux_loss = self.router(x_ctx, training=self.training)

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
        best, best_conf, pruned, scores = self.judger(embeddings)

        return best, best_conf, pruned, confidences, logits, embeddings, scores, dense_gates, aux_loss
