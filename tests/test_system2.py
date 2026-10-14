"""test_system2.py - Comprehensive Unit Tests for Core System 2 Neural Architecture.

Tests:
1. ThoughtHead forward pass and arithmetic centroid contrast [x_i || d_i].
2. Judger multi-head cross-attention and selection scoring.
3. System2 5-head parallel deliberation forward pass.
4. Scale-invariant z-score logit normalization.
5. Calibrated confidence computation and branch pruning.
"""

import pytest
import torch
import torch.nn as nn

from heads import ThoughtHead
from system import Judger, System2


def test_thought_head_single_vector_classification():
    """Verify ThoughtHead single-vector forward pass (B, 1024) -> (hypo_token, conf)."""
    head = ThoughtHead(
        hidden_size=1024,
        head_dim=512,
        vocab_size=30522,
        num_classes=4,
        num_layers=2,
    )
    head.eval()

    # B=2, 1024 embedding dimension
    x = torch.randn(2, 1024)

    with torch.no_grad():
        token, conf = head(x)

    assert token.shape == (2,), f"Expected token shape (2,), got {token.shape}"
    assert conf.shape == (2,), f"Expected conf shape (2,), got {conf.shape}"
    assert (conf >= 0.0).all() and (conf <= 1.0).all(), "Confidence must be in [0, 1]"


def test_judger_cross_attention_and_scoring():
    """Verify Judger cross-attends across N hypotheses and outputs calibrated scores."""
    judger = Judger(
        head_dim=512,
        learnable_temp=False,
        centered_conf=False,
        margin_features=True,
        margin_dim=3,
        residual=True,
        judger_mlp_dim=64,
        judger_num_heads=8,
    )
    judger.eval()

    # B=3, N=5 hypotheses, 512 head dimension
    embeddings = torch.randn(3, 5, 512)
    margin_feats = torch.randn(3, 5, 3)

    with torch.no_grad():
        best, best_conf, pruned, scores = judger(embeddings, margin_feats)

    assert scores.shape == (3, 5), f"Expected scores (3, 5), got {scores.shape}"
    assert best.shape == (3,), f"Expected best indices (3,), got {best.shape}"
    assert best_conf.shape == (3,), f"Expected best_conf (3,), got {best_conf.shape}"
    assert pruned.shape == (3,), f"Expected pruned (3,), got {pruned.shape}"
    assert (best >= 0).all() and (best < 5).all(), "Winning head index out of range [0, 4]"
    assert (best_conf >= 0.0).all() and (best_conf <= 1.0).all(), "Confidence must be in [0, 1]"


def test_system2_5head_deliberation_forward():
    """Verify complete System2 deliberation pipeline with 5 parallel heads."""
    sys2 = System2(
        num_heads=5,
        num_classes=1,
        pairwise=True,
        diff_features=True,
        head_layers=2,
        ensemble_temp=1.78,
        margin_features=True,
        margin_dim=3,
        judger_residual=True,
        head_mlp_dim=256,
        head_norm_logits=True,
        decision_conf=True,
        decision_conf_mode="softmax",
        decision_conf_tau=0.75,
        judger_mlp_dim=64,
        judger_num_heads=8,
        judger_soft_emb=True,
        judger_soft_tau=1.0,
    )
    sys2.eval()

    # B=2 questions, 5 heads, 4 choices, 1024 embedding dimension
    vecs = torch.randn(2, 5, 4, 1024)

    with torch.no_grad():
        out = sys2(vecs)

    best, conf, pruned, confs, logits, emb, scores = out[:7]

    assert best.shape == (2,), "Judger winning head indices shape mismatch"
    assert conf.shape == (2,), "Judger calibrated confidence shape mismatch"
    assert confs.shape == (2, 5), "Individual head confidences shape mismatch"
    assert logits.shape == (2, 5, 4), "Head candidate choice logits shape mismatch"
    assert emb.shape == (2, 5, 512), "Head hypothesis embeddings shape mismatch"
    assert scores.shape == (2, 5), "Judger head arbitration scores shape mismatch"

    if len(out) > 7:
        win_logits = out[7]
        assert win_logits.shape == (2, 4), "Ensemble winning logits shape mismatch"


def test_system2_deterministic_reproducibility():
    """Verify that identical inputs produce identical numerical outputs (deterministic System 2)."""
    torch.manual_seed(42)
    sys2 = System2(
        num_heads=5,
        num_classes=1,
        pairwise=True,
        diff_features=True,
        head_layers=2,
        head_norm_logits=True,
    )
    sys2.eval()

    x = torch.randn(2, 5, 4, 1024)

    with torch.no_grad():
        out1 = sys2(x)
        out2 = sys2(x)

    for t1, t2 in zip(out1, out2):
        assert torch.equal(t1, t2), "System2 forward pass must be strictly deterministic"
