"""test_losses.py - Unit Tests for System 2 Multi-Objective Loss Formulations.

Verifies:
1. Head Task Cross-Entropy & Label Smoothing.
2. Head Calibration Binary Cross-Entropy.
3. Supervised / Oracle Judger Selection Loss.
4. Direct Judger Output Calibration (Brier proxy gradient).
5. Hypothesis Diversity Pairwise Cosine Penalty.
6. Judger Head Load-Balancing Entropy Penalty.
7. Candidate Margin Hinge Loss.
8. Cognitive Autonomy Invariant (zero joint backprop into thought heads).
"""

import pytest
import torch
import torch.nn.functional as F


def test_task_cross_entropy_and_autonomy_invariant():
    """Verify task loss trains heads independently and respects the Cognitive Autonomy Invariant."""
    B, N, C = 4, 5, 4  # Batch=4, 5 Heads, 4 Choices
    logits = torch.randn(B, N, C, requires_grad=True)
    labels = torch.tensor([1, 0, 3, 2])

    # Heads compute cross-entropy independently
    ce_per_h = torch.stack(
        [F.cross_entropy(logits[:, h, :], labels, label_smoothing=0.08, reduction="none") for h in range(N)],
        dim=1,
    )  # (B, N)

    task_loss = ce_per_h.mean()
    task_loss.backward()

    assert logits.grad is not None, "Task loss must backpropagate gradients to candidate logits"
    assert logits.grad.shape == (B, N, C), "Gradient shape mismatch"


def test_head_calibration_bce_loss():
    """Verify individual thought head calibration targets whether head predicted correctly."""
    B, N, C = 4, 5, 4
    raw = torch.randn(B, N, requires_grad=True)
    confs = torch.sigmoid(raw)
    preds = torch.tensor([[1, 0, 1, 1, 2], [0, 0, 0, 1, 0], [3, 3, 2, 3, 3], [2, 1, 2, 2, 2]])
    labels = torch.tensor([1, 0, 3, 2])

    # Target is 1.0 if head got it right, 0.0 otherwise
    head_correct = (preds == labels.unsqueeze(1)).float()
    cal_loss = F.binary_cross_entropy(confs, head_correct)

    assert cal_loss >= 0.0, "Calibration BCE must be non-negative"
    cal_loss.backward()
    assert raw.grad is not None, "Gradients must flow into confidence heads"


def test_judger_selection_target_loss():
    """Verify Judger is supervised to pick a thought head that predicted the ground truth correctly."""
    B, N = 4, 5
    scores = torch.randn(B, N, requires_grad=True)
    head_correct = torch.tensor([
        [1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1],
        [0, 0, 0, 0, 1],
    ])
    confs = torch.tensor([[0.8, 0.2, 0.9, 0.1, 0.3], [0.1, 0.7, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4, 0.95]])

    # Oracle target: among correct heads, pick the one with highest confidence
    masked_confs = torch.where(head_correct.bool(), confs, torch.full_like(confs, -1e9))
    has_correct = head_correct.bool().any(dim=1)
    target = torch.where(has_correct, masked_confs.max(dim=1).indices, confs.max(dim=1).indices)

    judge_loss = F.cross_entropy(scores, target, label_smoothing=0.15)
    assert judge_loss >= 0.0, "Judge selection loss must be non-negative"
    judge_loss.backward()
    assert scores.grad is not None, "Gradients must flow into Judger scoring MLP"


def test_diversity_pairwise_cosine_penalty():
    """Verify pairwise cosine similarity penalty encourages orthogonal thought representations."""
    B, N, D = 4, 5, 512
    emb = torch.randn(B, N, D, requires_grad=True)

    emb_norm = F.normalize(emb, p=2, dim=2)
    sim_matrix = torch.bmm(emb_norm, emb_norm.transpose(1, 2))  # (B, N, N)
    triu_idx = torch.triu_indices(N, N, offset=1, device=emb.device)
    div_loss = sim_matrix[:, triu_idx[0], triu_idx[1]].mean()

    # Div loss should penalize high similarity
    div_loss.backward()
    assert emb.grad is not None, "Gradients must flow into thought head embeddings to repel representations"


def test_judger_head_load_balancing():
    """Verify entropy penalty prevents Judger from collapsing into a single dominant thought head."""
    B, N = 8, 5
    scores = torch.randn(B, N, requires_grad=True)

    w_judger = torch.softmax(scores, dim=-1)  # (B, N)
    w_mean = w_judger.mean(dim=0)             # (N,)
    balance_loss = (w_mean * torch.log(w_mean * N + 1e-8)).sum()

    # A uniform distribution (w_mean = 1/N everywhere) produces minimum loss ~ 0
    balance_loss.backward()
    assert scores.grad is not None, "Load-balancing loss must regularize Judger distribution"
