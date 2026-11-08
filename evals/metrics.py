"""evals/metrics.py - Calibration, Accuracy, Brier, ECE, and Consensus telemetry."""

import itertools
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch


def compute_accuracy(preds: Sequence[int], targets: Sequence[int]) -> float:
    """Compute standard classification accuracy."""
    if len(preds) == 0:
        return 0.0
    correct = sum(1 for p, t in zip(preds, targets) if p == t)
    return float(correct / len(preds))


def compute_binary_brier(confs: torch.Tensor, correct: torch.Tensor) -> float:
    """Compute binary decision Brier score: mean((conf - is_correct)^2).

    Mirrors the canonical formula in eval.py used for guarded metric calculation.
    """
    if len(confs) == 0:
        return 0.0
    diff = confs.float() - correct.float()
    return float(torch.mean(diff ** 2).item())


def compute_multiclass_brier(probs: Sequence[Sequence[float]], targets: Sequence[int]) -> float:
    """Compute multi-choice categorical Brier score: mean(sum_c (p_c - y_c)^2)."""
    if len(probs) == 0:
        return 0.0
    total = 0.0
    for p_vec, target in zip(probs, targets):
        for c, p in enumerate(p_vec):
            y = 1.0 if c == target else 0.0
            total += (p - y) ** 2
    return float(total / len(probs))


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 10) -> float:
    """Compute true Expected Calibration Error (ECE) across M confidence bins.

    Decoupled from BERT model imports while strictly matching eval.compute_ece.
    """
    if len(probs) == 0:
        return 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs > bin_lower) & (probs <= bin_upper) if i > 0 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].float().mean().item()
            avg_conf_in_bin = probs[in_bin].mean().item()
            ece += abs(accuracy_in_bin - avg_conf_in_bin) * prop_in_bin
    return float(ece)


def compute_guarded_metric(accuracy: float, brier: float) -> float:
    """Compute System 2 Guarded Metric: 0.5 * accuracy + 0.5 * (1.0 - brier)."""
    return float(0.5 * accuracy + 0.5 * (1.0 - brier))


def compute_latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Compute mean and p95 latencies in milliseconds."""
    if len(latencies_ms) == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0}
    arr = np.array(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


def compute_consensus_stats(
    per_head_choices_list: Sequence[Sequence[int]],
    winning_choices: Sequence[int],
    targets: Sequence[int],
) -> Dict[str, Any]:
    """Compute inter-head consensus rates and disagreement metrics across 5 ThoughtHeads.

    Args:
        per_head_choices_list: List of length-5 integer lists indicating each head's predicted index.
        winning_choices: The adjudicated final index selected by System 2.
        targets: The ground-truth gold indices.

    Returns:
        Dict with distribution of consensus (1/5..5/5), mean consensus on correct vs incorrect,
        unanimity rate, and inter-head disagreement rate.
    """
    total = len(per_head_choices_list)
    if total == 0:
        return {
            "consensus_counts": {"1/5": 0, "2/5": 0, "3/5": 0, "4/5": 0, "5/5": 0},
            "unanimity_rate": 0.0,
            "mean_consensus_correct": 0.0,
            "mean_consensus_incorrect": 0.0,
            "inter_head_disagreement": 0.0,
        }

    counts = {"1/5": 0, "2/5": 0, "3/5": 0, "4/5": 0, "5/5": 0}
    correct_agreements = []
    incorrect_agreements = []
    disagreement_scores = []

    pairs = list(itertools.combinations(range(5), 2))  # 10 pairs

    for heads, win_idx, target in zip(per_head_choices_list, winning_choices, targets):
        vote_count = heads.count(win_idx)
        key = f"{vote_count}/5" if 1 <= vote_count <= 5 else "1/5"
        counts[key] = counts.get(key, 0) + 1

        is_correct = (win_idx == target)
        if is_correct:
            correct_agreements.append(vote_count)
        else:
            incorrect_agreements.append(vote_count)

        # Pairwise agreement between the 5 heads
        num_agreeing_pairs = sum(1 for (h1, h2) in pairs if heads[h1] == heads[h2])
        pairwise_agreement = num_agreeing_pairs / len(pairs)
        disagreement_scores.append(1.0 - pairwise_agreement)

    unanimity = counts.get("5/5", 0) / total
    mean_correct = float(np.mean(correct_agreements)) if correct_agreements else 0.0
    mean_incorrect = float(np.mean(incorrect_agreements)) if incorrect_agreements else 0.0
    mean_disagreement = float(np.mean(disagreement_scores)) if disagreement_scores else 0.0

    return {
        "consensus_counts": counts,
        "unanimity_rate": round(unanimity, 4),
        "mean_consensus_correct": round(mean_correct, 2),
        "mean_consensus_incorrect": round(mean_incorrect, 2),
        "inter_head_disagreement": round(mean_disagreement, 4),
    }
