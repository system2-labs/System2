"""evals package - Unified System2 Evaluation & Adversarial Red-Team Suite."""

from evals.common import (
    README_REFERENCE,
    TRACK_DATA_MAP,
    load_track_samples,
    resolve_device,
    set_seed,
)
from evals.metrics import (
    compute_accuracy,
    compute_binary_brier,
    compute_consensus_stats,
    compute_ece,
    compute_guarded_metric,
    compute_latency_stats,
    compute_multiclass_brier,
)
from evals.reporting import get_system_metadata, render_markdown_table, save_json_report

__all__ = [
    "resolve_device",
    "set_seed",
    "load_track_samples",
    "TRACK_DATA_MAP",
    "README_REFERENCE",
    "compute_accuracy",
    "compute_binary_brier",
    "compute_multiclass_brier",
    "compute_ece",
    "compute_guarded_metric",
    "compute_latency_stats",
    "compute_consensus_stats",
    "render_markdown_table",
    "save_json_report",
    "get_system_metadata",
]
