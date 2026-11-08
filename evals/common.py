"""evals/common.py - Shared dataset loaders, device resolution, and track metadata."""

import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from datasets import load_from_disk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Canonical mapping from Track identifier to held-out test dataset
TRACK_DATA_MAP: Dict[str, str] = {
    "science": os.path.join(ROOT_DIR, "data", "arc"),
    "code": os.path.join(ROOT_DIR, "data", "code_traces"),
    "med": os.path.join(ROOT_DIR, "data", "med_grounded_expanded"),
    "electrical": os.path.join(ROOT_DIR, "data", "electrical_expanded"),
    "auto": os.path.join(ROOT_DIR, "data", "auto"),
    "culinary": os.path.join(ROOT_DIR, "data", "culinary"),
    "math": os.path.join(ROOT_DIR, "data", "math"),
    "legal": os.path.join(ROOT_DIR, "data", "legal_multi_expanded"),
    "finance": os.path.join(ROOT_DIR, "data", "finance_expanded"),
    "physics": os.path.join(ROOT_DIR, "data", "physics_grounded_expanded"),
    "safety": os.path.join(ROOT_DIR, "data", "safety"),
    "eval": os.path.join(ROOT_DIR, "data", "eval"),
    "logic": os.path.join(ROOT_DIR, "data", "logic"),
}

# Grounded Reference Metrics from README.md & .best_metric* files
README_REFERENCE: Dict[str, Dict[str, float]] = {
    "science": {"guarded": 0.7397, "accuracy": 0.7397, "brier": 0.1700, "ece": 0.0821},
    "code": {"guarded": 0.9010, "accuracy": 0.8911, "brier": 0.0891, "ece": 0.0542},
    "med": {"guarded": 0.8954, "accuracy": 0.8692, "brier": 0.0785, "ece": 0.1800},
    "electrical": {"guarded": 0.9114, "accuracy": 0.8903, "brier": 0.0674, "ece": 0.1870},
    "auto": {"guarded": 0.9881, "accuracy": 0.9950, "brier": 0.0188, "ece": 0.0124},
    "culinary": {"guarded": 0.7487, "accuracy": 0.6846, "brier": 0.1872, "ece": 0.0762},
    "math": {"guarded": 0.9462, "accuracy": 0.9560, "brier": 0.0636, "ece": 0.0412},
    "legal": {"guarded": 0.6700, "accuracy": 0.5714, "brier": 0.2315, "ece": 0.0854},
    "finance": {"guarded": 0.8453, "accuracy": 0.7971, "brier": 0.1065, "ece": 0.1511},
    "physics": {"guarded": 0.8697, "accuracy": 0.8256, "brier": 0.0862, "ece": 0.1521},
    "safety": {"guarded": 0.9817, "accuracy": 0.9950, "brier": 0.0315, "ece": 0.1599},
    "eval": {"guarded": 0.9856, "accuracy": 1.0000, "brier": 0.0289, "ece": 0.0289},
    "logic": {"guarded": 0.9855, "accuracy": 1.0000, "brier": 0.0289, "ece": 0.0289},
}


def resolve_device(requested: Optional[str] = None) -> str:
    """Resolve hardware execution target with automatic Apple Silicon MPS support."""
    if requested is not None:
        req = requested.strip().lower()
        if req == "mps":
            if torch.backends.mps.is_available():
                return "mps"
            print("[evals.common] Warning: MPS requested but not available; falling back to CPU", file=sys.stderr)
            return "cpu"
        elif req == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            print("[evals.common] Warning: CUDA requested but not available; falling back to CPU", file=sys.stderr)
            return "cpu"
        return req

    # Default to MPS on Apple Silicon if available
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """Set global seeds for deterministic reproducible evaluation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_track_samples(
    track: str,
    n: Optional[int] = None,
    seed: int = 42,
    data_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load test samples for a specified track deterministically.

    Args:
        track: Domain track name (e.g. 'science', 'code', 'safety')
        n: Optional maximum number of samples. None returns full test set.
        seed: Deterministic shuffle/subsampling seed.
        data_dir: Optional path override for dataset directory.

    Returns:
        List of sample dicts containing 'question', 'choices', 'label', and metadata.
    """
    path = data_dir or TRACK_DATA_MAP.get(track)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Dataset path for track '{track}' not found at: {path}")

    ds = load_from_disk(path)
    if "test" in ds:
        test_split = ds["test"]
    elif "validation" in ds:
        test_split = ds["validation"]
    else:
        test_split = ds[list(ds.keys())[0]]

    total_len = len(test_split)
    if n is None or n >= total_len:
        selected_indices = list(range(total_len))
    else:
        rng = np.random.RandomState(seed)
        selected_indices = sorted(rng.choice(total_len, size=n, replace=False).tolist())

    samples: List[Dict[str, Any]] = []
    for idx in selected_indices:
        row = test_split[idx]
        choices = list(row.get("choices") or [])
        # Ensure 4 candidate choices
        while len(choices) < 4:
            choices.append("N/A")

        sample = {
            "index": idx,
            "track": track,
            "question": row.get("question", ""),
            "choices": choices[:4],
            "label": int(row.get("label", 0)),
            "category": row.get("category"),
            "command": row.get("command"),
            "source": row.get("source"),
        }
        samples.append(sample)

    return samples
