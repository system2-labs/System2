"""sys2.distill - corrective trajectory-to-LoRA flywheel.

Importing this package is free: no torch, transformers, or ModernBERT load until
a function that needs them runs (see steer_distill.run_corrective_distill_smoke).
Uses PEP 562 lazy attribute loading so `python3 -m sys2.distill.steer_distill`
runs without a runpy RuntimeWarning.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "LoadStats",
    "RecoveryPair",
    "evaluate_flip_rate",
    "export_steer_distill_shards",
    "load_trajectory_pairs",
    "run_corrective_distill_smoke",
    "split_by_session",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        mod = import_module(".steer_distill", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
