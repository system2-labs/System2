"""test_checkpoint.py - Unit Tests for Robust Checkpoint Serialization & Config Loading.

Verifies:
1. System 2 checkpoints save complete, explicit hyperparameters and configuration metadata.
2. Direct checkpoint reconstruction without fragile tensor shape guessing.
3. Numerical round-trip invariance: state before save == state after load.
"""

import os
import tempfile
import pytest
import torch

from system import System2


def test_checkpoint_explicit_config_roundtrip():
    """Verify that saving and loading with an explicit config dictionary reproduces exact logits."""
    config = {
        "num_heads": 5,
        "num_classes": 1,
        "pairwise": True,
        "diff_features": True,
        "head_layers": 2,
        "ensemble_temp": 1.78,
        "margin_features": True,
        "margin_dim": 3,
        "judger_residual": True,
        "head_mlp_dim": 256,
        "head_norm_logits": True,
        "decision_conf": True,
        "decision_conf_mode": "softmax",
        "decision_conf_tau": 0.75,
        "judger_mlp_dim": 64,
        "judger_num_heads": 8,
    }

    # Initialize model
    model = System2(**config)
    model.eval()

    # Generate dummy input
    x = torch.randn(2, 5, 4, 1024)
    with torch.no_grad():
        original_output = model(x)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name

    try:
        # Save checkpoint with standardized schema
        save_payload = {
            "sys2": model.state_dict(),
            "config": config,
            "metadata": {
                "track": "test_track",
                "accuracy": 0.95,
                "brier": 0.05,
            },
        }
        torch.save(save_payload, tmp_path)

        # Reload checkpoint using explicit config
        loaded = torch.load(tmp_path, map_location="cpu", weights_only=False)
        assert "config" in loaded, "Checkpoint missing 'config' dictionary"
        assert "sys2" in loaded, "Checkpoint missing 'sys2' state dict"

        # Reconstruct model purely from config dictionary (zero tensor shape guessing)
        reconstructed = System2(**loaded["config"])
        reconstructed.load_state_dict(loaded["sys2"])
        reconstructed.eval()

        with torch.no_grad():
            reloaded_output = reconstructed(x)

        # Verify exact numerical equivalence across all output tensors
        for orig, rel in zip(original_output, reloaded_output):
            assert torch.allclose(orig, rel, atol=1e-6), "Reloaded model outputs diverged from original model"

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
