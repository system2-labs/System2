"""test_loader.py - Unit tests for System2 checkpoint loader and config reverse-engineering."""

import os
import pytest
import torch

from load_system2 import extract_config, load_system2


def test_extract_config_from_checkpoint():
    ckpt_path = "models/system2.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"{ckpt_path} not found")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config, sys2_state = extract_config(state)

    assert config["num_heads"] == 5
    assert config["head_layers"] == 2
    assert config["pairwise"] is True
    assert config["diff_features"] is True
    assert config["judger_residual"] is True
    assert config["margin_features"] is True
    assert config["decision_conf"] is True
    assert len(sys2_state) > 100


def test_load_system2_cpu():
    ckpt_path = "models/system2.pt"
    if not os.path.exists(ckpt_path):
        pytest.skip(f"{ckpt_path} not found")

    model, config = load_system2(ckpt_path, device="cpu", apply_encoder_lora=False)
    assert model is not None
    assert config["num_heads"] == 5

    # Test single-vector 2D input (B, 1024)
    vec_2d = torch.randn(1, 1024)
    out_2d = model(vec_2d)
    assert len(out_2d) == 8
    best, best_conf, pruned, confs, logits, emb, scores, ens_logits = out_2d
    assert best.shape == (1,)
    assert confs.shape == (1, 5)
    assert logits.shape == (1, 5, 2)

    # Test candidate 3D input (B, 4, 1024)
    vec_3d = torch.randn(1, 4, 1024)
    out_3d = model(vec_3d)
    best_3d, best_conf_3d, pruned_3d, confs_3d, logits_3d, emb_3d, scores_3d, ens_logits_3d = out_3d
    assert best_3d.shape == (1,)
    assert confs_3d.shape == (1, 5)
    assert logits_3d.shape == (1, 5, 4)
