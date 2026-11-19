"""Unit tests for System2Engine._load_adapter config inference and System2 dual-stream duel boost."""

from pathlib import Path

import pytest
import torch

import sys2.engine as engine_mod
from load_system2 import extract_config
from sys2.engine import CHECKPOINT_MAP, System2Engine
from system import System2

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"


def _make_lightweight_engine(device: str = "cpu") -> System2Engine:
    """Create a bare System2Engine shell for _load_adapter tests without loading ModernBERT."""
    # _load_adapter only touches self.device, self.adapters, and CHECKPOINT_MAP,
    # so bypassing __init__ avoids a 395M backbone load while exercising the real method.
    engine = System2Engine.__new__(System2Engine)
    engine.device = device
    engine.adapters = {}
    return engine


def test_load_adapter_infers_non_default_config_and_weights(tmp_path, monkeypatch):
    """_load_adapter must infer non-default hyperparameters, exact weights, and lora_norms."""
    torch.manual_seed(7)
    non_default_cfg = {
        "num_heads": 3,
        "num_classes": 1,
        "pairwise": True,
        "diff_features": True,
        "head_layers": 1,
        "ensemble_temp": 0.65,
        "margin_features": False,
        "consensus_features": True,
        "judger_residual": False,
        "judger_soft_emb": True,
        "judger_soft_tau": 0.8,
        "head_mlp_dim": 128,
        "head_norm_diff": True,
        "head_norm_logits": True,
        "head_weights": [1.2, 0.9, 0.7],
        "dual_stream_contest": True,
        "dual_stream_blend": 0.4,
        "dual_stream_alpha": 0.6,
        "decision_conf": True,
        "decision_conf_mode": "margin",
        "decision_conf_tau": 0.55,
        "decision_conf_learnable": True,
        "decision_conf_alpha": 0.25,
        "judger_mlp_dim": 32,
    }
    src_sys2 = System2(**non_default_cfg)
    src_sys2.eval()

    lora_weights = {"layers.27.attn.Wqkv.lora_A": torch.randn(4, 8)}
    lora_norm_weights = {"layers.27.attn_norm.weight": torch.randn(8)}
    ckpt_path = tmp_path / "non_default_system2.pt"
    torch.save(
        {
            "sys2": src_sys2.state_dict(),
            "lora": lora_weights,
            "lora_norms": lora_norm_weights,
            **non_default_cfg,
        },
        ckpt_path,
    )

    monkeypatch.setitem(CHECKPOINT_MAP, "synthetic_custom", str(ckpt_path))
    engine = _make_lightweight_engine(device="cpu")

    encoder_state, loaded_sys2 = engine._load_adapter("synthetic_custom")

    # Topology & every non-default hyperparameter must match the saved checkpoint.
    assert len(loaded_sys2.heads) == 3
    assert len(loaded_sys2.heads[0].layers.layers) == 1
    assert loaded_sys2.head_mlp_dim == 128
    assert loaded_sys2.judger_mlp_dim == 32
    assert loaded_sys2.consensus_features is True
    assert loaded_sys2.margin_dim == 5
    assert loaded_sys2.head_norm_diff is True
    assert loaded_sys2.judger_residual is False
    assert loaded_sys2.judger_soft_emb is True
    assert loaded_sys2.judger_soft_tau == pytest.approx(0.8)
    assert loaded_sys2.ensemble_temp == pytest.approx(0.65)
    assert loaded_sys2.head_weights == [1.2, 0.9, 0.7]
    assert loaded_sys2.dual_stream_contest is True
    assert loaded_sys2.dual_stream_blend == pytest.approx(0.4)
    assert loaded_sys2.dual_stream_alpha == pytest.approx(0.6)
    assert loaded_sys2.decision_conf is True
    assert loaded_sys2.decision_conf_mode == "margin"
    assert loaded_sys2.decision_conf_tau == pytest.approx(0.55)
    assert loaded_sys2.decision_conf_learnable is True
    assert loaded_sys2.decision_conf_alpha == pytest.approx(0.25)

    # Exact state_dict parity catches any silent dropped or re-initialized parameter.
    src_sd = src_sys2.state_dict()
    loaded_sd = loaded_sys2.state_dict()
    assert set(loaded_sd.keys()) == set(src_sd.keys())
    for key in src_sd:
        assert torch.equal(loaded_sd[key], src_sd[key]), f"Tensor mismatch on {key}"

    # Encoder extraction must preserve LoRA weights AND merge lora_norms when present.
    assert "layers.27.attn.Wqkv.lora_A" in encoder_state
    assert "layers.27.attn_norm.weight" in encoder_state
    assert torch.equal(encoder_state["layers.27.attn.Wqkv.lora_A"], lora_weights["layers.27.attn.Wqkv.lora_A"])
    assert torch.equal(encoder_state["layers.27.attn_norm.weight"], lora_norm_weights["layers.27.attn_norm.weight"])

    # End-to-end forward parity confirms behavioral equivalence, not just attribute copying.
    x = torch.randn(2, 3, 4, 1024)
    with torch.no_grad():
        src_out = src_sys2(x)
        loaded_out = loaded_sys2(x)
    for t_src, t_loaded in zip(src_out, loaded_out):
        assert torch.allclose(t_src, t_loaded, atol=1e-6)


def test_load_adapter_strict_fallback_warns_to_stderr(tmp_path, monkeypatch, capsys):
    """When strict=True fails on an unexpected key, _load_adapter warns to stderr and falls back."""
    torch.manual_seed(11)
    cfg = {
        "num_heads": 2,
        "num_classes": 1,
        "pairwise": True,
        "diff_features": True,
        "head_layers": 1,
        "head_mlp_dim": 64,
        "judger_mlp_dim": 32,
    }
    src_sys2 = System2(**cfg)
    corrupted_sd = dict(src_sys2.state_dict())
    corrupted_sd["legacy_unused_buffer"] = torch.zeros(1)

    ckpt_path = tmp_path / "strict_fallback.pt"
    torch.save({"sys2": corrupted_sd, **cfg}, ckpt_path)

    monkeypatch.setitem(CHECKPOINT_MAP, "synthetic_fallback", str(ckpt_path))
    engine = _make_lightweight_engine(device="cpu")

    _, loaded_sys2 = engine._load_adapter("synthetic_fallback")

    captured = capsys.readouterr()
    assert "strict=False" in captured.err
    assert len(loaded_sys2.heads) == 2
    assert torch.equal(
        loaded_sys2.state_dict()["heads.0.input_proj.weight"],
        src_sys2.state_dict()["heads.0.input_proj.weight"],
    )


def test_load_adapter_preserves_encoder_extraction_branches(tmp_path, monkeypatch):
    """_load_adapter preserves 'encoder' dict and flat 'lora_'/'layers.' extraction branches."""
    torch.manual_seed(13)
    cfg = {"num_heads": 2, "num_classes": 1, "pairwise": True, "diff_features": True, "head_layers": 1}
    src_sys2 = System2(**cfg)

    enc_tensor = torch.randn(2, 2)
    ckpt_enc = tmp_path / "with_encoder.pt"
    torch.save({"sys2": src_sys2.state_dict(), "encoder": {"layers.20.attn.Wqkv.lora_B": enc_tensor}, **cfg}, ckpt_enc)

    flat_tensor = torch.randn(2, 2)
    flat_payload = dict(src_sys2.state_dict())
    flat_payload["layers.21.attn.Wo.lora_A"] = flat_tensor
    ckpt_flat = tmp_path / "flat_state.pt"
    torch.save(flat_payload, ckpt_flat)

    monkeypatch.setitem(CHECKPOINT_MAP, "synthetic_enc", str(ckpt_enc))
    monkeypatch.setitem(CHECKPOINT_MAP, "synthetic_flat", str(ckpt_flat))
    engine = _make_lightweight_engine(device="cpu")

    enc_state, _ = engine._load_adapter("synthetic_enc")
    assert torch.equal(enc_state["layers.20.attn.Wqkv.lora_B"], enc_tensor)

    flat_enc_state, _ = engine._load_adapter("synthetic_flat")
    assert "layers.21.attn.Wo.lora_A" in flat_enc_state
    assert torch.equal(flat_enc_state["layers.21.attn.Wo.lora_A"], flat_tensor)


def test_all_real_checkpoints_load_strictly_with_weights_only_and_engine(capsys):
    """All 13 models/system2*.pt checkpoints load cleanly with weights_only=True, strict=True, and _load_adapter."""
    ckpt_paths = sorted(MODELS_DIR.glob("system2*.pt"))
    if not ckpt_paths:
        pytest.skip("No models/system2*.pt checkpoints found")

    assert len(ckpt_paths) == 13, f"Expected 13 domain checkpoints, found {len(ckpt_paths)}"

    # Verify weights_only=True preserves every metadata key used by extract_config
    # and reconstructs a strict-matching System2 for all 13 production checkpoints.
    expected_configs = {}
    for ckpt_path in ckpt_paths:
        state_safe = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state_full = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg_safe, sys2_state = extract_config(state_safe)
        cfg_full, _ = extract_config(state_full)
        assert cfg_safe == cfg_full, f"weights_only=True dropped or altered config keys for {ckpt_path.name}"

        model = System2(**cfg_safe)
        model.load_state_dict(sys2_state, strict=True)
        expected_configs[ckpt_path.name] = cfg_safe

    # Verify System2Engine._load_adapter loads every mapped domain with strict=True
    # (no fallback warning) and preserves checkpoint-specific hyperparameters.
    engine = _make_lightweight_engine(device="cpu")
    for domain, mapped_path in engine_mod.CHECKPOINT_MAP.items():
        ckpt_name = Path(mapped_path).name
        assert ckpt_name in expected_configs
        expected = expected_configs[ckpt_name]

        _, loaded_sys2 = engine._load_adapter(domain)
        assert len(loaded_sys2.heads) == expected["num_heads"]
        assert loaded_sys2.ensemble_temp == pytest.approx(expected["ensemble_temp"])
        assert loaded_sys2.judger_soft_emb == expected["judger_soft_emb"]
        assert loaded_sys2.head_weights == expected["head_weights"]
        assert loaded_sys2.decision_conf_learnable == expected["decision_conf_learnable"]
        if expected.get("decision_conf_learnable"):
            assert hasattr(loaded_sys2, "log_decision_tau")

    captured = capsys.readouterr()
    assert "strict=False" not in captured.err, f"Unexpected strict fallback warning: {captured.err}"


def _build_mixed_conflict_batch(model: System2) -> tuple[torch.Tensor, int, int]:
    """Construct a B=2 batch where row 0 has top-2 heads disagreeing (c1 != c2) and row 1 agreeing (c1 == c2).

    ThoughtHead has no positional encoding across choices and pools embeddings with a
    permutation-invariant mean, so permuting the 4 choices fed to a single head rotates
    its preferred candidate index without altering Judger scores or top-2 head ordering.
    """
    torch.manual_seed(123)
    num_heads = len(model.heads)
    base_choices = torch.randn(4, 1024)
    probe = base_choices.view(1, 1, 4, 1024).expand(1, num_heads, 4, 1024).clone()

    with torch.no_grad():
        _, _, _, _, probe_logits, _, probe_scores, _ = model(probe)

    sorted_heads = probe_scores[0].argsort(descending=True)
    h1 = int(sorted_heads[0].item())
    h2 = int(sorted_heads[1].item())
    natural_c1 = int(probe_logits[0, h1].argmax().item())
    natural_c2 = int(probe_logits[0, h2].argmax().item())

    def _rotate_to_target(choices: torch.Tensor, current_best: int, target_idx: int) -> torch.Tensor:
        shift = (target_idx - current_best) % choices.size(0)
        return torch.roll(choices, shifts=shift, dims=0)

    # Row 0 (conflict): place h1's top choice at index 0 and h2's top choice at index 1.
    row0 = base_choices.unsqueeze(0).expand(num_heads, 4, 1024).clone()
    row0[h1] = _rotate_to_target(base_choices, natural_c1, target_idx=0)
    row0[h2] = _rotate_to_target(base_choices, natural_c2, target_idx=1)

    # Row 1 (agreement): place both h1's and h2's top choices at index 2.
    row1 = base_choices.unsqueeze(0).expand(num_heads, 4, 1024).clone()
    row1[h1] = _rotate_to_target(base_choices, natural_c1, target_idx=2)
    row1[h2] = _rotate_to_target(base_choices, natural_c2, target_idx=2)

    batch = torch.stack([row0, row1], dim=0)  # (2, num_heads, 4, 1024)
    return batch, h1, h2


def test_dual_stream_contest_boosts_only_conflicting_rows_and_updates_win_conf():
    """dual_stream_contest adds +dual_stream_blend to row 0's duel winner only, leaving agreeing row 1 untouched."""
    torch.manual_seed(42)
    shared_kwargs = dict(
        num_heads=3,
        num_classes=1,
        pairwise=True,
        diff_features=True,
        head_layers=1,
        head_mlp_dim=64,
        judger_mlp_dim=32,
        head_norm_logits=True,
        decision_conf=True,
        decision_conf_mode="softmax",
        decision_conf_tau=0.75,
        dual_stream_blend=0.5,
        dual_stream_alpha=0.5,
    )
    model_off = System2(dual_stream_contest=False, **shared_kwargs)
    model_on = System2(dual_stream_contest=True, **shared_kwargs)
    model_on.load_state_dict(model_off.state_dict(), strict=True)
    model_off.eval()
    model_on.eval()

    batch, _, _ = _build_mixed_conflict_batch(model_off)

    with torch.no_grad():
        _, conf_off, _, _, logits_off, _, scores_off, ens_off = model_off(batch)
        _, conf_on, _, _, logits_on, _, scores_on, ens_on = model_on(batch)

    # Upstream head logits and Judger scores are unaffected by post-ensemble duel boost.
    assert torch.allclose(logits_on, logits_off, atol=1e-6)
    assert torch.allclose(scores_on, scores_off, atol=1e-6)

    # Confirm the batch actually exercises one conflicting row (row 0) and one agreeing row (row 1).
    sorted_heads = scores_off.argsort(dim=-1, descending=True)
    h1, h2 = sorted_heads[:, 0], sorted_heads[:, 1]
    idx = torch.arange(2)
    l1 = logits_off[idx, h1]
    l2 = logits_off[idx, h2]
    c1 = l1.argmax(dim=-1)
    c2 = l2.argmax(dim=-1)
    assert c1[0].item() != c2[0].item(), "Row 0 must have top-2 heads disagreeing"
    assert c1[1].item() == c2[1].item(), "Row 1 must have top-2 heads agreeing"

    # Compute the analytical duel winner for row 0 using the exact dual-stream formula.
    d1_0 = l1[0, c1[0]] - l1[0, c2[0]]
    d2_0 = l2[0, c2[0]] - l2[0, c1[0]]
    s1_0 = scores_off[0, h1[0]]
    s2_0 = scores_off[0, h2[0]]
    duel_diff_0 = (s1_0 - s2_0) + 0.5 * (d1_0 - d2_0)
    expected_winner_0 = int((c1[0] if duel_diff_0 > 0 else c2[0]).item())

    # Row 0: winning candidate logit increases by exactly +0.5; other candidates unchanged.
    delta_0 = ens_on[0] - ens_off[0]
    expected_delta_0 = torch.zeros_like(delta_0)
    expected_delta_0[expected_winner_0] = 0.5
    assert torch.allclose(delta_0, expected_delta_0, atol=1e-6)

    # Row 1: non-conflicting row must NOT receive any spurious boost.
    assert torch.allclose(ens_on[1], ens_off[1], atol=1e-6)

    # decision_conf runs after the duel boost, so win_conf must reflect boosted ens_logits on row 0
    # and stay identical on row 1.
    expected_conf_0 = torch.softmax(ens_on[0] / 0.75, dim=-1).max()
    assert torch.allclose(conf_on[0], expected_conf_0, atol=1e-6)
    assert not torch.allclose(conf_on[0], conf_off[0], atol=1e-6)
    assert torch.allclose(conf_on[1], conf_off[1], atol=1e-6)
