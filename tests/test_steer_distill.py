"""Tests for sys2.distill (Build 1 scorecard: Gates Q2/Q4, Primary Q1, Secondaries)."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sys2.distill import (
    evaluate_flip_rate,
    export_steer_distill_shards,
    load_trajectory_pairs,
    run_corrective_distill_smoke,
    split_by_session,
)

ROOT = Path(__file__).resolve().parent.parent


def rec(i, session, fix=True):
    return {
        "timestamp": f"2026-11-20T00:00:{i % 60:02d}.{i:04d}Z",
        "session_id": session,
        "stage": ["pre_guard", "post_guard", "rollback"][i % 3],
        "context": f"ctx {i}",
        "rejected_call": {"tool": "bash", "args": {"cmd": f"rm {i}"}},
        "guard_reason": f"reason {i}",
        "eventual_passing_fix": {"tool": "edit", "args": {"p": i}, "summary": "s"} if fix else None,
        "rollback_latency_ms": 12,
    }


def write_dir(d, records, extra_lines=()):
    d.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r) for r in records] + list(extra_lines)
    (d / "a.jsonl").write_text("\n".join(lines) + "\n")
    return d


def test_gate1_corrupt_lines_and_session_leakage(tmp_path):
    recs = [rec(i, f"s{i % 10}") for i in range(100)] + [rec(1000 + i, "su", fix=False) for i in range(5)]
    bad = ['{"truncated": ', "not json", "[]", '{"session_id": 3}', "\x00\x01", json.dumps({**rec(1, "x"), "stage": "bogus"})]
    d = write_dir(tmp_path / "t", recs, bad)
    pairs, stats = load_trajectory_pairs(d)
    assert stats.valid_pairs == 100 and stats.unpaired_rejections == 5 and stats.corrupt_lines == len(bad)
    for seed in range(5):
        train, val = split_by_session(pairs, 0.3, seed)
        assert len(train) + len(val) == 100 and train and val
        assert {p.session_id for p in train}.isdisjoint({p.session_id for p in val})
    assert split_by_session(pairs, 0.3, 1) == split_by_session(pairs, 0.3, 1)
    assert load_trajectory_pairs(tmp_path / "missing")[1].valid_pairs == 0


def _digest(root):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[str(p)] = (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
    return out


def test_gate2_no_mutation_of_models_or_best_metric(tmp_path):
    models = ROOT / "models"
    watched = [p for p in list(models.glob("system2*.pt")) + list(ROOT.glob(".best_metric*"))]
    before = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in watched}
    pairs, _ = load_trajectory_pairs(write_dir(tmp_path / "t", [rec(i, f"s{i % 4}") for i in range(20)]))
    tr, va = split_by_session(pairs)
    export_steer_distill_shards(tr, va, tmp_path / "out", dry_run=True)
    assert not (tmp_path / "out").exists()  # dry run writes nothing
    export_steer_distill_shards(tr, va, tmp_path / "out")
    after = {str(p): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns) for p in watched}
    assert before == after
    with pytest.raises(ValueError):
        export_steer_distill_shards(tr, va, tmp_path / "models" / "x")


def test_full_conversion_both_formats(tmp_path):
    pairs, _ = load_trajectory_pairs(write_dir(tmp_path / "t", [rec(i, f"s{i % 7}") for i in range(50)]))
    tr, va = split_by_session(pairs)
    rep = export_steer_distill_shards(tr, va, tmp_path / "o")
    assert sum(rep["mlx_lora"].values()) == 50 == sum(rep["dpo"].values())
    for split in ("train", "valid"):
        mlx = [json.loads(x) for x in (tmp_path / "o/mlx_lora" / f"{split}.jsonl").read_text().splitlines()]
        dpo = [json.loads(x) for x in (tmp_path / "o/dpo" / f"{split}.jsonl").read_text().splitlines()]
        assert len(mlx) == len(dpo)
        for m, d in zip(mlx, dpo):
            assert m["messages"][0]["role"] == "user" and m["messages"][1]["content"] == d["chosen"]
            assert set(d) == {"prompt", "chosen", "rejected", "metadata"}
            assert d["metadata"]["guard_reason"] in d["prompt"] and d["metadata"]["stage"]
            assert d["chosen"] != d["rejected"]


def test_corrective_smoke_loss_drop_and_control_within_noise():
    r = run_corrective_distill_smoke()
    for arm in ("lora", "deletion_mask"):
        a = r[arm]
        assert a["heldout_corrective_loss_after"] < a["heldout_corrective_loss_before"]
        f = a["clean_control_flip"]
        assert f["flip_rate"] <= f["noise_floor"] + 0.02
        assert f["degraded_flips"] == 0
    assert r["deletion_mask"]["trainable_or_changed_params"] < r["lora"]["trainable_or_changed_params"]
    assert r == run_corrective_distill_smoke()  # deterministic


def test_flip_rate_exposes_hidden_degraded_flips():
    gold = [1] * 10
    base = [[1] * 5 + [0] * 5] * 2  # right on first 5
    cand = [[0] * 5 + [1] * 5] * 2  # right on last 5: 5 improved, 5 degraded
    r = evaluate_flip_rate(base, cand, gold)
    assert r["net_accuracy_delta"] == 0.0
    assert r["improved_flips"] == 5 and r["degraded_flips"] == 5
    assert r["flip_rate"] == 1.0 and r["above_noise_floor"]
    same = evaluate_flip_rate(base, base, gold)
    assert same["flip_rate"] == 0.0 and not same["above_noise_floor"]
    noisy = evaluate_flip_rate([[1, 1, 0, 0], [1, 0, 0, 0]], [[1, 1, 0, 0], [1, 0, 0, 0]], [1, 1, 0, 0])
    assert noisy["noise_floor"] == 0.25 and not noisy["above_noise_floor"]
    with pytest.raises(ValueError):
        evaluate_flip_rate(base[:1], cand, gold)


def _pipeline_once(tmp_path, n=1000):
    d = tmp_path / f"t{time.perf_counter_ns()}"
    recs = [rec(i, f"s{i % 50}", fix=(i % 10 != 0)) for i in range(n)]
    write_dir(d, recs)
    t0 = time.perf_counter()
    pairs, stats = load_trajectory_pairs(d)
    tr, va = split_by_session(pairs)
    export_steer_distill_shards(tr, va, d / "out")
    gold = [i % 6 for i in range(len(pairs))]
    evaluate_flip_rate([gold, gold], [gold, gold], gold)
    return time.perf_counter() - t0, stats


def test_throughput_1000_lines_under_2s(tmp_path):
    for _ in range(2):
        dt, stats = _pipeline_once(tmp_path)
        assert stats.lines == 1000 and dt <= 2.0


def test_import_is_lazy():
    # The parent sys2/__init__ already imports torch, so only transformers and
    # the ModernBERT cache are asserted; steer_distill itself imports torch lazily.
    assert "import torch" not in (ROOT / "sys2/distill/steer_distill.py").read_text().split("def run_corrective")[0].replace("    import torch", "")
    code = (
        "import sys\n"
        "import sys2.distill\n"
        "assert 'transformers' not in sys.modules\n"
        "import bert\n"
        "assert bert._cached_models == {}, bert._cached_models\n"
    )
    p = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stderr[-2000:]
