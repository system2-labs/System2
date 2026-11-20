"""SteerDistill flywheel: guard/rollback trajectories -> corrective training shards.

Architecture
------------
pi-system2 (`core/system2/trajectory.ts`) appends JSONL records when the guard
rejects a tool call and when a later call passes. This module is the offline
consumer:

  load_trajectory_pairs -> split_by_session -> export_steer_distill_shards
                                            -> run_corrective_distill_smoke
  evaluate_flip_rate (per-decision flip harness, used by the smoke run)

Ideas borrowed (arXiv ids as given in the build scorecard, not independently
re-verified here): SteerDistill 2611.13410 (corrective turn steered by the
guard reason), Deletion Tuning 2611.10866 (binary mask that zeroes a small
fraction of weights instead of adding a low-rank update), Noise Floor
2611.15671 (judge a change by per-decision flips vs run-to-run noise, since
flat aggregate accuracy can hide improved and degraded flips cancelling out).

Constraints
-----------
* Lazy imports: torch is imported inside the smoke function only. Nothing here
  may import transformers / bert.py, so `import sys2.distill` stays cheap.
* Never touch production checkpoints: exporters refuse any output path that
  sits under a `models/` directory or names `system2*.pt` / `.best_metric*`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

DEFAULT_TRAJECTORIES_DIR = Path.home() / ".system2" / "agent" / "trajectories"

_VALID_STAGES = {"pre_guard", "post_guard", "rollback"}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryPair:
    """One valid (rejection -> passing fix) record, schema of trajectory.ts."""

    timestamp: str
    session_id: str
    stage: str
    context: str
    rejected_call: dict
    guard_reason: str
    eventual_passing_fix: dict
    rollback_latency_ms: float | None = None
    sentinel_errors: list = field(default_factory=list)


@dataclass
class LoadStats:
    files: int = 0
    lines: int = 0
    valid_pairs: int = 0
    corrupt_lines: int = 0  # bad JSON or schema-invalid
    unpaired_rejections: int = 0  # eventual_passing_fix is None

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def _is_call(obj: Any) -> bool:
    return isinstance(obj, dict) and isinstance(obj.get("tool"), str) and isinstance(obj.get("args"), dict)


def _parse_record(obj: Any) -> RecoveryPair | None | str:
    """Return a pair, None for unpaired rejection, or 'corrupt' for bad schema."""
    if not isinstance(obj, dict):
        return "corrupt"
    for key in ("timestamp", "session_id", "context", "guard_reason"):
        if not isinstance(obj.get(key), str):
            return "corrupt"
    if obj.get("stage") not in _VALID_STAGES or not _is_call(obj.get("rejected_call")):
        return "corrupt"
    fix = obj.get("eventual_passing_fix")
    if fix is None:
        return None
    if not _is_call(fix):
        return "corrupt"
    lat = obj.get("rollback_latency_ms")
    se = obj.get("sentinel_errors")
    return RecoveryPair(
        timestamp=obj["timestamp"],
        session_id=obj["session_id"],
        stage=obj["stage"],
        context=obj["context"],
        rejected_call=obj["rejected_call"],
        guard_reason=obj["guard_reason"],
        eventual_passing_fix=fix,
        rollback_latency_ms=float(lat) if isinstance(lat, (int, float)) else None,
        sentinel_errors=se if isinstance(se, list) else [],
    )


def load_trajectory_pairs(trajectories_dir: Path | str | None = None) -> tuple[list[RecoveryPair], LoadStats]:
    """Read every *.jsonl in the dir; never raises on bad data (Gate 1)."""
    d = Path(trajectories_dir) if trajectories_dir else DEFAULT_TRAJECTORIES_DIR
    stats = LoadStats()
    pairs: list[RecoveryPair] = []
    if not d.is_dir():
        return pairs, stats
    for path in sorted(d.glob("*.jsonl")):
        stats.files += 1
        try:
            # errors="replace": a partial multibyte write must not abort the file.
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            stats.lines += 1
            try:
                parsed = _parse_record(json.loads(line))
            except (ValueError, RecursionError):
                stats.corrupt_lines += 1
                continue
            if parsed == "corrupt":
                stats.corrupt_lines += 1
            elif parsed is None:
                stats.unpaired_rejections += 1
            else:
                pairs.append(parsed)
                stats.valid_pairs += 1
    return pairs, stats


def split_by_session(
    pairs: Sequence[RecoveryPair], val_fraction: float = 0.2, seed: int = 0
) -> tuple[list[RecoveryPair], list[RecoveryPair]]:
    """Deterministic split on session_id so a session is wholly train or val."""
    sessions = sorted({p.session_id for p in pairs})
    # sha256 ordering is stable across processes (unlike hash()) and seed-tunable.
    ordered = sorted(sessions, key=lambda s: hashlib.sha256(f"{seed}:{s}".encode()).hexdigest())
    n_val = 0
    if len(ordered) >= 2 and val_fraction > 0:
        n_val = min(len(ordered) - 1, max(1, round(len(ordered) * val_fraction)))
    val_sessions = set(ordered[:n_val])
    train = [p for p in pairs if p.session_id not in val_sessions]
    val = [p for p in pairs if p.session_id in val_sessions]
    return train, val


# --------------------------------------------------------------------------- #
# Shard export
# --------------------------------------------------------------------------- #
def _call_text(call: dict) -> str:
    return json.dumps({"tool": call["tool"], "args": call["args"]}, sort_keys=True, ensure_ascii=False)


def _prompt(p: RecoveryPair) -> str:
    # The guard reason steers the turn: model sees what failed and why.
    return (
        f"{p.context}\n\n[guard:{p.stage}] The previous tool call was rejected.\n"
        f"Rejected call: {_call_text(p.rejected_call)}\nReason: {p.guard_reason}\n"
        "Emit a corrected tool call."
    )


def _assert_safe_out_dir(out_dir: Path) -> None:
    resolved = out_dir.resolve()
    if "models" in resolved.parts:
        raise ValueError(f"refusing to write shards under a models/ directory: {resolved}")


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    name = path.name
    if (name.startswith("system2") and name.endswith(".pt")) or name.startswith(".best_metric"):
        raise ValueError(f"refusing to write protected file {name}")
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _mlx_row(p: RecoveryPair) -> dict:
    return {
        "messages": [
            {"role": "user", "content": _prompt(p)},
            {"role": "assistant", "content": _call_text(p.eventual_passing_fix)},
        ]
    }


def _dpo_row(p: RecoveryPair) -> dict:
    return {
        "prompt": _prompt(p),
        "chosen": _call_text(p.eventual_passing_fix),
        "rejected": _call_text(p.rejected_call),
        "metadata": {"guard_reason": p.guard_reason, "stage": p.stage, "session_id": p.session_id},
    }


def export_steer_distill_shards(
    train: Sequence[RecoveryPair],
    val: Sequence[RecoveryPair],
    out_dir: Path | str,
    dry_run: bool = False,
) -> dict:
    """Write mlx_lora/{train,valid}.jsonl and dpo/{train,valid}.jsonl.

    mlx_lm expects exactly train.jsonl / valid.jsonl names in the data dir.
    dry_run computes counts without touching disk.
    """
    out = Path(out_dir)
    _assert_safe_out_dir(out)
    mlx = {"train": [_mlx_row(p) for p in train], "valid": [_mlx_row(p) for p in val]}
    dpo = {"train": [_dpo_row(p) for p in train], "valid": [_dpo_row(p) for p in val]}
    if not dry_run:
        for fmt, shards in (("mlx_lora", mlx), ("dpo", dpo)):
            (out / fmt).mkdir(parents=True, exist_ok=True)
            for split, rows in shards.items():
                _write_jsonl(out / fmt / f"{split}.jsonl", rows)
    return {
        "dry_run": dry_run,
        "out_dir": str(out),
        "input_pairs": len(train) + len(val),
        "mlx_lora": {k: len(v) for k, v in mlx.items()},
        "dpo": {k: len(v) for k, v in dpo.items()},
    }


# --------------------------------------------------------------------------- #
# Flip-rate harness
# --------------------------------------------------------------------------- #
def _flips(base: Sequence, cand: Sequence, gold: Sequence) -> tuple[int, int, int]:
    n_flip = improved = degraded = 0
    for b, c, g in zip(base, cand, gold):
        if b != c:
            n_flip += 1
            if c == g:
                improved += 1
            elif b == g:
                degraded += 1
    return n_flip, improved, degraded


def evaluate_flip_rate(
    baseline_runs: Sequence[Sequence],
    candidate_runs: Sequence[Sequence],
    gold: Sequence,
    noise_margin: float = 0.02,
) -> dict:
    """Per-decision flip analysis over >=2 repeat runs each.

    * noise_floor: mean flip rate between every pair of *baseline* repeats
      (how much the engine disagrees with itself).
    * flip_rate: mean flip rate of candidate repeat i vs baseline repeat i.
    * improved/degraded: baseline wrong->candidate right / right->wrong, mean
      per run. net_accuracy_delta can be 0.0 while both are > 0.
    * above_noise_floor: flip_rate > noise_floor + noise_margin.
    """
    if len(baseline_runs) < 2 or len(candidate_runs) < 2:
        raise ValueError("need >=2 repeat runs for baseline and candidate")
    n = len(gold)
    if n == 0 or any(len(r) != n for r in list(baseline_runs) + list(candidate_runs)):
        raise ValueError("all runs must have the same non-zero length as gold")
    pair_rates = [
        _flips(baseline_runs[i], baseline_runs[j], gold)[0] / n
        for i in range(len(baseline_runs))
        for j in range(i + 1, len(baseline_runs))
    ]
    noise_floor = sum(pair_rates) / len(pair_rates)
    k = min(len(baseline_runs), len(candidate_runs))
    tot = [_flips(baseline_runs[i], candidate_runs[i], gold) for i in range(k)]
    flip_rate = sum(t[0] for t in tot) / (k * n)
    improved = sum(t[1] for t in tot) / k
    degraded = sum(t[2] for t in tot) / k

    def acc(runs: Sequence[Sequence]) -> float:
        return sum(sum(1 for p, g in zip(r, gold) if p == g) for r in runs) / (len(runs) * n)

    return {
        "n_decisions": n,
        "repeats": k,
        "flip_rate": flip_rate,
        "noise_floor": noise_floor,
        "improved_flips": improved,
        "degraded_flips": degraded,
        "net_accuracy_delta": acc(candidate_runs) - acc(baseline_runs),
        "above_noise_floor": flip_rate > noise_floor + noise_margin,
    }


# --------------------------------------------------------------------------- #
# Corrective micro-trainer (LoRA vs Deletion Tuning mask)
# --------------------------------------------------------------------------- #
_TOOLS = ["bash", "edit", "write", "read", "grep", "ls"]
# Synthetic guard-reason templates -> correct tool. Held-out shares templates
# but not exact phrasing, so a loss drop on val means generalisation, not recall.
_REASONS = {
    "edit": ["file not read before edit", "stale file content", "edit target missing"],
    "read": ["path unknown inspect first", "need to inspect directory listing", "cannot guess contents"],
    "grep": ["symbol not located search required", "search the repo for definition", "unknown call site"],
    "ls": ["directory existence unverified", "list folder before writing", "path may not exist"],
}


def _hash_features(text: str, dim: int) -> list[float]:
    v = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        v[h % dim] += 1.0 if (h >> 20) & 1 else -1.0
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def _synthetic_smoke_data(seed: int, dim: int, n_corr: int, n_ctrl: int):
    rng = random.Random(seed)
    fixes = list(_REASONS)
    corr = []  # (features, fix_label, rejected_label)
    for i in range(n_corr):
        fix = fixes[i % len(fixes)]
        reason = rng.choice(_REASONS[fix]) + f" case{rng.randrange(50)}"
        rejected = "bash"  # the recurring mistake: shelling out instead of the right tool
        corr.append((_hash_features(f"guard rejected {reason}", dim), _TOOLS.index(fix), _TOOLS.index(rejected)))
    ctrl = []  # clean prompts: no guard reason, label from a different vocabulary
    for _ in range(n_ctrl):
        lab = rng.choice([_TOOLS.index("write"), _TOOLS.index("bash")])
        word = "compose" if lab == _TOOLS.index("write") else "execute"
        ctrl.append((_hash_features(f"task {word} item{rng.randrange(50)}", dim), lab, lab))
    return corr, ctrl


def run_corrective_distill_smoke(
    seed: int = 0, dim: int = 64, rank: int = 4, mask_fraction: float = 0.02, steps: int = 200
) -> dict:
    """Compare LoRA rank-r update vs Deletion-Tuning binary mask on a synthetic shard.

    Base student: linear map dim->tools fitted to repeat the rejected call on
    corrective prompts and to be right on clean controls. Then:
      * LoRA: trains W0 + (alpha/r) B A with replay of clean controls.
      * Deletion mask: zeroes the `mask_fraction` of W0 entries whose removal
        most reduces train corrective loss (first-order w*grad score); no new
        parameters trained.
    Reports held-out corrective CE before/after and clean-control flip rate
    against a noise floor from jittered repeat inference.
    """
    import torch  # lazy: keeps `import sys2.distill` torch-free
    import torch.nn.functional as F

    torch.manual_seed(seed)
    n_tools = len(_TOOLS)
    corr, ctrl = _synthetic_smoke_data(seed, dim, 96, 96)

    def tens(rows):
        x = torch.tensor([r[0] for r in rows], dtype=torch.float32)
        return x, torch.tensor([r[1] for r in rows]), torch.tensor([r[2] for r in rows])

    cx, cy, crej = tens(corr)
    kx, ky, _ = tens(ctrl)
    tr, va = slice(0, 64), slice(64, 96)  # corrective train / held-out
    ctr, cva = slice(0, 64), slice(64, 96)

    # --- base student: makes the recorded mistake on corrective prompts
    w0 = torch.zeros(n_tools, dim, requires_grad=True)
    opt = torch.optim.Adam([w0], lr=0.1)
    bx = torch.cat([cx[tr], kx[ctr]])
    by = torch.cat([crej[tr], ky[ctr]])
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(bx @ w0.T * 4, by).backward()
        opt.step()
    w0 = w0.detach()

    def loss(w, x, y):
        return F.cross_entropy(x @ w.T * 4, y)

    def preds(w, x, jitter_seed):
        g = torch.Generator().manual_seed(jitter_seed)
        # Simulated engine nondeterminism: small logit jitter per repeat run.
        return (x @ w.T * 4 + 0.5 * torch.randn(x.shape[0], n_tools, generator=g)).argmax(1).tolist()

    base = {"heldout_corrective_loss": loss(w0, cx[va], cy[va]).item()}

    # --- LoRA
    gen = torch.Generator().manual_seed(seed + 1)
    a = (torch.randn(rank, dim, generator=gen) * 0.1).requires_grad_()
    b = torch.zeros(n_tools, rank, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=0.05)
    tx = torch.cat([cx[tr], kx[ctr]])
    ty = torch.cat([cy[tr], ky[ctr]])  # corrective target + clean replay
    for _ in range(steps):
        opt.zero_grad()
        loss(w0 + b @ a, tx, ty).backward()
        opt.step()
    w_lora = (w0 + b @ a).detach()

    # --- Deletion Tuning mask
    wg = w0.clone().requires_grad_()
    loss(wg, cx[tr], cy[tr]).backward()
    score = (w0 * wg.grad).flatten()  # >0: zeroing w lowers loss to first order
    k = max(1, int(mask_fraction * score.numel()))
    idx = torch.topk(score, k).indices
    mask = torch.ones_like(w0).flatten()
    mask[idx] = 0.0
    w_mask = w0 * mask.view_as(w0)

    def summarize(w, n_params):
        run_ids = (101, 202, 303)
        base_ctrl = [preds(w0, kx[cva], s) for s in run_ids]
        cand_ctrl = [preds(w, kx[cva], s) for s in run_ids]
        flip = evaluate_flip_rate(base_ctrl, cand_ctrl, ky[cva].tolist())
        fail_base = [preds(w0, cx[va], s) for s in run_ids]
        fail_cand = [preds(w, cx[va], s) for s in run_ids]
        corr_flip = evaluate_flip_rate(fail_base, fail_cand, cy[va].tolist())
        return {
            "trainable_or_changed_params": n_params,
            "heldout_corrective_loss_before": base["heldout_corrective_loss"],
            "heldout_corrective_loss_after": loss(w, cx[va], cy[va]).item(),
            "clean_control_loss_before": loss(w0, kx[cva], ky[cva]).item(),
            "clean_control_loss_after": loss(w, kx[cva], ky[cva]).item(),
            "clean_control_flip": flip,
            "corrective_heldout_flip": corr_flip,
        }

    return {
        "seed": seed,
        "dim": dim,
        "rank": rank,
        "mask_fraction": mask_fraction,
        "weights_total": int(score.numel()),
        "lora": summarize(w_lora, rank * (dim + n_tools)),
        "deletion_mask": summarize(w_mask, int(k)),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python3 -m sys2.distill.steer_distill")
    ap.add_argument("--trajectories-dir", default=str(DEFAULT_TRAJECTORIES_DIR))
    ap.add_argument("--out-dir", default="./steer_distill_out")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    pairs, stats = load_trajectory_pairs(args.trajectories_dir)
    train, val = split_by_session(pairs, args.val_fraction)
    report = export_steer_distill_shards(train, val, args.out_dir, dry_run=args.dry_run)
    report["load_stats"] = stats.as_dict()
    report["elapsed_s"] = time.perf_counter() - t0
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
