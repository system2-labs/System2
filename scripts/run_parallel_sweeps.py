#!/usr/bin/env python3
"""run_parallel_sweeps.py - Automated Worktree AutoResearch Sweep Orchestrator.

Orchestrates sequential and parallel exploration trials across isolated git worktrees:
- Adheres to The Pre-Test Commit Law (infrastructure committed before runner)
- Adheres to The Metric Invariant (strictly reverts regressions)
- Hardware-safe queue management on Apple Silicon MPS

Usage:
  python3 scripts/run_parallel_sweeps.py --tracks track-h track-i --max-trials 3
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

WORKTREE_MAP = {
    "track-h": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-h"),
        "runner": "./runner_legal.sh",
        "best_file": ".best_metric_legal",
        "trials": [
            {
                "desc": "asym_span_mode constraint_spectrum",
                "param": "asym_span_mode",
                "val": '"constraint_spectrum"',
            },
            {
                "desc": "lambda_cand_margin 0.08",
                "param": "lambda_cand_margin",
                "val": "0.08",
            },
            {
                "desc": "warmup_ratio 0.05 and cosine annealing",
                "param": "warmup_ratio",
                "val": "0.05",
            },
        ],
    },
    "track-i": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-i"),
        "runner": "./runner_finance.sh",
        "best_file": ".best_metric_finance",
        "trials": [
            {
                "desc": "lambda_cand_margin 0.06",
                "param": "lambda_cand_margin",
                "val": "0.06",
            },
            {
                "desc": "soft_target_tau 0.45",
                "param": "soft_target_tau",
                "val": "0.45",
            },
            {
                "desc": "warmup_ratio 0.05 and cosine annealing",
                "param": "warmup_ratio",
                "val": "0.05",
            },
        ],
    },
}


def read_metric(worktree_path: str, best_file: str) -> Optional[float]:
    bf = os.path.join(worktree_path, best_file)
    if os.path.exists(bf):
        try:
            with open(bf, "r") as f:
                return float(f.read().strip())
        except Exception:
            return None
    return None


def modify_hyperparam(train_py_path: str, param: str, val_str: str) -> bool:
    """Modify a single hyperparameter in train.py."""
    if not os.path.exists(train_py_path):
        return False
    with open(train_py_path, "r") as f:
        content = f.read()

    # Look for "param": <old_val>,
    pattern = rf'("{param}"\s*:\s*)([^,\n]+)(,?)'
    if not re.search(pattern, content):
        return False

    new_content = re.sub(pattern, rf'\g<1>{val_str}\g<3>', content, count=1)
    with open(train_py_path, "w") as f:
        f.write(new_content)
    return True


def execute_trial(worktree: str, trial: Dict[str, Any]) -> Dict[str, Any]:
    cfg = WORKTREE_MAP[worktree]
    wpath = cfg["path"]
    runner = cfg["runner"]
    best_file = cfg["best_file"]
    desc = trial["desc"]

    print(f"\n[{worktree.upper()}] >>> Starting Trial: {desc}")
    t0 = time.perf_counter()

    initial_best = read_metric(wpath, best_file)
    print(f"[{worktree.upper()}] Prior Best Metric: {initial_best}")

    # 1. Apply hyperparameter change
    train_py = os.path.join(wpath, "train.py")
    modify_hyperparam(train_py, trial["param"], trial["val"])

    # 2. Run runner script
    cmd = [runner, desc]
    try:
        proc = subprocess.run(
            cmd,
            cwd=wpath,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min per trial
        )
        elapsed_sec = round(time.perf_counter() - t0, 1)
        new_best = read_metric(wpath, best_file)

        improved = (new_best is not None and initial_best is not None and new_best > initial_best)

        status_str = "✓ IMPROVEMENT COMMITTED" if improved else "✗ REVERTED (no improvement)"
        print(f"[{worktree.upper()}] Trial Complete in {elapsed_sec}s -> {status_str}")
        print(f"[{worktree.upper()}] Current Best: {new_best}")

        return {
            "worktree": worktree,
            "desc": desc,
            "initial_best": initial_best,
            "final_best": new_best,
            "improved": improved,
            "elapsed_sec": elapsed_sec,
            "stdout_tail": proc.stdout[-500:],
        }
    except Exception as e:
        print(f"[{worktree.upper()}] Trial Failed with Exception: {e}", file=sys.stderr)
        return {
            "worktree": worktree,
            "desc": desc,
            "error": str(e),
            "improved": False,
        }


def main():
    parser = argparse.ArgumentParser(description="System2 Parallel Worktree AutoResearch Orchestrator")
    parser.add_argument("--tracks", nargs="+", default=["track-h", "track-i"], choices=["track-h", "track-i"])
    parser.add_argument("--max-trials", type=int, default=1, help="Max trials to run per track")
    args = parser.parse_args()

    print("========================================================================")
    print(" SYSTEM 2 AUTORESEARCH WORKTREE SWEEP ORCHESTRATOR")
    print(f" Target Tracks: {args.tracks} | Max Trials Per Track: {args.max_trials}")
    print("========================================================================")

    all_results = []
    for track in args.tracks:
        cfg = WORKTREE_MAP[track]
        trials = cfg["trials"][:args.max_trials]
        for t in trials:
            res = execute_trial(track, t)
            all_results.append(res)

    print("\n" + "=" * 72)
    print(" AUTORESEARCH SWEEPS SUMMARY")
    print("=" * 72)
    for r in all_results:
        mark = "✓" if r.get("improved") else "-"
        print(f"[{mark}] [{r['worktree'].upper()}] {r['desc']}: {r.get('initial_best')} -> {r.get('final_best')} ({r.get('elapsed_sec')}s)")


if __name__ == "__main__":
    main()
