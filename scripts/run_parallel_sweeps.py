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
    "track-d": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-d"),
        "runner": "./runner_electrical.sh",
        "best_file": ".best_metric_electrical",
        "trials": [
            {
                "desc": "trial D-4: pure electrical dataset with baseline regularized regime and ensemble_temp 0.50",
                "params": {
                    "ensemble_temp": "0.50",
                },
            },
            {
                "desc": "trial D-5: pure electrical dataset with causal_attn span pooling and ensemble_temp 0.50",
                "params": {
                    "asym_span_mode": '"causal_attn"',
                    "ensemble_temp": "0.50",
                },
            },
        ],
    },
    "track-i": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-i"),
        "runner": "./runner_finance.sh",
        "best_file": ".best_metric_finance",
        "trials": [
            {
                "desc": "trial I-2: causal_attn span pooling with ensemble_temp 0.50",
                "params": {
                    "asym_span_mode": '"causal_attn"',
                    "ensemble_temp": "0.50",
                },
            },
            {
                "desc": "trial I-3: ensemble_temp 0.45 and decision_conf_tau 0.65",
                "params": {
                    "ensemble_temp": "0.45",
                    "decision_conf_tau": "0.65",
                },
            },
            {
                "desc": "trial I-4: lambda_cand_margin 0.08 with ensemble_temp 0.50",
                "params": {
                    "lambda_cand_margin": "0.08",
                    "ensemble_temp": "0.50",
                },
            },
        ],
    },
    "track-h": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-h"),
        "runner": "./runner_legal.sh",
        "best_file": ".best_metric_legal",
        "trials": [
            {
                "desc": "trial H-3: ensemble_temp 0.50 with lambda_cand_margin 0.08",
                "params": {
                    "ensemble_temp": "0.50",
                    "lambda_cand_margin": "0.08",
                },
            },
            {
                "desc": "trial H-4: causal_attn span pooling with ensemble_temp 0.50 and lambda_cand_margin 0.08",
                "params": {
                    "asym_span_mode": '"causal_attn"',
                    "ensemble_temp": "0.50",
                    "lambda_cand_margin": "0.08",
                },
            },
            {
                "desc": "trial H-5: decision_conf_tau 0.65 and soft_target_tau 0.45",
                "params": {
                    "decision_conf_tau": "0.65",
                    "soft_target_tau": "0.45",
                },
            },
        ],
    },
    "track-c": {
        "path": os.path.join(ROOT_DIR, "scratch", "worktrees", "track-c"),
        "runner": "./runner_med.sh",
        "best_file": ".best_metric_med",
        "trials": [
            {
                "desc": "trial C-1: causal_attn span pooling with ensemble_temp 0.50 on USMLE vignettes",
                "params": {
                    "asym_span_mode": '"causal_attn"',
                    "ensemble_temp": "0.50",
                },
            },
            {
                "desc": "trial C-2: lambda_cand_margin 0.08 and ensemble_temp 0.50 on USMLE vignettes",
                "params": {
                    "lambda_cand_margin": "0.08",
                    "ensemble_temp": "0.50",
                },
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

    # 1. Apply hyperparameter changes
    train_py = os.path.join(wpath, "train.py")
    if "params" in trial:
        for k, v in trial["params"].items():
            modify_hyperparam(train_py, k, v)
    elif "param" in trial and "val" in trial:
        modify_hyperparam(train_py, trial["param"], trial["val"])

    # 2. Run runner script with live line-by-line streaming and process-group cleanup
    import signal
    cmd = [runner, desc]
    stdout_lines = []
    proc = subprocess.Popen(
        cmd,
        cwd=wpath,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            sline = line.rstrip()
            if sline.startswith("epoch ") or sline.startswith("[runner") or "BETTER:" in sline or "WORSE" in sline:
                print(f"  {sline}", flush=True)
            if time.perf_counter() - t0 > 3000:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                raise TimeoutError(f"Trial exceeded 3000s timeout: {desc}")
        proc.wait(timeout=60)
        elapsed_sec = round(time.perf_counter() - t0, 1)
        new_best = read_metric(wpath, best_file)

        improved = (new_best is not None and initial_best is not None and new_best > initial_best)

        status_str = "✓ IMPROVEMENT COMMITTED" if improved else "✗ REVERTED (no improvement)"
        print(f"[{worktree.upper()}] Trial Complete in {elapsed_sec}s -> {status_str}", flush=True)
        print(f"[{worktree.upper()}] Current Best: {new_best}", flush=True)

        return {
            "worktree": worktree,
            "desc": desc,
            "initial_best": initial_best,
            "final_best": new_best,
            "improved": improved,
            "elapsed_sec": elapsed_sec,
            "stdout_tail": "".join(stdout_lines)[-500:],
        }
    except Exception as e:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass
        subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=wpath, capture_output=True)
        print(f"[{worktree.upper()}] Trial Failed with Exception: {e}", file=sys.stderr, flush=True)
        return {
            "worktree": worktree,
            "desc": desc,
            "error": str(e),
            "improved": False,
        }


def main():
    parser = argparse.ArgumentParser(description="System2 Parallel Worktree AutoResearch Orchestrator")
    parser.add_argument("--tracks", nargs="+", default=["track-d", "track-i", "track-h", "track-c"], choices=["track-d", "track-i", "track-h", "track-c"])
    parser.add_argument("--max-trials", type=int, default=3, help="Max trials to run per track")
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
