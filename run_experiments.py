#!/usr/bin/env python3
"""run_experiments.py - Runs 3 sequential AutoResearch experiments back-to-back.

Test 1: Deeper LoRA (top-14 layers, rank 16)
Test 2: LoRA Rank Expansion (rank 32, alpha 32)
Test 3: Judger Temperature Scaling (learnable tau)
"""
import json
import os
import re
import subprocess
import sys
import time


def set_hyperparam(key, value):
    with open("train.py", "r") as f:
        content = f.read()
    if isinstance(value, str):
        val_str = f'"{value}"'
    elif isinstance(value, bool):
        val_str = "True" if value else "False"
    else:
        val_str = str(value)
    new_content = re.sub(rf'"{key}":\s*[^,\n]+,', f'"{key}": {val_str},', content)
    with open("train.py", "w") as f:
        f.write(new_content)


def get_best_metric():
    with open(".best_metric", "r") as f:
        return float(f.read().strip())


def run_experiment(name, hyperparams, description):
    print(f"\n{'='*75}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Description: {description}")
    print(f"Hyperparams to apply: {hyperparams}")
    print(f"Current best metric threshold: {get_best_metric()}")
    print(f"{'='*75}\n", flush=True)

    # Apply hyperparams
    for k, v in hyperparams.items():
        set_hyperparam(k, v)

    start_time = time.time()
    # Execute runner.sh
    proc = subprocess.run(["./runner.sh", description], capture_output=True, text=True)
    duration = time.time() - start_time

    stdout = proc.stdout
    stderr = proc.stderr
    print(stdout)
    if stderr:
        print("STDERR:", stderr, file=sys.stderr)

    # Parse outcome
    committed = "BETTER:" in stdout
    reverted = "WORSE or equal:" in stdout

    # Extract metric line if available
    metric_match = re.search(r'\{"metric":\s*([0-9.]+),\s*"accuracy":\s*([0-9.]+),\s*"brier":\s*([0-9.]+)\}', stdout)
    if metric_match:
        metric = float(metric_match.group(1))
        acc = float(metric_match.group(2))
        brier = float(metric_match.group(3))
    else:
        metric, acc, brier = None, None, None

    result = {
        "name": name,
        "description": description,
        "hyperparams": hyperparams,
        "metric": metric,
        "accuracy": acc,
        "brier": brier,
        "committed": committed,
        "duration_sec": round(duration, 1),
        "new_best_metric": get_best_metric(),
    }
    print(f"\n>>> FINISHED {name}: Metric={metric}, Acc={acc}, Brier={brier}, Committed={committed} ({duration:.1f}s)\n", flush=True)
    return result


def main():
    results = []

    # Test 1: Deeper LoRA (top-14 layers, rank 16)
    r1 = run_experiment(
        name="Test 1: Deeper LoRA (top-14 layers)",
        hyperparams={"lora_layers": 14, "lora_r": 16, "lora_alpha": 16, "learnable_temp": False},
        description="Phase 2: Frozen ModernBERT + Deeper LoRA (top-14 layers, rank 16)",
    )
    results.append(r1)

    # Test 2: LoRA Rank Expansion (rank 32, alpha 32)
    r2 = run_experiment(
        name="Test 2: LoRA Rank Expansion (r=32, alpha=32)",
        hyperparams={"lora_r": 32, "lora_alpha": 32, "learnable_temp": False},
        description="Phase 2: LoRA Rank Expansion (rank 32, alpha 32)",
    )
    results.append(r2)

    # Test 3: Judger Temperature Scaling
    r3 = run_experiment(
        name="Test 3: Judger Temperature Scaling (learnable tau)",
        hyperparams={"learnable_temp": True},
        description="Phase 2: Judger Temperature Scaling (learnable tau)",
    )
    results.append(r3)

    print("\n" + "=" * 80)
    print("ALL 3 EXPERIMENTS COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Experiment':<44} | {'Metric':<8} | {'Accuracy':<10} | {'Brier':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        status = "COMMITTED (NEW SOTA 🏆)" if r["committed"] else "REVERTED"
        m_str = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        a_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        b_str = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
        print(f"{r['name']:<44} | {m_str:<8} | {a_str:<10} | {b_str:<8} | {status}")
    print("=" * 80)
    print(f"Final .best_metric: {get_best_metric()}\n")

    with open("experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
