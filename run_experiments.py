#!/usr/bin/env python3
"""run_experiments.py - Runs 2 sequential AutoResearch experiments back-to-back.

Test 1: Pure In-Domain ARC Data Scaling (4.2k train samples with validation split, epochs=4, cosine)
Test 2: Context Truncation Elimination (max_len=160, 4.2k train samples, epochs=4, cosine)
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
    
    m = re.search(r'(HYPERPARAMS\s*=\s*\{.*?\n\})', content, flags=re.DOTALL)
    if m:
        block = m.group(1)
        new_block = re.sub(rf'"{key}":\s*[^,\n]+,', f'"{key}": {val_str},', block)
        new_content = content[:m.start(1)] + new_block + content[m.end(1):]
    else:
        new_content = re.sub(rf'"{key}":\s*[^,\n]+,', f'"{key}": {val_str},', content, count=1)
        
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
    print(f"Current best metric threshold: {get_best_metric():.4f}")
    print(f"{'='*75}\n", flush=True)

    # Apply hyperparams
    for k, v in hyperparams.items():
        set_hyperparam(k, v)

    start_time = time.time()
    # Execute runner.sh with real-time streaming
    proc = subprocess.Popen(
        ["./runner.sh", description],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    stdout_lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        stdout_lines.append(line)
    proc.wait()
    duration = time.time() - start_time
    stdout = "".join(stdout_lines)

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

    # Active SOTA baseline state (56.63% accuracy, 0.2458 Brier, 0.6602 metric)
    active_train_size = 3500

    # Test 1: Pure In-Domain ARC Data Scaling (4.2k train samples)
    t1_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": 4300,
        "epochs": 4,
        "batch_size": 16,
        "max_len": 128,
        "num_heads": 5,
        "head_layers": 2,
        "use_lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_layers": 8,
        "lr_heads": 1e-3,
        "lr_judger": 3e-4,
        "lr_lora": 2e-4,
        "lambda_div": 0.01,
        "pairwise": True,
        "diff_features": True,
        "task_prefixes": False,
        "learnable_temp": False,
        "use_scheduler": True,
        "scheduler_min_lr": 1e-5,
        "soft_judge_targets": False,
    }
    r1 = run_experiment(
        name="Test 1: In-Domain Data Scaling (4.2k samples)",
        hyperparams=t1_params,
        description="Phase 2: In-Domain Data Scaling (4.2k train samples with validation split, 4 epochs, cosine)",
    )
    results.append(r1)

    if r1["committed"]:
        active_train_size = 4300
        print(f">>> Test 1 committed as new SOTA! Adopting train_size={active_train_size} for subsequent test.")
    else:
        print(f">>> Test 1 reverted. Retaining train_size={active_train_size}.")

    # Test 2: Context Truncation Elimination (max_len = 160)
    t2_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": active_train_size,
        "epochs": 4,
        "batch_size": 16,
        "max_len": 160,
        "num_heads": 5,
        "head_layers": 2,
        "use_lora": True,
        "lora_r": 32,
        "lora_alpha": 32,
        "lora_layers": 8,
        "lr_heads": 1e-3,
        "lr_judger": 3e-4,
        "lr_lora": 2e-4,
        "lambda_div": 0.01,
        "pairwise": True,
        "diff_features": True,
        "task_prefixes": False,
        "learnable_temp": False,
        "use_scheduler": True,
        "scheduler_min_lr": 1e-5,
        "soft_judge_targets": False,
    }
    r2 = run_experiment(
        name="Test 2: Context Truncation Elimination (max_len=160)",
        hyperparams=t2_params,
        description="Phase 2: Context Truncation Elimination (max_len=160, 4 epochs, cosine)",
    )
    results.append(r2)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Experiment':<48} | {'Metric':<8} | {'Accuracy':<10} | {'Brier':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        status = "COMMITTED (NEW SOTA 🏆)" if r["committed"] else "REVERTED"
        m_str = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        a_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        b_str = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
        print(f"{r['name']:<48} | {m_str:<8} | {a_str:<10} | {b_str:<8} | {status}")
    print("=" * 80)
    print(f"Final .best_metric: {get_best_metric():.4f}\n")

    with open("experiments_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
