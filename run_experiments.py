#!/usr/bin/env python3
"""run_experiments.py - Runs sequential AutoResearch experiments back-to-back.

Battery 8: Cognitive Spectrum Deliberation & Dynamic Arbitration Battery
- Test 1: Constraint-Spectrum Deliberation (asym_span_mode="constraint_spectrum", head_weights=[1.0, 1.0, 1.2, 1.0, 1.0], tau=1.80)
- Test 2: Probability Margin Decisive Arbitration (prob_margin_gamma=0.08, tau=1.80)
- Test 3: Authentic Candidate-Permuted Regularization (permute_choices=True, tau=1.80)
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
    elif value is None:
        val_str = "None"
    else:
        val_str = str(value)

    new_content = re.sub(
        rf'("{key}":\s*).*?(,\s*(?:#.*)?)$',
        rf'\g<1>{val_str}\g<2>',
        content,
        flags=re.MULTILINE,
    )
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

    # Active SOTA baseline: 0.6697 (Accuracy 58.30%, Brier 0.2435)
    # Battery 7: Pure In-Domain Causal Data Augmentation & Calibrated Arbitration Battery

    # Base SOTA parameters
    base_params = {
        "use_mote": False,
        "data_dir": "./data/arc",
        "train_size": 4300,
        "epochs": 5,
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
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "lambda_judge": 0.10,
        "lambda_cal": 0.50,
        "lambda_judge_cal": 0.06,
        "oracle_judge_target": False,
        "detach_judger": True,
        "shuffle_per_epoch": False,
        "seed": 0,
        "label_smoothing": 0.15,
        "soft_judge_targets": False,
        "ensemble_heads": True,
        "centered_conf": False,
        "judge_cal_smoothing": 0.30,
        "lambda_ens_ce": 0.0,
        "uniform_ensemble": False,
        "ensemble_temp": 1.80,
        "lambda_head_balance": 0.0,
        "margin_features": False,
        "span_pooling": False,
        "lora_dropout": 0.0,
        "head_dropout": 0.0,
        "head_label_smoothing": 0.08,
        "margin_gamma": 0.0,
        "prob_margin_gamma": 0.0,
        "prob_ensemble": False,
        "head_prob_temp": 1.0,
        "head_norm_first": False,
        "judger_residual": False,
        "head_mlp_dim": 256,
        "head_norm_diff": False,
        "lambda_cand_margin": 0.05,
        "cand_margin": 1.0,
        "veto_blend": 0.0,
        "permute_choices": False,
        "asym_roles": False,
        "asym_spans": True,
        "asym_span_mode": "choice_forward",
        "head_weights": [1.0, 1.0, 1.2, 1.0, 0.4],
        "train_head_weights": None,
    }
    # Test 1: Constraint-Spectrum Multi-Focal Deliberation
    t1_params = dict(base_params)
    t1_params["asym_span_mode"] = "constraint_spectrum"
    t1_params["head_weights"] = [1.0, 1.0, 1.2, 1.0, 1.0]
    r1 = run_experiment(
        name="Test 1: Constraint-Spectrum Deliberation (constraint_spectrum, tau=1.80)",
        hyperparams=t1_params,
        description="Battery 8: Constraint-Spectrum Deliberation (constraint_spectrum, tau=1.80)",
    )
    results.append(r1)

    if r1["committed"]:
        print(f">>> Test 1 committed as new SOTA! Retaining improvements.")
        base_params = dict(t1_params)
    else:
        print(f">>> Test 1 reverted.")

    # Test 2: Probability Margin Decisive Arbitration (prob_margin_gamma=0.08, tau=1.80)
    t2_params = dict(base_params)
    t2_params["prob_margin_gamma"] = 0.08
    r2 = run_experiment(
        name="Test 2: Probability Margin Decisive Arbitration (prob_margin_gamma=0.08, tau=1.80)",
        hyperparams=t2_params,
        description="Battery 8: Probability Margin Decisive Arbitration (prob_margin_gamma=0.08, tau=1.80)",
    )
    results.append(r2)

    if r2["committed"]:
        print(f">>> Test 2 committed as new SOTA! Retaining improvements.")
        base_params = dict(t2_params)
    else:
        print(f">>> Test 2 reverted.")

    # Test 3: Authentic Candidate-Permuted Regularization (permute_choices=True, tau=1.80)
    t3_params = dict(base_params)
    t3_params["permute_choices"] = True
    r3 = run_experiment(
        name="Test 3: Authentic Candidate-Permuted Regularization (permute_choices=True, tau=1.80)",
        hyperparams=t3_params,
        description="Battery 8: Authentic Candidate-Permuted Regularization (permute_choices=True, tau=1.80)",
    )
    results.append(r3)

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED - SUMMARY REPORT")
    print("=" * 80)
    print(f"{'Experiment':<52} | {'Metric':<8} | {'Accuracy':<10} | {'Brier':<8} | {'Status'}")
    print("-" * 80)
    for r in results:
        status = "COMMITTED (NEW SOTA 🏆)" if r["committed"] else "REVERTED"
        m_str = f"{r['metric']:.4f}" if r["metric"] is not None else "N/A"
        a_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        b_str = f"{r['brier']:.4f}" if r["brier"] is not None else "N/A"
        print(f"{r['name']:<52} | {m_str:<8} | {a_str:<10} | {b_str:<8} | {status}")
    print("=" * 80)
    print(f"Final .best_metric: {get_best_metric():.4f}\n")

    try:
        with open("experiments_summary.json", "r") as f:
            all_history = json.load(f)
            if not isinstance(all_history, list):
                all_history = []
    except Exception:
        all_history = []
    all_history.extend(results)
    with open("experiments_summary.json", "w") as f:
        json.dump(all_history, f, indent=2)


if __name__ == "__main__":
    main()
