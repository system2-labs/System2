#!/usr/bin/env python3
"""scripts/run_eval_suite.py - Unified 13-Track Benchmark & Calibration Suite CLI.

Evaluates Oracle Routing and End-to-End Dynamic Routing across all 13 specialized tracks:
- Accuracy, Binary Brier, Multiclass Brier, 10-bin ECE, Guarded Metric
- Mean & p95 Routing Latency (ms) and Deliberation Latency (ms)
- Inter-Head Disagreement & Consensus Distribution
- Grounded comparison against README reference metrics
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import numpy as np
import torch

from evals.common import (
    README_REFERENCE,
    TRACK_DATA_MAP,
    load_track_samples,
    resolve_device,
    set_seed,
)
from evals.metrics import (
    compute_accuracy,
    compute_binary_brier,
    compute_consensus_stats,
    compute_ece,
    compute_guarded_metric,
    compute_latency_stats,
    compute_multiclass_brier,
)
from evals.reporting import (
    get_system_metadata,
    render_markdown_table,
    save_json_report,
)
from routing.domain_router import DomainRouter
from sys2.engine import DeliberationResult, System2Engine


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified System2 13-Track Evaluation & Calibration Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tracks",
        default="all",
        help="Comma-separated track names to evaluate, or 'all' for all 13 tracks",
    )
    parser.add_argument(
        "--samples-per-track",
        type=int,
        default=100,
        help="Number of held-out test samples per track (0 or negative for all samples)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Target device: mps, cpu, cuda (defaults to auto-detecting Apple Silicon MPS)",
    )
    parser.add_argument(
        "--modes",
        default="oracle,e2e",
        help="Evaluation modes to run: 'oracle', 'e2e', or 'oracle,e2e'",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=1024,
        help="Maximum sequence length ceiling for ModernBERT pairwise mean pooling",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic test subsampling",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT_DIR, "reports", "eval_suite_report.json"),
        help="Path to write structured JSON evaluation report",
    )
    return parser.parse_args()


def evaluate_track_mode(
    engine: System2Engine,
    track: str,
    samples: List[Dict[str, Any]],
    mode: str,
    max_len: int = 1024,
) -> Dict[str, Any]:
    """Evaluate a single track under oracle or end-to-end dynamic routing."""
    preds = []
    targets = []
    confs = []
    probs_list = []
    head_choices_list = []
    routing_latencies = []
    deliberation_latencies = []
    routed_tracks = []
    routing_methods = []

    # Warmup pass (untimed)
    if samples:
        first = samples[0]
        _ = engine.adjudicate(
            first["question"],
            first["choices"],
            domain=track if mode == "oracle" else None,
            max_len=max_len,
        )
        if engine.device == "mps":
            torch.mps.synchronize()

    for s in samples:
        q = s["question"]
        choices = s["choices"]
        gold = s["label"]

        if engine.device == "mps":
            torch.mps.synchronize()

        domain_arg = track if mode == "oracle" else None
        res: DeliberationResult = engine.adjudicate(q, choices, domain=domain_arg, max_len=max_len)

        if engine.device == "mps":
            torch.mps.synchronize()

        preds.append(res.selected_index)
        targets.append(gold)
        confs.append(res.confidence)
        probs_list.append(res.choice_probabilities or [0.25, 0.25, 0.25, 0.25])
        head_choices_list.append(res.per_head_choices)
        routing_latencies.append(res.routing_latency_ms)
        deliberation_latencies.append(res.deliberation_latency_ms)
        routed_tracks.append(res.domain)
        routing_methods.append(res.routing_method)

    confs_t = torch.tensor(confs, dtype=torch.float32)
    is_cor_t = torch.tensor([1.0 if p == t else 0.0 for p, t in zip(preds, targets)], dtype=torch.float32)

    acc = compute_accuracy(preds, targets)
    brier = compute_binary_brier(confs_t, is_cor_t)
    mc_brier = compute_multiclass_brier(probs_list, targets)
    ece = compute_ece(confs_t, is_cor_t, n_bins=10)
    guarded = compute_guarded_metric(acc, brier)

    route_lat = compute_latency_stats(routing_latencies)
    delib_lat = compute_latency_stats(deliberation_latencies)
    consensus_stats = compute_consensus_stats(head_choices_list, preds, targets)

    # In E2E mode, compute domain router precision
    routing_acc = (
        sum(1 for rt in routed_tracks if rt == track) / len(routed_tracks)
        if routed_tracks
        else 1.0
    )

    return {
        "track": track,
        "mode": mode,
        "num_samples": len(samples),
        "accuracy": round(acc, 4),
        "binary_brier": round(brier, 4),
        "multiclass_brier": round(mc_brier, 4),
        "ece_10bin": round(ece, 4),
        "guarded_metric": round(guarded, 4),
        "routing_accuracy": round(routing_acc, 4) if mode == "e2e" else 1.0,
        "routing_latency_ms": route_lat,
        "deliberation_latency_ms": delib_lat,
        "consensus": consensus_stats,
    }


def main():
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)

    print(f"\n================================================================================", file=sys.stderr)
    print(f" SYSTEM 2 UNIFIED 13-TRACK BENCHMARK & CALIBRATION SUITE", file=sys.stderr)
    print(f" Device: {device.upper()} | Samples/Track: {args.samples_per_track} | Seed: {args.seed}", file=sys.stderr)
    print(f"================================================================================\n", file=sys.stderr)

    # Determine tracks to evaluate
    all_tracks = list(TRACK_DATA_MAP.keys())
    if args.tracks.strip().lower() == "all":
        selected_tracks = all_tracks
    else:
        selected_tracks = [t.strip().lower() for t in args.tracks.split(",") if t.strip()]
        for t in selected_tracks:
            if t not in TRACK_DATA_MAP:
                raise ValueError(f"Unknown track '{t}'. Available tracks: {all_tracks}")

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    n_samples = args.samples_per_track if args.samples_per_track > 0 else None

    # Initialize System2Engine with shared ModernBERT backbone
    engine = System2Engine(device=device)

    # Run benchmarks
    results_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}

    for track in selected_tracks:
        print(f"--> Ingesting held-out split for track: [{track.upper()}]...", file=sys.stderr)
        samples = load_track_samples(track, n=n_samples, seed=args.seed)

        for mode in modes:
            print(f"    Evaluating mode: {mode.upper()} ({len(samples)} samples)...", file=sys.stderr)
            track_res = evaluate_track_mode(
                engine=engine,
                track=track,
                samples=samples,
                mode=mode,
                max_len=args.max_len,
            )
            results_by_mode[mode].append(track_res)

    # Generate Markdown Summary Tables for stdout
    headers = [
        "Track",
        "Mode",
        "Guarded",
        "README",
        "Delta",
        "Raw Acc",
        "Brier",
        "ECE",
        "Route p95",
        "Delib p95",
        "Disagree",
    ]

    report_payload: Dict[str, Any] = {
        "metadata": get_system_metadata(device),
        "config": {
            "tracks": selected_tracks,
            "samples_per_track": args.samples_per_track,
            "modes": modes,
            "max_len": args.max_len,
            "seed": args.seed,
        },
        "results": results_by_mode,
        "macro_averages": {},
    }

    markdown_sections = []

    for mode in modes:
        rows = []
        mode_records = results_by_mode[mode]

        guarded_list = [r["guarded_metric"] for r in mode_records]
        acc_list = [r["accuracy"] for r in mode_records]
        brier_list = [r["binary_brier"] for r in mode_records]
        ece_list = [r["ece_10bin"] for r in mode_records]
        delib_p95_list = [r["deliberation_latency_ms"]["p95_ms"] for r in mode_records]
        route_p95_list = [r["routing_latency_ms"]["p95_ms"] for r in mode_records]
        disagree_list = [r["consensus"]["inter_head_disagreement"] for r in mode_records]

        for r in mode_records:
            t = r["track"]
            ref = README_REFERENCE.get(t, {}).get("guarded", 0.0)
            delta = r["guarded_metric"] - ref
            delta_str = f"{delta:+.4f}"

            rows.append([
                t,
                mode,
                f"{r['guarded_metric']:.4f}",
                f"{ref:.4f}",
                delta_str,
                f"{r['accuracy'] * 100:.1f}%",
                f"{r['binary_brier']:.4f}",
                f"{r['ece_10bin']:.4f}",
                f"{r['routing_latency_ms']['p95_ms']:.2f}ms",
                f"{r['deliberation_latency_ms']['p95_ms']:.2f}ms",
                f"{r['consensus']['inter_head_disagreement']:.3f}",
            ])

        # Macro average row
        macro_avg = {
            "guarded_metric": round(float(np.mean(guarded_list)), 4),
            "accuracy": round(float(np.mean(acc_list)), 4),
            "binary_brier": round(float(np.mean(brier_list)), 4),
            "ece_10bin": round(float(np.mean(ece_list)), 4),
            "deliberation_p95_ms": round(float(np.mean(delib_p95_list)), 2),
            "routing_p95_ms": round(float(np.mean(route_p95_list)), 2),
            "inter_head_disagreement": round(float(np.mean(disagree_list)), 4),
        }
        report_payload["macro_averages"][mode] = macro_avg

        rows.append([
            "MACRO_AVG",
            mode,
            f"{macro_avg['guarded_metric']:.4f}",
            "-",
            "-",
            f"{macro_avg['accuracy'] * 100:.1f}%",
            f"{macro_avg['binary_brier']:.4f}",
            f"{macro_avg['ece_10bin']:.4f}",
            f"{macro_avg['routing_p95_ms']:.2f}ms",
            f"{macro_avg['deliberation_p95_ms']:.2f}ms",
            f"{macro_avg['inter_head_disagreement']:.3f}",
        ])

        table_md = render_markdown_table(headers, rows)
        markdown_sections.append(f"### Mode: {mode.upper()} Benchmarks\n\n{table_md}\n")

    full_markdown = "\n".join(markdown_sections)
    print("\n" + full_markdown)

    save_json_report(args.output, report_payload)
    print(f"\n[OK] Structured JSON evaluation report saved to: {args.output}\n", file=sys.stderr)


if __name__ == "__main__":
    main()
