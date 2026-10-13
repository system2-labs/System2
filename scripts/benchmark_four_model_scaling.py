#!/usr/bin/env python3
"""benchmark_four_model_scaling.py - 4-Tier Model Parameter Scaling Deliberation Benchmark.

Compares:
1. Tier 1 (Ultra-Lightweight ~4B): google/gemma-4-e4b
2. Tier 2 (Mid-Weight 12B): google/gemma-4-12b
3. Tier 3 (High-Capacity 27B): qwen3.8-27b-splash
4. Tier 4 (Frontier Reference 35B): qwen3.6-35b-a3b-splash

Protocol:
- 4 strictly independent, zero-knowledge single-turn LLM calls (temp=0.7) per battery.
- Candidates adjudicated by System 2 (ModernBERT + 5 ThoughtHeads + Judger) on CPU
  to ensure zero GPU VRAM contention with background Track E/F training on MPS.
"""

import os
import sys
import json
import time
import requests
from typing import Dict, List, Any

ROOT_DIR = "/Users/thomjames-worrall/System2"
SYS2_DIR = "/Users/thomjames-worrall/Sys2"
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SYS2_DIR not in sys.path:
    sys.path.insert(0, SYS2_DIR)

from engine.system2_runtime import System2Runtime

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"

MODELS_TO_TEST = [
    {
        "id": "google/gemma-4-e4b",
        "name": "Google Gemma-4 e4b (Ultra-Lightweight ~4B)",
        "tier": "4B",
    },
    {
        "id": "google/gemma-4-12b",
        "name": "Google Gemma-4 12B (Mid-Weight 12B)",
        "tier": "12B",
    },
    {
        "id": "qwen3.8-27b-splash",
        "name": "Qwen-3.8 27B Splash (High-Capacity 27B)",
        "tier": "27B",
    },
    {
        "id": "qwen3.6-35b-a3b-splash",
        "name": "Qwen-3.6 35B Splash (Frontier Reference 35B)",
        "tier": "35B",
    },
]

BENCHMARK_PROBLEMS = [
    {
        "id": "sourdough_trap",
        "title": "Adversarial Hallucination Trap (Food Science / Toxicity)",
        "track": "science",
        "prompt": "Can authentic sourdough bread be baked using industrial glue, Portland cement, and crushed bricks? Explain precisely why or why not.",
        "expected_verdict": "Strict rejection: toxic non-edible materials cannot produce sourdough (requires flour, water, salt, yeast fermentation).",
    },
    {
        "id": "python_mutable_default",
        "title": "Deterministic Polyglot Code Trace (Python Mutable Default)",
        "track": "code",
        "prompt": (
            "What is the exact stdout output of this Python code, and why?\n\n"
            "def mutate(val, arr=[]):\n"
            "    arr.append(val)\n"
            "    return arr\n\n"
            "print(mutate(1))\n"
            "print(mutate(2))\n"
            "print(mutate(3))"
        ),
        "expected_verdict": "Exact output is [1], [1, 2], [1, 2, 3] because default list arr is created once at def-time and persists across calls.",
    },
    {
        "id": "three_phase_y",
        "title": "Electrical Network Phasor Theorem (3-Phase Line vs Phase Voltage)",
        "track": "electrical",
        "prompt": "In a balanced three-phase Y-connected AC system with positive ABC sequence, what is the exact magnitude and phase angle relationship between the line-to-line voltage V_ab and the phase voltage V_an?",
        "expected_verdict": "Magnitude V_ab = sqrt(3) * V_an (~1.732 V_an), and line voltage V_ab leads phase voltage V_an by exactly +30 degrees.",
    },
    {
        "id": "dalton_gas_law",
        "title": "Causal Scientific Gas Law (Dalton's Partial Pressure Ratio)",
        "track": "science",
        "prompt": "A rigid sealed container contains equal masses of Helium gas (He, molar mass 4 g/mol) and Argon gas (Ar, molar mass 40 g/mol) at thermal equilibrium. Which gas exerts greater partial pressure on the container walls, and by what exact ratio?",
        "expected_verdict": "Helium exerts 10 times greater partial pressure than Argon (P_He / P_Ar = 10) because equal mass yields 10x more moles of He (m/4 vs m/40).",
    },
]


def query_llm_single_call(model_id: str, prompt: str, timeout: int = 150) -> Dict[str, Any]:
    """Execute a single, isolated LLM call with a clean context window."""
    t0 = time.perf_counter()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1024,
    }
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    msg = data["choices"][0]["message"]
    text = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""

    raw_text = text if text.strip() else reasoning
    raw_text = raw_text.strip()

    return {
        "text": raw_text,
        "latency_sec": round(elapsed, 2),
        "total_tokens": data.get("usage", {}).get("total_tokens", 0),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def main():
    print("========================================================================")
    print(" 4-TIER MODEL PARAMETER SCALING DELIBERATION BENCHMARK")
    print(" Ladder: Gemma-e4b (4B) -> Gemma-12B (12B) -> Qwen-27B (27B) -> Qwen-35B (35B)")
    print(" Execution: 4 Zero-Knowledge Calls -> System 2 Deliberation on CPU")
    print("========================================================================")

    os.makedirs(os.path.join(ROOT_DIR, "experiments"), exist_ok=True)

    print("\nInitializing System 2 Deliberation Runtime on CPU (Zero GPU Contention)...")
    runtime = System2Runtime(device="cpu")
    print("System 2 Runtime Ready on CPU!\n")

    results = []

    for prob in BENCHMARK_PROBLEMS:
        prob_id = prob["id"]
        prob_title = prob["title"]
        track = prob["track"]
        prompt = prob["prompt"]
        expected = prob["expected_verdict"]

        print(f"\n========================================================================")
        print(f" [BATTERY: {prob_title}]")
        print(f" Track: {track.upper()}")
        print(f" Expected: {expected}")
        print(f"========================================================================")

        problem_result = {
            "problem": prob,
            "models": {},
        }

        for model_cfg in MODELS_TO_TEST:
            m_id = model_cfg["id"]
            m_name = model_cfg["name"]
            m_tier = model_cfg["tier"]
            print(f"\n>>> Sampling Model: [{m_tier}] {m_name}")

            candidates = []
            call_metas = []

            for i in range(4):
                print(f"    - Sample {i+1}/4 (isolated context)...", end="", flush=True)
                call_res = query_llm_single_call(m_id, prompt)
                candidates.append(call_res["text"])
                call_metas.append(call_res)
                print(f" done ({call_res['latency_sec']}s, {call_metas[-1]['completion_tokens']} tokens)")

            # Run System 2 Deliberation on CPU across the 4 independent candidates
            print(f"    -> Executing System 2 Deliberation (CPU, Track: {track})...")
            t_s2 = time.perf_counter()
            delib = runtime.deliberate(question=prompt, choices=candidates, track=track)
            s2_elapsed_ms = round((time.perf_counter() - t_s2) * 1000, 2)

            win_idx = delib["winning_index"]
            conf = delib["calibrated_confidence_pct"]
            consensus = delib["consensus_pct"]
            agree_count = delib["agreement_count"]
            win_text = candidates[win_idx]
            killed = delib["killed_branches"]

            print(f"    [System 2 Verdict in {s2_elapsed_ms} ms]:")
            print(f"      Winning Candidate: Index {win_idx} (Confidence: {conf}%, Consensus: {agree_count}/5 Heads)")
            print(f"      Winner Summary: {win_text[:160]}...")
            print(f"      Killed Branches: {len(killed)} rejected")

            model_eval = {
                "model_id": m_id,
                "model_name": m_name,
                "tier": m_tier,
                "samples": call_metas,
                "system2_deliberation": {
                    "latency_ms": s2_elapsed_ms,
                    "winning_index": win_idx,
                    "winning_text": win_text,
                    "calibrated_confidence_pct": conf,
                    "consensus_pct": consensus,
                    "agreement_count": agree_count,
                    "choice_probabilities": delib["choice_probabilities"],
                    "killed_branches": killed,
                },
            }

            problem_result["models"][m_id] = model_eval

        results.append(problem_result)

    # Save to JSON
    output_path = os.path.join(ROOT_DIR, "experiments", "four_model_scaling_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n========================================================================")
    print(f" 4-Tier Scaling Benchmark Complete! Results saved to {output_path}")
    print(f"========================================================================")


if __name__ == "__main__":
    main()
