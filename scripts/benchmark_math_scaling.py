#!/usr/bin/env python3
"""benchmark_math_scaling.py - 4-Tier Scaling Deliberation Benchmark on Phase 3 GSM8K Multi-Step Math.

Evaluates:
- Gemma-4 e4b (4B)
- Gemma-4 12B (12B)
- Qwen-3.8 27B (27B)
- Qwen-3.6 35B (35B)

Protocol:
- 4 strictly independent zero-knowledge calls per model (temperature=0.7).
- Adjudicated by System 2 Mathematical Reasoner (models/system2_math.pt) on CPU.
- Verification of whether System 2 allows 4B/12B models to match or exceed 35B models on multi-step arithmetic.
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

MATH_PROBLEM = {
    "id": "gsm8k_market_eggs",
    "title": "Multi-Step Mathematical Reasoning (GSM8K Daily Profit)",
    "track": "math",
    "prompt": (
        "Solve this step-by-step and provide the final numerical answer:\n"
        "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning "
        "and bakes muffins for her friends every day with four. She sells the remainder "
        "at the farmers' market daily for $2 per egg. How much in dollars does Janet "
        "make every day at the farmers' market?"
    ),
    "expected_verdict": "Janet sells 16 - 3 - 4 = 9 eggs per day. At $2 each, 9 * 2 = 18. Final answer: $18.",
}


def query_llm_single_call(model_id: str, prompt: str, timeout: int = 150, max_tokens: int = 4096) -> Dict[str, Any]:
    """Execute a single, isolated LLM call with a clean context window.

    Dynamically separates reasoning_content from final content to prevent
    truncation and withhold scratchpad thoughts from System 2 deliberation.
    """
    import re
    t0 = time.perf_counter()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0

    if resp.status_code != 200:
        raise RuntimeError(f"LM Studio returned status {resp.status_code}: {resp.text}")

    data = resp.json()
    msg = data["choices"][0]["message"]
    raw_content = msg.get("content", "") or ""
    reasoning_content = msg.get("reasoning_content", "") or ""

    # Check for embedded <think>...</think> tags in content
    had_thinking_tags = False
    if "<think>" in raw_content:
        had_thinking_tags = True
        parts = re.split(r"</think>", raw_content, maxsplit=1)
        if len(parts) == 2:
            extracted_think = re.sub(r"^<think>", "", parts[0]).strip()
            clean_answer = parts[1].strip()
            if not reasoning_content:
                reasoning_content = extracted_think
            final_text = clean_answer
        else:
            final_text = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    else:
        final_text = raw_content.strip()

    # If content was empty and only reasoning was returned, fallback
    if not final_text and reasoning_content:
        final_text = reasoning_content.strip()

    used_thinking = bool(reasoning_content) or had_thinking_tags

    return {
        "text": final_text,
        "reasoning": reasoning_content,
        "used_thinking": used_thinking,
        "latency_sec": round(elapsed, 2),
        "total_tokens": data.get("usage", {}).get("total_tokens", 0),
        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
    }


def main():
    print("========================================================================")
    print(" 4-TIER MODEL PARAMETER SCALING DELIBERATION: PHASE 3 MATH (GSM8K)")
    print(" Ladder: Gemma-e4b (4B) -> Gemma-12B (12B) -> Qwen-27B (27B) -> Qwen-35B (35B)")
    print(" Execution: 4 Zero-Knowledge Calls Per Model -> System 2 Math Deliberation")
    print("========================================================================")

    os.makedirs(os.path.join(ROOT_DIR, "experiments"), exist_ok=True)

    print("\nInitializing System 2 Deliberation Runtime on CPU (Zero GPU Contention)...")
    runtime = System2Runtime(device="cpu")
    print("System 2 Runtime Ready on CPU!\n")

    prob = MATH_PROBLEM
    print(f"\n========================================================================")
    print(f" [PROBLEM: {prob['title']}]")
    print(f" Expected: {prob['expected_verdict']}")
    print(f"========================================================================")

    model_evaluations = {}

    for model_cfg in MODELS_TO_TEST:
        m_id = model_cfg["id"]
        m_name = model_cfg["name"]
        m_tier = model_cfg["tier"]
        print(f"\n>>> Sampling Model: [{m_tier}] {m_name}")

        candidates = []
        call_metas = []

        for i in range(4):
            print(f"    - Sample {i+1}/4 (isolated context)...", end="", flush=True)
            call_res = query_llm_single_call(m_id, prob["prompt"])
            candidates.append(call_res["text"])
            call_metas.append(call_res)
            print(f" done ({call_res['latency_sec']}s, {call_res['completion_tokens']} tokens)")

        # Run System 2 Deliberation on CPU across the 4 independent candidates
        print(f"    -> Executing System 2 Deliberation (Track: auto/math)...")
        t_s2 = time.perf_counter()
        delib = runtime.deliberate(question=prob["prompt"], choices=candidates, track="auto")
        s2_elapsed_ms = round((time.perf_counter() - t_s2) * 1000, 2)

        win_idx = delib["winning_index"]
        conf = delib["calibrated_confidence_pct"]
        consensus = delib["consensus_pct"]
        agree_count = delib["agreement_count"]
        win_text = candidates[win_idx]
        killed = delib["killed_branches"]
        routing = delib.get("routing", {})

        print(f"    [System 2 Verdict in {s2_elapsed_ms} ms]:")
        print(f"      Routed Track: {routing.get('track', 'math')} (Conf: {routing.get('confidence', 1.0):.2f}, Method: {routing.get('method', 'direct')})")
        print(f"      Winning Candidate: Index {win_idx} (Calibrated Confidence: {conf}%, Consensus: {agree_count}/5 Heads)")
        print(f"      Winner Excerpt: {win_text[:140].replace(chr(10), ' ')}...")
        print(f"      Killed Branches: {len(killed)} rejected")

        model_eval = {
            "model_id": m_id,
            "model_name": m_name,
            "tier": m_tier,
            "samples": call_metas,
            "system2_deliberation": {
                "latency_ms": s2_elapsed_ms,
                "routing": routing,
                "winning_index": win_idx,
                "winning_text": win_text,
                "calibrated_confidence_pct": conf,
                "consensus_pct": consensus,
                "agreement_count": agree_count,
                "choice_probabilities": delib["choice_probabilities"],
                "killed_branches": killed,
            },
        }
        model_evaluations[m_id] = model_eval

    # Save to JSON
    output_path = os.path.join(ROOT_DIR, "experiments", "gsm8k_four_model_scaling_results.json")
    with open(output_path, "w") as f:
        json.dump(model_evaluations, f, indent=2)

    print(f"\n========================================================================")
    print(f" GSM8K 4-Tier Scaling Benchmark Complete! Results saved to {output_path}")
    print(f"========================================================================")


if __name__ == "__main__":
    main()
