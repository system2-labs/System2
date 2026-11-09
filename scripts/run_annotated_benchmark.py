#!/usr/bin/env python3
"""scripts/run_annotated_benchmark.py - Evaluates Benchmark Prompts with Token Annotations.

Tests inputs formatted like:
  QUESTION: What causes ocean tides?
  ----------------------------------------------------------------------
     [A] [15 tok] Earth's rotation
     [B] [15 tok] Moon's gravity
     [C] [14 tok] Wind currents
     [D] [16 tok] Sun's magnetism

Demonstrates:
  1. The Token Pollution Effect: Passing raw '[A] [15 tok] ...' into neural representations.
  2. The Semantic Normalizer: Parsing and isolating the question & pure candidate claims.
  3. Perceptual vs. Causal Reasoning: System 1 associative bias vs. System 2 causal verification.
  4. Generative Pairing: Qwen 35B factual hypothesis generation + System 2 formal verification.
"""

import os
import sys
import time
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sys1.engine import System1Engine
from sys2.pipeline import DualProcessPipeline


PROMPT_1 = """
QUESTION: What causes ocean tides?
  ----------------------------------------------------------------------
     [A] [15 tok] Earth's rotation
     [B] [15 tok] Moon's gravity
     [C] [14 tok] Wind currents
     [D] [16 tok] Sun's magnetism
"""

PROMPT_2 = """
QUESTION: What is the speed of light in a vacuum?
  ----------------------------------------------------------------------
     [A] [22 tok] 299,792 km/s
     [B] [22 tok] 150,000 km/s
     [C] [22 tok] 1,080 km/s
     [D] [22 tok] 3,000 km/s
"""


def test_qwen_generation(raw_prompt: str) -> str:
    url = "http://localhost:1234/v1/chat/completions"
    p = f"{raw_prompt}\nSelect the single correct option. Give the letter and a 1-sentence physical explanation."
    try:
        res = requests.post(url, json={
            "model": "qwen3.6-35b-a3b-splash",
            "messages": [{"role": "user", "content": p}],
            "max_tokens": 1000,
            "temperature": 0.1,
        }, timeout=45).json()
        msg = res["choices"][0]["message"]
        content = msg.get("content", "").strip()
        reasoning = msg.get("reasoning_content", "").strip()
        return content if content else reasoning[:200]
    except Exception as e:
        return f"[Qwen Offline / Error: {e}]"


def run_benchmark():
    print("=" * 78)
    print(" EVALUATION: RAW ANNOTATED BENCHMARK PROMPTS VIA DUAL PROCESS SYSTEM")
    print("=" * 78)

    pipeline = DualProcessPipeline(device="mps")
    sys1 = pipeline.sys1

    cases = [
        ("Question 1: Causal Ocean Tides", PROMPT_1, "science", "B"),
        ("Question 2: Physical Constant (Speed of Light)", PROMPT_2, "physics", "A"),
    ]

    for title, raw_prompt, domain, expected_letter in cases:
        print("\n" + "=" * 78)
        print(f" {title} (Target Answer: [{expected_letter}])")
        print("=" * 78)
        print(raw_prompt.strip())
        print("-" * 78)

        # Step 1: Parse prompt
        q, clean_choices, meta = pipeline.parse_annotated_prompt(raw_prompt)
        raw_choices = [line.strip() for line in raw_prompt.strip().split("\n") if "[" in line and "tok" in line]

        print(f"[Parsed Question]: {q}")
        print(f"[Clean Choices]:  {clean_choices}")
        print(f"[Extracted Meta]: {meta}")

        # Step 2: Test 1 - Raw Input with Token Pollution (System 1 Solo)
        print("\n--- [Stage 1: Raw Token-Polluted Input in System 1] ---")
        res_raw = sys1.adjudicate(q, raw_choices, domain=domain)
        raw_win_char = chr(ord("A") + res_raw.selected_index)
        print(f"  System 1 Raw Winner: ({raw_win_char}) {res_raw.selected_choice}")
        print(f"  System 1 Raw Conf:   {res_raw.confidence * 100:.1f}% | Consensus: {res_raw.head_consensus}")

        # Step 3: Test 2 - Clean Input in System 1
        print("\n--- [Stage 2: Clean Semantic Input in System 1] ---")
        res_clean_s1 = sys1.adjudicate(q, clean_choices, domain=domain)
        clean_win_char = chr(ord("A") + res_clean_s1.selected_index)
        print(f"  System 1 Clean Winner: ({clean_win_char}) {res_clean_s1.selected_choice}")
        print(f"  System 1 Clean Conf:   {res_clean_s1.confidence * 100:.1f}% | Consensus: {res_clean_s1.head_consensus}")
        print(f"  Choice Probabilities:  {res_clean_s1.choice_probabilities}")

        # Step 4: Test 3 - Full Dual-Process Pipeline (System 1 + System 2)
        print("\n--- [Stage 3: Full Dual-Process Pipeline (System 1 + System 2 Deliberation)] ---")
        res_dual = pipeline.evaluate(q, clean_choices, domain=domain)
        print(f"  Dual-Process Winner:   ({res_dual.winning_choice_char}) {res_dual.winning_choice_text}")
        print(f"  Final Delib Conf:      {res_dual.confidence * 100:.1f}%")
        print(f"  System 2 Verified:     {res_dual.sys2_verified}")
        print(f"  Mode Taken:            {'⚡ Fast Path' if res_dual.fast_path_taken else '🧠 Slow Path (True Deliberation)'}")
        print(f"  Total Latency:         {res_dual.latency_ms:.2f} ms")

        # Step 5: Test 4 - Paired Generative Model (Qwen 35B)
        print("\n--- [Stage 4: Paired Qwen 35B Generative Verification] ---")
        qwen_reply = test_qwen_generation(raw_prompt)
        print(f"  Qwen 35B Verdict:      {qwen_reply}")
        print("=" * 78)


if __name__ == "__main__":
    run_benchmark()
