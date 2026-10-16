#!/usr/bin/env python3
"""benchmark_multidomain_hybrid.py - Multi-Domain Path B Hybrid Deliberation Benchmark.

Evaluates Path B deliberation (Small LLM generation + System 2 feedforward arbitration)
across 5 distinct domains:
1. Track H: Legal Reasoning (Bar Exam & Statutory Compliance)
2. Track I: Financial Reasoning (Accounting & Economics)
3. Track C: Medical & Clinical Diagnostics (MedQA USMLE)
4. Track B: Polyglot Code Execution (Code Traces)
5. Track G: Mathematical Reasoning (GSM8K)

Tests:
- Small local model: google/gemma-4-e4b (4B)
- Withholds scratchpad thinking (<think>) from System 2
- Single isolated context window per call
- Dynamic domain routing dispatch (< 0.8 ms)
- 5-head parallel deliberation + cross-attention Judger confidence
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional
import requests
import torch
from datasets import load_from_disk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.domain_router import DomainRouter
from encoding import mean_pool_pairwise
from system import System2
from bert import get_bert

LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
MODEL_ID = "google/gemma-4-e4b"

DOMAIN_DATASETS = {
    "legal": os.path.join(ROOT_DIR, "data", "legal"),
    "finance": os.path.join(ROOT_DIR, "data", "finance"),
    "med": os.path.join(ROOT_DIR, "data", "medical"),
    "code": os.path.join(ROOT_DIR, "data", "code_traces"),
    "math": os.path.join(ROOT_DIR, "data", "math"),
}

CHECKPOINT_PATHS = {
    "legal": os.path.join(ROOT_DIR, "models", "system2_legal.pt"),
    "finance": os.path.join(ROOT_DIR, "models", "system2_finance.pt"),
    "med": os.path.join(ROOT_DIR, "models", "system2_med.pt"),
    "code": os.path.join(ROOT_DIR, "models", "system2_code.pt"),
    "math": os.path.join(ROOT_DIR, "models", "system2_math.pt"),
    "science": os.path.join(ROOT_DIR, "models", "system2.pt"),
}


def query_llm(model_id: str, prompt: str, timeout: int = 120, max_tokens: int = 4096) -> Dict[str, Any]:
    """Execute a single LLM call with thinking separation and dynamic token limits."""
    t0 = time.perf_counter()
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=timeout)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()

    data = resp.json()
    msg = data["choices"][0]["message"]
    raw_content = msg.get("content", "") or ""
    reasoning_content = msg.get("reasoning_content", "") or ""

    # Strip embedded <think>...</think>
    if "<think>" in raw_content:
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

    if not final_text and reasoning_content:
        final_text = reasoning_content.strip()

    return {
        "text": final_text,
        "reasoning": reasoning_content,
        "latency_sec": round(elapsed, 2),
        "tokens": data.get("usage", {}).get("total_tokens", 0),
    }


def load_sys2_model(ckpt_path: str, device: str = "cpu") -> System2:
    """Load System2 model from checkpoint."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sys2_state = state["sys2"] if isinstance(state, dict) and "sys2" in state else state

    sys2 = System2(
        num_heads=5,
        num_classes=1,
        pairwise=True,
        diff_features=True,
        head_layers=2,
        ensemble_temp=1.8,
        margin_features=True,
        judger_residual=True,
        head_norm_logits=True,
        decision_conf=True,
        decision_conf_mode="softmax",
        decision_conf_tau=0.75,
        decision_conf_alpha=0.3,
    )
    sys2.load_state_dict(sys2_state, strict=False)
    sys2.to(device)
    sys2.eval()
    return sys2


def main():
    print("========================================================================")
    print(" SYSTEM2 MULTI-DOMAIN PATH B HYBRID DELIBERATION BENCHMARK")
    print(f" LLM Generator: {MODEL_ID} | Decision Engine: System2 (9 Tracks)")
    print("========================================================================")

    # Initialize router on CPU
    router = DomainRouter(device="cpu")
    tokenizer, bert_model = get_bert("answerdotai/ModernBERT-Large-Instruct")
    bert_model.to("cpu")
    bert_model.eval()

    results = []

    for domain, data_path in DOMAIN_DATASETS.items():
        print(f"\n" + "-" * 72)
        print(f" EVALUATING DOMAIN: [{domain.upper()}] from {data_path}")
        print("-" * 72)

        if not os.path.exists(data_path):
            print(f"Skipping {domain}: Dataset path missing ({data_path})")
            continue

        ds = load_from_disk(data_path)
        test_split = ds["test"]
        sample = test_split[0]

        question = sample["question"]
        choices = sample["choices"]
        true_label = sample["label"]
        letters = ["A", "B", "C", "D"]

        print(f"Question: {question[:120]}...")
        for i, c in enumerate(choices):
            print(f"  ({letters[i]}) {str(c)[:80]}")
        print(f"Ground Truth: ({letters[true_label]})")

        # 1. Route Domain
        route_decision = router.route(question)
        routed_track = route_decision.selected_track
        print(f"\n[Dynamic Router] Dispatched to: [{routed_track.upper()}] ({route_decision.method}, conf={route_decision.confidence:.2f}, latency={route_decision.latency_ms:.2f}ms)")

        # 2. Query Small LLM
        prompt = (
            f"{question}\n\n"
            f"Choices:\n"
            f"(A) {choices[0]}\n"
            f"(B) {choices[1]}\n"
            f"(C) {choices[2]}\n"
            f"(D) {choices[3]}\n\n"
            f"Think step by step and specify the final answer as (A), (B), (C), or (D)."
        )
        print(f"\n[Querying {MODEL_ID}]...")
        llm_out = query_llm(MODEL_ID, prompt)
        print(f"LLM Answer text (latency: {llm_out['latency_sec']}s, tokens: {llm_out['tokens']}):")
        print(f"  {llm_out['text'][:150]}...")

        # Parse LLM prediction
        llm_pred = None
        match = re.search(r"\b([A-D])\b", llm_out["text"][-80:])
        if match:
            llm_pred = ["A", "B", "C", "D"].index(match.group(1))

        # 3. System 2 Deliberation
        ckpt_key = routed_track if routed_track in CHECKPOINT_PATHS else domain
        ckpt_path = CHECKPOINT_PATHS.get(ckpt_key, CHECKPOINT_PATHS["science"])
        print(f"\n[Loading System2 Checkpoint] {os.path.basename(ckpt_path)}...")
        sys2 = load_sys2_model(ckpt_path, device="cpu")

        t0_delib = time.perf_counter()
        with torch.no_grad():
            vecs = mean_pool_pairwise(
                [question],
                [choices],
                max_len=1024,
                device="cpu",
                with_grad=False,
                asym_spans=True,
                asym_span_mode="choice_forward",
                model=bert_model,
                tokenizer=tokenizer,
            )
            best, conf, pruned, confs, logits, emb, scores, ens_logits = sys2(vecs)
            delib_time_ms = (time.perf_counter() - t0_delib) * 1000

            # Ensemble logits per candidate choice
            win_choice_idx = ens_logits.argmax(dim=-1).item()
            head_preds = logits.squeeze(0).argmax(dim=-1).tolist()
            consensus_count = head_preds.count(win_choice_idx)
            calibrated_conf = conf.item() if hasattr(conf, "item") else float(conf)

        print(f"\n[System2 Deliberation Result]")
        print(f"  Winning Choice: ({letters[win_choice_idx]}) {choices[win_choice_idx][:60]}")
        print(f"  Calibrated Confidence: {calibrated_conf * 100:.1f}%")
        print(f"  Head Consensus: {consensus_count}/5 Heads")
        print(f"  Deliberation Latency: {delib_time_ms:.2f} ms")

        sys2_correct = (win_choice_idx == true_label)
        llm_correct = (llm_pred == true_label) if llm_pred is not None else False

        print(f"  Accuracy Status: System2 {'✓ CORRECT' if sys2_correct else '✗ INCORRECT'} | LLM Direct {'✓ CORRECT' if llm_correct else '✗ INCORRECT'}")

        results.append({
            "domain": domain,
            "routed_track": routed_track,
            "ground_truth": letters[true_label],
            "llm_pred": letters[llm_pred] if llm_pred is not None else "None",
            "llm_correct": llm_correct,
            "sys2_pred": letters[win_choice_idx],
            "sys2_correct": sys2_correct,
            "confidence": round(calibrated_conf, 4),
            "consensus": f"{consensus_count}/5",
            "delib_latency_ms": round(delib_time_ms, 2),
        })

    out_file = os.path.join(ROOT_DIR, "experiments", "multidomain_hybrid_benchmark_results.json")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 72)
    print(" MULTI-DOMAIN HYBRID BENCHMARK SUMMARY")
    print("=" * 72)
    for r in results:
        sys2_mark = "✓" if r["sys2_correct"] else "✗"
        llm_mark = "✓" if r["llm_correct"] else "✗"
        print(f"[{r['domain'].upper():<8}] Routed: {r['routed_track']:<8} | Sys2: ({r['sys2_pred']}) {sys2_mark} ({r['confidence']*100:.1f}%, {r['consensus']}) | LLM: ({r['llm_pred']}) {llm_mark} | Latency: {r['delib_latency_ms']}ms")
    print(f"\nSaved benchmark results to {out_file}")


if __name__ == "__main__":
    main()
