#!/usr/bin/env python3
"""prepare_eval.py - Prepare Rubric, Faithfulness & Hallucination Meta-Judge Dataset for System2 (Track L / Eval).

Models openai/evals model-graded evaluation tasks (closedqa, fact, cot_classify, rubric compliance)
into the canonical System2 non-autoregressive 4-choice deliberation schema:
- question: str
- choices: list[str] (4 candidate evaluation verdicts or candidate responses)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- category: str ("closedqa_select", "rubric_verdict", "cot_audit", "fact_entailment")
- source: str ("sys2_eval_meta_judge")

Covers two complementary evaluation paradigms:
1. Candidate Selection (Best-of-4): Given a Reference Context, Query, and Rubric, select the only
   candidate response that is 100% grounded and rubric-compliant (rejecting Extrinsic Hallucinations,
   Intrinsic Contradictions, and Rubric Constraint Violations).
2. Rubric Verdict Classification: Given a Reference Context, Rubric, and a single Candidate Output,
   classify the exact audit outcome (PASS: Faithful & Compliant, FAIL: Extrinsic Hallucination,
   FAIL: Intrinsic Contradiction, FAIL: Rubric Constraint Breach) based on the actual defect in the output.

Saves DatasetDict to ./data/eval.
"""

import argparse
import os
import random
from typing import Dict, List, Tuple
from datasets import Dataset, DatasetDict

OUTPUT_DIR = "./data/eval"

VERDICT_PASS = (
    "PASS (Faithful & Rubric-Compliant): Every factual claim and numeric figure is strictly entailed "
    "by the reference context, and the output obeys all positive and negative rubric constraints."
)
VERDICT_HALLUCINATION = (
    "FAIL (Extrinsic Hallucination): The candidate response introduces unsupported external entities, "
    "fabricated citations, or unstated numeric statistics absent from the reference context."
)
VERDICT_CONTRADICTION = (
    "FAIL (Intrinsic Contradiction): The candidate response directly inverts a causal relationship, "
    "boolean condition, or numeric threshold explicitly established in the reference context."
)
VERDICT_RUBRIC_BREACH = (
    "FAIL (Rubric Constraint Violation): Although the underlying facts match the reference context, "
    "the output violates an explicit formatting, scope, or negative-constraint rule in the rubric."
)


def _pad_parity(choices: List[str]) -> List[str]:
    """Ensure character length parity across all 4 options so length cannot be used as a shortcut."""
    max_len = max(len(c) for c in choices)
    padded = []
    for c in choices:
        if len(c) < max_len * 0.84:
            c = f"{c} [verified against reference context and rubric criteria]"
        padded.append(c)
    return padded


def _make_record(question: str, correct: str, distractors: List[str], category: str, rng: random.Random) -> Dict:
    raw_choices = _pad_parity([correct] + distractors[:3])
    indexed = list(enumerate(raw_choices))
    rng.shuffle(indexed)
    shuffled = [item[1] for item in indexed]
    label = [item[0] for item in indexed].index(0)

    formatted_text = (
        f"Question: {question}\n"
        f"(A) {shuffled[0]}\n"
        f"(B) {shuffled[1]}\n"
        f"(C) {shuffled[2]}\n"
        f"(D) {shuffled[3]}"
    )
    return {
        "question": question,
        "choices": shuffled,
        "text": formatted_text,
        "label": label,
        "category": category,
        "source": "sys2_eval_meta_judge",
    }


def generate_closedqa_selection_samples(count: int, seed: int) -> List[Dict]:
    """Generate Best-of-4 ClosedQA & Rubric selection samples."""
    rng = random.Random(seed)
    samples = []

    systems = [
        ("ChronosDB", "distributed time-series engine", "raft-v3", "quorum reads", "eventual consistency"),
        ("AegisKV", "in-memory key-value store", "paxos-lite", "linearizable reads", "stale snapshot reads"),
        ("VortexMQ", "zero-copy message broker", "epaxos", "strict FIFO ordering", "out-of-order batching"),
        ("NebulaFS", "content-addressed object store", "chain-replication", "immutable write sealing", "in-place mutation"),
        ("PulseGraph", "real-time graph analytics engine", "two-phase-commit", "serializable isolation", "dirty read isolation"),
        ("HyperShard", "sharded relational proxy", "zab-consensus", "synchronous failover", "asynchronous split-brain"),
        ("QuasarCache", "write-through edge cache", "gossip-crdt", "monotonic read guarantees", "non-monotonic eviction"),
        ("StratosSearch", "inverted vector index", "hnsw-quantized", "exact L2 re-ranking", "uncalibrated cosine pruning"),
    ]

    forbidden_words = [
        ("guaranteed", "never use the word 'guaranteed'"),
        ("seamlessly", "do not include the adverb 'seamlessly'"),
        ("unlimited", "must not contain the word 'unlimited'"),
        ("instantaneously", "prohibited from using the word 'instantaneously'"),
        ("flawless", "must omit the marketing term 'flawless'"),
        ("zero-latency", "strictly forbid the phrase 'zero-latency'"),
    ]

    ext_entities = [
        "commissioned by the Geneva Cloud Consortium in 2021",
        "certified under ISO/IEC 42001-Tier4 by external auditors",
        "co-designed with the Zurich Distributed Systems Lab",
        "backed by a 128-node hardware HSM enclave in Frankfurt",
        "benchmarked at 4.8 million IOPS on AWS Graviton4 instances",
        "funded through a $42M Series B infrastructure grant",
    ]

    for i in range(count):
        sys_name, sys_desc, proto, true_prop, false_prop = rng.choice(systems)
        latency_p99 = rng.randint(4, 48)
        wrong_latency = latency_p99 * rng.choice([2, 3, 5])
        max_nodes = rng.choice([16, 32, 64, 128, 256])
        wrong_nodes = max_nodes + rng.choice([64, 128, 512])
        retention_days = rng.choice([7, 14, 30, 90, 180])
        bad_word, rubric_rule = rng.choice(forbidden_words)
        ext_claim = rng.choice(ext_entities)

        question = (
            f"[EVAL_RUBRIC: ClosedQA Faithfulness & Constraint Audit]\n"
            f"Reference Context: {sys_name} is a {sys_desc} that uses the {proto} protocol to enforce "
            f"{true_prop} across clusters of up to {max_nodes} nodes. Under nominal load, {sys_name} "
            f"maintains a p99 latency of {latency_p99} ms and retains audit logs for {retention_days} days.\n"
            f"Rubric Constraints:\n"
            f"1. Faithfulness: State ONLY facts present in the Reference Context (no extrinsic claims).\n"
            f"2. Accuracy: Preserve exact numeric thresholds and protocol properties.\n"
            f"3. Negative Constraint: The response must {rubric_rule}.\n"
            f"Task: Select the candidate response that passes all three evaluation rubric criteria."
        )

        correct = (
            f"{sys_name} operates as a {sys_desc} utilizing {proto} to provide {true_prop} across up to "
            f"{max_nodes} nodes, achieving a {latency_p99} ms p99 latency with a {retention_days}-day log retention window."
        )
        # Distractor 1: Extrinsic hallucination
        d_halluc = (
            f"{sys_name} operates as a {sys_desc} utilizing {proto} across up to {max_nodes} nodes with "
            f"{latency_p99} ms p99 latency, and was {ext_claim}."
        )
        # Distractor 2: Intrinsic contradiction (flips true_prop -> false_prop and/or numbers)
        d_contra = (
            f"{sys_name} operates as a {sys_desc} utilizing {proto} to rely on {false_prop} across up to "
            f"{wrong_nodes} nodes, exhibiting a {wrong_latency} ms p99 latency window."
        )
        # Distractor 3: Rubric negative constraint violation (uses bad_word)
        d_rubric = (
            f"{sys_name} is a {bad_word} {sys_desc} utilizing {proto} to provide {true_prop} across up to "
            f"{max_nodes} nodes with {latency_p99} ms p99 latency and {retention_days} days of log retention."
        )

        samples.append(_make_record(question, correct, [d_halluc, d_contra, d_rubric], "closedqa_select", rng))

    return samples


def generate_rubric_verdict_samples(count: int, seed: int) -> List[Dict]:
    """Generate single-candidate model-graded rubric verdict classification samples."""
    rng = random.Random(seed)
    samples = []

    domains = [
        ("API Rate Limiter Policy", "token-bucket algorithm", "requests exceeding burst capacity receive HTTP 429", "requests exceeding burst capacity are silently queued with HTTP 200"),
        ("TLS Handshake Specification", "ECDHE-RSA-AES256-GCM-SHA384 cipher suite", "renegotiation with legacy TLS 1.0 clients is strictly rejected", "legacy TLS 1.0 clients are automatically downgraded and accepted"),
        ("Database Backup SLA", "incremental WAL archiving", "point-in-time recovery is bounded by a 15-minute RPO", "point-in-time recovery has an unbounded 24-hour RPO window"),
        ("Container Sandbox Profile", "seccomp-bpf syscall filtering", "unprivileged containers cannot invoke mount or ptrace syscalls", "unprivileged containers are permitted full ptrace kernel access"),
        ("Payment Settlement Ledger", "double-entry idempotency keys", "duplicate webhook payloads with identical keys are no-op deduplicated", "duplicate webhook payloads trigger a secondary settlement charge"),
        ("Telemetry Ingestion Pipeline", "Apache Arrow batch serialization", "malformed schema records are routed to the dead-letter queue", "malformed schema records halt the entire ingestion partition"),
    ]

    hallucinated_addenda = [
        "Furthermore, the subsystem was audited by NIST under Special Publication 800-999 in Q4.",
        "Additionally, the service replicates state to three oceanic submarine data centers in Reykjavik.",
        "In production benchmarks, this reduces AWS NatGateway egress billing by 73.4% annually.",
        "The module was originally authored in Erlang by the Ericsson Telecommunications Group in 1998.",
    ]

    modes = ["pass", "hallucination", "contradiction", "rubric_breach"]

    for i in range(count):
        title, mechanism, true_behavior, contra_behavior = rng.choice(domains)
        timeout_sec = rng.choice([5, 10, 15, 30, 45, 60])
        retry_limit = rng.choice([2, 3, 4, 5])
        max_words_rule = "Must not include markdown code blocks or SQL snippets"
        mode = modes[i % 4]

        if mode == "pass":
            candidate = (
                f"The {title} enforces the {mechanism} with a {timeout_sec}-second timeout and at most "
                f"{retry_limit} retries, ensuring that {true_behavior}."
            )
            correct_verdict = VERDICT_PASS
            distractors = [VERDICT_HALLUCINATION, VERDICT_CONTRADICTION, VERDICT_RUBRIC_BREACH]
        elif mode == "hallucination":
            addendum = rng.choice(hallucinated_addenda)
            candidate = (
                f"The {title} enforces the {mechanism} with a {timeout_sec}-second timeout and {retry_limit} "
                f"retries so that {true_behavior}. {addendum}"
            )
            correct_verdict = VERDICT_HALLUCINATION
            distractors = [VERDICT_PASS, VERDICT_CONTRADICTION, VERDICT_RUBRIC_BREACH]
        elif mode == "contradiction":
            candidate = (
                f"The {title} bypasses the {mechanism} using a {timeout_sec * 10}-second timeout and {retry_limit + 8} "
                f"retries, meaning that {contra_behavior}."
            )
            correct_verdict = VERDICT_CONTRADICTION
            distractors = [VERDICT_PASS, VERDICT_HALLUCINATION, VERDICT_RUBRIC_BREACH]
        else:  # rubric_breach
            candidate = (
                f"The {title} enforces the {mechanism} ({timeout_sec}s timeout, {retry_limit} retries) so "
                f"{true_behavior}. ```sql SELECT * FROM policy_table WHERE retries = {retry_limit}; ```"
            )
            correct_verdict = VERDICT_RUBRIC_BREACH
            distractors = [VERDICT_PASS, VERDICT_HALLUCINATION, VERDICT_CONTRADICTION]

        question = (
            f"[EVAL_RUBRIC: Model-Graded Output Verification]\n"
            f"Reference Context: The {title} uses the {mechanism} with a strict timeout of {timeout_sec} seconds "
            f"and a maximum of {retry_limit} retry attempts. Under this specification, {true_behavior}.\n"
            f"Rubric Formatting Rule: {max_words_rule}.\n"
            f"Candidate Output to Grade: \"{candidate}\"\n"
            f"Evaluate the Candidate Output against the Reference Context and Rubric Formatting Rule:"
        )

        samples.append(_make_record(question, correct_verdict, distractors, "rubric_verdict", rng))

    return samples


def generate_cot_audit_samples(count: int, seed: int) -> List[Dict]:
    """Generate Chain-of-Thought step verification samples (modeled after openai/evals cot_classify)."""
    rng = random.Random(seed)
    samples = []

    cot_pass = (
        "COT_VALID (Sound Step-by-Step Derivation): Every intermediate step uses exact premises from the "
        "prompt, applies valid algebraic transitions, and arrives at the consistent final quantity."
    )
    cot_premise_err = (
        "COT_INVALID (Premise Transcription Error): Step 1 misquotes an initial numeric rate or baseline "
        "quantity from the problem statement, propagating a corrupted value through subsequent steps."
    )
    cot_op_err = (
        "COT_INVALID (Intermediate Operator Inversion): Step 1 transcribes the premises accurately, but "
        "Step 2 applies an inverted arithmetic operation (subtracting instead of adding or multiplying)."
    )
    cot_flip_err = (
        "COT_INVALID (Final Conclusion Mismatch): Both Step 1 and Step 2 compute the correct intermediate "
        "values, but Step 3 states a final conclusion that contradicts Step 2's derived number."
    )

    modes = ["valid", "premise_err", "op_err", "flip_err"]

    for i in range(count):
        base_units = rng.randint(20, 90)
        rate_per_hour = rng.randint(6, 25)
        hours = rng.randint(3, 8)
        defect_deduction = rng.randint(4, 15)

        step1_true = rate_per_hour * hours
        step2_true = base_units + step1_true - defect_deduction

        mode = modes[i % 4]
        if mode == "valid":
            trace = (
                f"Step 1: Compute new output over {hours} hours = {rate_per_hour} * {hours} = {step1_true}. "
                f"Step 2: Add initial inventory ({base_units}) and subtract {defect_deduction} defects = "
                f"{base_units} + {step1_true} - {defect_deduction} = {step2_true}. "
                f"Step 3: Final verified total is {step2_true} units."
            )
            correct = cot_pass
            distractors = [cot_premise_err, cot_op_err, cot_flip_err]
        elif mode == "premise_err":
            wrong_rate = rate_per_hour + 5
            w_s1 = wrong_rate * hours
            w_s2 = base_units + w_s1 - defect_deduction
            trace = (
                f"Step 1: Using a rate of {wrong_rate} per hour over {hours} hours = {wrong_rate} * {hours} = {w_s1}. "
                f"Step 2: Combine with initial {base_units} minus {defect_deduction} defects = {w_s2}. "
                f"Step 3: Final verified total is {w_s2} units."
            )
            correct = cot_premise_err
            distractors = [cot_pass, cot_op_err, cot_flip_err]
        elif mode == "op_err":
            w_s2 = base_units - step1_true + defect_deduction
            trace = (
                f"Step 1: Compute new output over {hours} hours = {rate_per_hour} * {hours} = {step1_true}. "
                f"Step 2: Subtract new output from initial inventory ({base_units} - {step1_true} + {defect_deduction}) = {w_s2}. "
                f"Step 3: Final verified total is {w_s2} units."
            )
            correct = cot_op_err
            distractors = [cot_pass, cot_premise_err, cot_flip_err]
        else:  # flip_err
            flipped_final = step2_true + rng.choice([17, 29, 43])
            trace = (
                f"Step 1: Compute new output over {hours} hours = {rate_per_hour} * {hours} = {step1_true}. "
                f"Step 2: Add initial inventory ({base_units}) and subtract {defect_deduction} defects = {step2_true}. "
                f"Step 3: Therefore, the final verified total is {flipped_final} units."
            )
            correct = cot_flip_err
            distractors = [cot_pass, cot_premise_err, cot_op_err]

        question = (
            f"[EVAL_RUBRIC: Chain-of-Thought Process Supervision Audit]\n"
            f"Problem Statement: A warehouse starts with {base_units} units, produces {rate_per_hour} units/hour "
            f"for {hours} hours, and removes {defect_deduction} defective units. Ground-Truth Total = {step2_true}.\n"
            f"Candidate Reasoning Trace: \"{trace}\"\n"
            f"Audit the Candidate Reasoning Trace and identify its exact process evaluation status:"
        )

        samples.append(_make_record(question, correct, distractors, "cot_audit", rng))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Track L (Eval Meta-Judge) Dataset for System2")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Destination directory for DatasetDict")
    parser.add_argument("--num-train", type=int, default=1600, help="Target training set size")
    parser.add_argument("--num-test", type=int, default=400, help="Target held-out test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    total = args.num_train + args.num_test
    n_closedqa = int(total * 0.40)
    n_verdict = int(total * 0.35)
    n_cot = total - (n_closedqa + n_verdict)

    print("=" * 68)
    print(" GENERATING TRACK L (EVAL / RUBRIC META-JUDGE) DATASET FOR SYSTEM 2")
    print(f" Train: {args.num_train} | Test: {args.num_test} | Total: {total}")
    print("=" * 68)

    pool = (
        generate_closedqa_selection_samples(n_closedqa, seed=args.seed + 1)
        + generate_rubric_verdict_samples(n_verdict, seed=args.seed + 2)
        + generate_cot_audit_samples(n_cot, seed=args.seed + 3)
    )

    rng = random.Random(args.seed)
    rng.shuffle(pool)

    test_samples = pool[:args.num_test]
    train_samples = pool[args.num_test:args.num_test + args.num_train]

    ds_dict = DatasetDict({
        "train": Dataset.from_list(train_samples),
        "test": Dataset.from_list(test_samples),
    })

    os.makedirs(args.output_dir, exist_ok=True)
    ds_dict.save_to_disk(args.output_dir)
    print(f"Saved Track L dataset to {args.output_dir} (train={len(train_samples)}, test={len(test_samples)})")


if __name__ == "__main__":
    main()
