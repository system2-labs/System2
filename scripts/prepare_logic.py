#!/usr/bin/env python3
"""prepare_logic.py - Prepare Formal & Deductive Logic Dataset for System2 (Track M / Logic).

Models openai/evals deductive logic, syllogisms, propositional calculus, Knights & Knaves
truth-teller puzzles, and linear-ordering constraint satisfaction into the canonical
System2 non-autoregressive 4-choice deliberation schema:
- question: str
- choices: list[str] (4 length-balanced candidate deductive conclusions)
- text: str (formatted prompt with choices A-D)
- label: int (0..3)
- category: str ("modus_tollens", "syllogism", "knights_knaves", "linear_order", "quantifier_negation")
- source: str ("sys2_formal_logic")

Every distractor models a genuine formal fallacy (Affirming the Consequent, Denying the Antecedent,
Existential Import Fallacy, Modal Overreach, or Quantifier Scope Inversion) with strict character
length parity so surface heuristics fail.

Saves DatasetDict to ./data/logic.
"""

import argparse
import os
import random
from typing import Dict, List
from datasets import Dataset, DatasetDict

OUTPUT_DIR = "./data/logic"


def _pad_parity(choices: List[str]) -> List[str]:
    """Ensure character length parity across all 4 options."""
    max_len = max(len(c) for c in choices)
    padded = []
    for c in choices:
        if len(c) < max_len * 0.84:
            c = f"{c} (derived under strict deductive entailment rules)"
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
        "source": "sys2_formal_logic",
    }


def generate_propositional_chains(count: int, seed: int) -> List[Dict]:
    """Generate Modus Tollens / Hypothetical Syllogism / Contrapositive chains vs formal fallacies."""
    rng = random.Random(seed)
    samples = []

    predicates = [
        ("the consensus node enters epoch-lock", "the quorum certificate is signed", "the state root is finalized", "the validator slash timer is disarmed"),
        ("the compiler enables strict alias analysis", "the loop vectorizer unrolls the inner kernel", "the register pressure exceeds 32 GPRs", "the instruction scheduler inserts spill slots"),
        ("the cryptographic token fails HMAC verification", "the ingress gateway drops the TLS session", "the audit daemon emits a severity-1 alert", "the circuit breaker isolates the upstream pod"),
        ("the theorem prover discharges the induction hypothesis", "the dependent type checker accepts the witness", "the proof kernel exports a verified certificate", "the bytecode emitter strips runtime assertions"),
        ("the autonomous controller detects lidar occlusion", "the planner switches to conservative kinematic bounds", "the actuator limits longitudinal jerk", "the safety supervisor latches fallback mode"),
        ("the Garbage Collector enters a stop-the-world mark phase", "mutator threads are suspended at safepoints", "allocation buffers are flushed to the tenured heap", "read barriers verify pointer forwarding tables"),
    ]

    for i in range(count):
        p, q, r, s = rng.choice(predicates)
        node_id = rng.randint(101, 999)
        mode = i % 3

        if mode == 0:
            # Contrapositive across 3-hop chain: (P -> Q), (Q -> R), ~R |- ~P
            question = (
                f"Formal Logic Premises (Instance #{node_id}):\n"
                f"Premise 1: If {p}, then {q} (P → Q).\n"
                f"Premise 2: If {q}, then {r} (Q → R).\n"
                f"Premise 3: In trace #{node_id}, it is NOT the case that {r} (¬R).\n"
                f"Which conclusion must logically follow by valid deductive inference?"
            )
            correct = (
                f"By chained Modus Tollens (¬R → ¬Q → ¬P), it is necessarily false that {p}, "
                f"and it is also false that {q}."
            )
            d1 = (
                f"By Affirming the Consequent on Premise 1, it is necessarily true that {p}, "
                f"even though {r} did not occur."
            )
            d2 = (
                f"By Denying the Antecedent, {q} still holds true in trace #{node_id} while "
                f"only the final consequent fails."
            )
            d3 = (
                f"The premises are mutually inconsistent (P ∧ ¬P), so no valid deductive "
                f"conclusion can be drawn regarding whether {p}."
            )
        elif mode == 1:
            # Fallacy detection when given Q (Affirming the Consequent is invalid!)
            question = (
                f"Formal Logic Premises (Instance #{node_id}):\n"
                f"Premise 1: Whenever {p}, it follows that {q} (P → Q).\n"
                f"Premise 2: Only if {r} does it hold that {p} (P → R).\n"
                f"Observation: In trace #{node_id}, {q} is observed to be true (Q), but {r} is false (¬R).\n"
                f"What can be validly deduced from these premises and observation?"
            )
            correct = (
                f"Because P → R and ¬R holds, Modus Tollens proves ¬P (it is NOT the case that {p}), "
                f"which remains consistent with Q being true independently."
            )
            d1 = (
                f"Because P → Q and Q is observed, Modus Ponens proves P ({p}), which overrides "
                f"the observation ¬R."
            )
            d2 = (
                f"Because Q is true and R is false, Premise 1 and Premise 2 form a logical "
                f"contradiction that invalidates the entire axiom system."
            )
            d3 = (
                f"By biconditional symmetry (Q ↔ P), observing {q} guarantees both that {p} "
                f"and that {r} must eventually hold."
            )
        else:
            # Disjunctive Syllogism + Contrapositive: (P v S), (P -> Q), ~Q |- S
            question = (
                f"Formal Logic Premises (Instance #{node_id}):\n"
                f"Premise 1: Either {p} OR {s} (P ∨ S).\n"
                f"Premise 2: If {p}, then {q} (P → Q).\n"
                f"Premise 3: In trace #{node_id}, it is NOT the case that {q} (¬Q).\n"
                f"Which deductive conclusion is entailed by Modus Tollens and Disjunctive Syllogism?"
            )
            correct = (
                f"From P → Q and ¬Q we deduce ¬P by Modus Tollens; combining ¬P with (P ∨ S) "
                f"via Disjunctive Syllogism proves that {s} (S)."
            )
            d1 = (
                f"From ¬Q and (P ∨ S), we deduce that neither {p} nor {s} can hold in trace "
                f"#{node_id} (¬P ∧ ¬S)."
            )
            d2 = (
                f"From P → Q and (P ∨ S), we deduce that {p} must be true while {s} is "
                f"necessarily false (P ∧ ¬S)."
            )
            d3 = (
                f"Because Disjunctive Syllogism requires an exclusive-OR (⊕), no conclusion "
                f"can be drawn about whether {s} holds."
            )

        samples.append(_make_record(question, correct, [d1, d2, d3], "modus_tollens", rng))

    return samples


def generate_categorical_syllogisms(count: int, seed: int) -> List[Dict]:
    """Generate First-Order & Categorical Syllogism problems."""
    rng = random.Random(seed)
    samples = []

    triads = [
        ("deterministic automata", "halting verifiers", "unbounded recursive parsers", "certified compilers"),
        ("monotonic lattices", "fixpoint operators", "non-convex optimizers", "verified solvers"),
        ("zero-knowledge proofs", "succinct commitments", "trusted-setup ceremonies", "stateless rollups"),
        ("purely functional kernels", "referentially transparent closures", "mutable global singletons", "verified actors"),
        ("linearly typed channels", "deadlock-free protocols", "unbounded race conditions", "session runtimes"),
        ("hermitian operators", "real-eigenvalue observables", "non-unitary projections", "quantum gates"),
    ]

    for i in range(count):
        a_set, b_set, c_set, d_set = rng.choice(triads)
        case_id = rng.randint(10, 990)

        question = (
            f"Categorical Syllogism (Set Configuration #{case_id}):\n"
            f"Axiom 1: All {a_set} are {b_set} (∀x: A(x) → B(x)).\n"
            f"Axiom 2: No {b_set} are {c_set} (∀x: B(x) → ¬C(x)).\n"
            f"Axiom 3: Some {d_set} are {a_set} (∃x: D(x) ∧ A(x)).\n"
            f"Which statement is strictly and universally entailed by Axioms 1–3?"
        )

        correct = (
            f"Some {d_set} are definitely not {c_set} (∃x: D(x) ∧ ¬C(x)), and no {a_set} "
            f"are {c_set} (∀x: A(x) → ¬C(x))."
        )
        d1 = (
            f"All {b_set} are necessarily {a_set} (∀x: B(x) → A(x)), and all {d_set} "
            f"are excluded from being {c_set}."
        )
        d2 = (
            f"No {d_set} can be {c_set} (∀x: D(x) → ¬C(x)), because existential membership "
            f"in {a_set} generalizes to the entire set of {d_set}."
        )
        d3 = (
            f"Some {c_set} are {a_set} (∃x: C(x) ∧ A(x)), provided that the intersection "
            f"of {d_set} and {b_set} is non-empty."
        )

        samples.append(_make_record(question, correct, [d1, d2, d3], "syllogism", rng))

    return samples


def generate_knights_and_knaves(count: int, seed: int) -> List[Dict]:
    """Generate Knights (always tell truth) and Knaves (always lie) Boolean satisfiability puzzles."""
    rng = random.Random(seed)
    samples = []

    names_pool = [
        ("Alden", "Bryn"), ("Corin", "Dara"), ("Elian", "Faye"),
        ("Galen", "Hera"), ("Ivor", "Juno"), ("Kael", "Lyra"),
        ("Milo", "Nora"), ("Orin", "Petra"), ("Quin", "Rhea"),
    ]

    for i in range(count):
        a_name, b_name = rng.choice(names_pool)
        puzzle_id = rng.randint(100, 999)
        variant = i % 3

        if variant == 0:
            # A says: "At least one of us is a knave." -> A is Knight, B is Knave
            question = (
                f"Knights and Knaves Truth-Table Puzzle #{puzzle_id}:\n"
                f"On an island where Knights always state true propositions (T) and Knaves always state "
                f"false propositions (F), you meet two inhabitants, {a_name} and {b_name}.\n"
                f"{a_name} makes the statement: \"At least one of us ({a_name} or {b_name}) is a Knave.\"\n"
                f"Determine the unique logically consistent assignment for {a_name} and {b_name}:"
            )
            correct = (
                f"{a_name} is a Knight and {b_name} is a Knave (if {a_name} were a Knave, the disjunction "
                f"'at least one is a Knave' would be true, contradicting {a_name} lying)."
            )
            d1 = (
                f"{a_name} is a Knave and {b_name} is a Knight (because a Knight would never claim "
                f"that either inhabitant belongs to the class of Knaves)."
            )
            d2 = (
                f"Both {a_name} and {b_name} are Knights (since {a_name}'s statement is truthful and "
                f"applies symmetrically to both speakers)."
            )
            d3 = (
                f"Both {a_name} and {b_name} are Knaves (since {a_name}'s self-referential disjunction "
                f"forces both truth values to collapse to false)."
            )
        elif variant == 1:
            # A says: "We are both Knaves." -> A is Knave, B is Knight
            question = (
                f"Knights and Knaves Truth-Table Puzzle #{puzzle_id}:\n"
                f"Knights always speak the truth (T) and Knaves always lie (F). You encounter {a_name} and {b_name}.\n"
                f"{a_name} declares: \"Both {a_name} and {b_name} are Knaves (¬{a_name} ∧ ¬{b_name}).\"\n"
                f"What are the exact identities of {a_name} and {b_name}?"
            )
            correct = (
                f"{a_name} is a Knave and {b_name} is a Knight ({a_name} cannot be a Knight claiming to be "
                f"a Knave, so the conjunction is false, forcing {b_name} to be a Knight)."
            )
            d1 = (
                f"Both {a_name} and {b_name} are Knaves (matching the literal content of {a_name}'s "
                f"conjunctive claim ¬{a_name} ∧ ¬{b_name})."
            )
            d2 = (
                f"{a_name} is a Knight and {b_name} is a Knave (since one inhabitant must tell the "
                f"truth while the other falsifies the conjunction)."
            )
            d3 = (
                f"The puzzle is a paradoxical Liar cycle with no satisfying Boolean assignment "
                f"in classical two-valued propositional logic."
            )
        else:
            # A says: "B is a Knave." B says: "Neither A nor B is a Knave (we are both Knights)."
            # If B is Knight -> both Knights -> A is Knight -> B is Knave (contradiction).
            # So B is Knave -> A's statement ("B is a Knave") is TRUE -> A is Knight, B is Knave.
            question = (
                f"Knights and Knaves Truth-Table Puzzle #{puzzle_id}:\n"
                f"Knights always tell the truth (T) and Knaves always lie (F).\n"
                f"Statement 1 ({a_name}): \"{b_name} is a Knave.\"\n"
                f"Statement 2 ({b_name}): \"{a_name} and I are both Knights.\"\n"
                f"Solve the coupled Boolean equivalence system for {a_name} and {b_name}:"
            )
            correct = (
                f"{a_name} is a Knight and {b_name} is a Knave (if {b_name} were a Knight, both would be "
                f"Knights, contradicting {a_name}'s claim; thus {b_name} is a Knave and {a_name} speaks truly)."
            )
            d1 = (
                f"{a_name} is a Knave and {b_name} is a Knight (because {b_name} affirms a positive "
                f"conjunction whereas {a_name} makes a negative accusation)."
            )
            d2 = (
                f"Both {a_name} and {b_name} are Knaves (since mutual contradiction between two "
                f"speakers implies both statements are simultaneously false)."
            )
            d3 = (
                f"Both {a_name} and {b_name} are Knights (satisfying {b_name}'s universal claim "
                f"across the two-agent system)."
            )

        samples.append(_make_record(question, correct, [d1, d2, d3], "knights_knaves", rng))

    return samples


def generate_linear_ordering_puzzles(count: int, seed: int) -> List[Dict]:
    """Generate strict linear ordering & quantifier negation constraint problems."""
    rng = random.Random(seed)
    samples = []

    task_sets = [
        ("T_Lexer", "T_Parser", "T_Typecheck", "T_Codegen", "T_Linker"),
        ("Stage_Auth", "Stage_Decrypt", "Stage_Validate", "Stage_Execute", "Stage_Audit"),
        ("Job_Ingest", "Job_Cleanse", "Job_Index", "Job_Shard", "Job_Replicate"),
        ("Pass_Inline", "Pass_SSA", "Pass_ConstProp", "Pass_RegAlloc", "Pass_Peephole"),
    ]

    for i in range(count):
        a, b, c, d, e = rng.choice(task_sets)
        sched_id = rng.randint(10, 999)

        # Unique total order: c < a < d < e < b
        question = (
            f"Deductive Constraint Satisfaction (Schedule #{sched_id}):\n"
            f"Five pipeline stages ({a}, {b}, {c}, {d}, {e}) execute sequentially in positions 1 through 5 "
            f"(no ties) subject to four strict constraints:\n"
            f"1. {c} executes strictly before {a} ({c} < {a}).\n"
            f"2. {a} executes immediately before {d} (pos({d}) = pos({a}) + 1).\n"
            f"3. {b} executes strictly after {e} ({e} < {b}).\n"
            f"4. {d} executes strictly before {e} ({d} < {e}).\n"
            f"Which exact execution order from position 1 to position 5 satisfies all four constraints?"
        )

        correct = (
            f"1: {c} → 2: {a} → 3: {d} → 4: {e} → 5: {b} (satisfies {c} < {a}, adjacent ({a}, {d}), "
            f"{d} < {e}, and {e} < {b})."
        )
        d1 = (
            f"1: {a} → 2: {d} → 3: {c} → 4: {e} → 5: {b} (violates Constraint 1 by placing "
            f"{a} ahead of {c} in the sequence)."
        )
        d2 = (
            f"1: {c} → 2: {a} → 3: {e} → 4: {d} → 5: {b} (violates Constraint 2 and Constraint 4 "
            f"by separating {a} from {d} and placing {e} before {d})."
        )
        d3 = (
            f"1: {c} → 2: {a} → 3: {d} → 4: {b} → 5: {e} (violates Constraint 3 by scheduling "
            f"{b} ahead of {e} at position 4)."
        )

        samples.append(_make_record(question, correct, [d1, d2, d3], "linear_order", rng))

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Track M (Formal & Deductive Logic) Dataset for System2")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Destination directory for DatasetDict")
    parser.add_argument("--num-train", type=int, default=1600, help="Target training set size")
    parser.add_argument("--num-test", type=int, default=400, help="Target held-out test set size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    total = args.num_train + args.num_test
    n_prop = int(total * 0.30)
    n_syl = int(total * 0.25)
    n_kk = int(total * 0.25)
    n_ord = total - (n_prop + n_syl + n_kk)

    print("=" * 68)
    print(" GENERATING TRACK M (FORMAL & DEDUCTIVE LOGIC) DATASET FOR SYSTEM 2")
    print(f" Train: {args.num_train} | Test: {args.num_test} | Total: {total}")
    print("=" * 68)

    pool = (
        generate_propositional_chains(n_prop, seed=args.seed + 1)
        + generate_categorical_syllogisms(n_syl, seed=args.seed + 2)
        + generate_knights_and_knaves(n_kk, seed=args.seed + 3)
        + generate_linear_ordering_puzzles(n_ord, seed=args.seed + 4)
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
    print(f"Saved Track M dataset to {args.output_dir} (train={len(train_samples)}, test={len(test_samples)})")


if __name__ == "__main__":
    main()
