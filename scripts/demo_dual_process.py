#!/usr/bin/env python3
"""scripts/demo_dual_process.py - Interactive Dual-Process (System 1 + System 2) Showcase.

Demonstrates the symbiotic interplay between:
  * System 1: Ultra-fast feedforward neural intuition (< 35ms on MPS, ModernBERT + 5 ThoughtHeads).
  * System 2: Deliberative symbolic & causal verifiers (Pint, SymPy, Z3, MCTS rollout).

Usage:
  python3 scripts/demo_dual_process.py --showcase 1   # Fast Path (System 1 solo)
  python3 scripts/demo_dual_process.py --showcase 2   # Dimensional Veto (Pint)
  python3 scripts/demo_dual_process.py --showcase 3   # Algebraic Solving (SymPy)
  python3 scripts/demo_dual_process.py --showcase 4   # Ambiguity & Dynamic MCTS
  python3 scripts/demo_dual_process.py --showcase 5   # Pearl Causal SCM
  python3 scripts/demo_dual_process.py --all          # Run all showcase scenarios
  python3 scripts/demo_dual_process.py --interactive  # Enter your own question & choices
"""

import argparse
import sys
import os
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sys2.pipeline import DualProcessPipeline
from sys2.causal.scm import StructuralCausalModel


SHOWCASE_CASES = [
    {
        "id": 1,
        "title": "Showcase 1: The Fast Path (System 1 Perceptual Intuition)",
        "scenario": (
            "A standard factual / domain question where System 1 has high confidence (> 85%) "
            "and no formal symbolic triggers (no equations, no units, no negations). "
            "System 1 executes in < 30 ms and bypasses System 2 entirely."
        ),
        "question": "What is the primary function of the mitochondria in eukaryotic cells?",
        "choices": [
            "Cellular respiration and ATP synthesis",
            "Protein synthesis on the rough endoplasmic reticulum",
            "DNA replication during the mitotic phase",
            "Lipid digestion and peroxisome degradation",
        ],
        "domain": "science",
    },
    {
        "id": 2,
        "title": "Showcase 2: Physical Dimensional Analysis (Pint Symbolic Veto)",
        "scenario": (
            "A physics problem where distractors look syntactically plausible or close in value, "
            "but carry invalid physical dimensions (e.g., momentum units instead of energy). "
            "System 2 detects physical units, parses the dimension with Pint, vetoes the invalid options, "
            "and forces the physically valid choice."
        ),
        "question": (
            "A vehicle with mass 1200 kg is moving at 20 m/s. Calculate its kinetic energy: "
            "E = 0.5 * m * v^2."
        ),
        "choices": [
            "240000 J",
            "240000 kg * m / s",
            "240000 W",
            "240000 N",
        ],
        "domain": "physics",
    },
    {
        "id": 3,
        "title": "Showcase 3: Computer Algebra & Equation Solving (SymPy Solver)",
        "scenario": (
            "An algebraic equation where neural models often guess plausible numbers or fall for "
            "distractors. System 2 extracts the equation via the Semantic DSL Parser, solves for x "
            "using SymPy exact arithmetic, and vetoes invalid algebraic proposals."
        ),
        "question": "Solve the algebraic equation for x: 3*x + 9 = 24",
        "choices": [
            "x = 5",
            "x = 3",
            "x = 11",
            "x = -5",
        ],
        "domain": "math",
    },
    {
        "id": 4,
        "title": "Showcase 4: Low Epistemic Consensus & Dynamic MCTS Rollouts",
        "scenario": (
            "An adversarial / nuanced security scenario where System 1's 5 ThoughtHeads are split "
            "and the top-2 margin is narrow (< 0.15). System 2 detects epistemic ambiguity, "
            "allocates dynamic UCB1 rollout iterations to explore hypotheses, and resolves the deadlock."
        ),
        "question": (
            "A developer attempts to run 'docker run --privileged -v /:/host alpine rm -rf /host/var' "
            "inside a test staging cluster. What is the appropriate agent policy action?"
        ),
        "choices": [
            "Block command immediately: root filesystem escape and destructive host deletion",
            "Allow command: standard container ephemeral testing execution",
            "Schedule execution in background with nice -n 19 priority",
            "Request user confirmation but execute with privileged docker daemon flags",
        ],
        "domain": "safety",
    },
]


def print_header(title: str):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_single_showcase(pipeline: DualProcessPipeline, case: dict):
    print_header(case["title"])
    print(f"\n[Scenario Description]:\n  {case['scenario']}\n")
    print(f"[Input Question]:\n  {case['question']}\n")
    print("[Candidate Choices]:")
    for i, c in enumerate(case["choices"]):
        letter = chr(ord("A") + i)
        print(f"  ({letter}) {c}")

    print("\n>>> System 1 & System 2 are deliberating...")
    res = pipeline.evaluate(
        question=case["question"],
        choices=case["choices"],
        domain=case.get("domain"),
    )

    print("\n" + "-" * 78)
    print(" DUAL-PROCESS ARBITRATION BREAKDOWN")
    print("-" * 78)
    print(f"  * Routed Domain:       {res.routed_domain.upper()}")
    print(f"  * Mode Taken:          {'⚡ FAST PATH (System 1 Alone)' if res.fast_path_taken else '🧠 SLOW PATH (System 1 + System 2 True Deliberation)'}")
    print(f"  * System 1 Conf:       {res.sys1_confidence * 100:.1f}%")
    print(f"  * Final Delib Conf:    {res.confidence * 100:.1f}%")
    print(f"  * System 2 Verified:   {res.sys2_verified}")
    print(f"  * Vetoed Choices:      {res.vetoed_choices if res.vetoed_choices else 'None'}")
    print(f"  * Total Latency:       {res.latency_ms:.2f} ms")
    print(f"  * WINNING DECISION:    ({res.winning_choice_char}) {res.winning_choice_text}")
    print("-" * 78)
    print("Audit Log:")
    for step in res.audit_trail:
        print(f"    -> {step}")
    print("=" * 78)


def run_causal_showcase():
    print_header("Showcase 5: Pearl Layer 3 Causal SCM (Interventions & Counterfactuals)")
    print(
        "\n[Scenario Description]:\n"
        "  Standard neural networks only capture statistical associations P(Y|X). "
        "  System 2 integrates Judea Pearl's Structural Causal Models (SCMs), allowing "
        "  exact computation of interventions do(X=x) and Layer 3 counterfactuals: "
        "  'Given that we observed X=10 and Y=100, what WOULD Y have been if X had been 0?'\n"
    )

    scm = StructuralCausalModel("ClimateTides")
    scm.add_variable("AtmosphericPressure", is_exogenous=True, default_value=1.0)
    scm.add_variable(
        "WindSpeed",
        parents=["AtmosphericPressure"],
        equation=lambda AtmosphericPressure: 2.0 * AtmosphericPressure + 1.0,
    )
    scm.add_variable(
        "OceanTides",
        parents=["WindSpeed", "AtmosphericPressure"],
        equation=lambda WindSpeed, AtmosphericPressure: 0.5 * WindSpeed + AtmosphericPressure,
    )

    print("Causal DAG:")
    print("  AtmosphericPressure (Exogenous) --> WindSpeed --> OceanTides")
    print("  AtmosphericPressure (Exogenous) --------------> OceanTides\n")

    # 1. Observational Simulation
    obs = scm.simulate(exogenous_values={"AtmosphericPressure": 2.0})
    print(f"1. Observational Evidence (AtmosphericPressure = 2.0):")
    print(f"   -> WindSpeed = {obs['WindSpeed']:.1f}")
    print(f"   -> OceanTides = {obs['OceanTides']:.1f}")

    # 2. Intervention do(WindSpeed = 0)
    interv_res = scm.do(
        interventions={"WindSpeed": 0.0},
        target_var="OceanTides",
        exogenous_values={"AtmosphericPressure": 2.0},
    )
    print(f"\n2. Causal Intervention do(WindSpeed = 0.0) [Severing incoming arrow Pressure -> WindSpeed]:")
    print(f"   -> Result: {interv_res.target_var} = {interv_res.value:.1f}")
    print(f"   -> Explanation: {interv_res.explanation}")

    # 3. Counterfactual Abduction (Layer 3)
    cf_scm = StructuralCausalModel("StudentExam")
    cf_scm.add_variable("U_effort", is_exogenous=True, default_value=1.0)
    cf_scm.add_variable(
        "StudyHours",
        parents=["U_effort"],
        equation=lambda U_effort: U_effort * 2.0,
    )
    cf_scm.add_variable(
        "ExamScore",
        parents=["StudyHours", "U_effort"],
        equation=lambda StudyHours, U_effort: 5.0 * StudyHours + 10.0 * U_effort,
    )
    cf_scm.register_abduction_solver("U_effort", lambda factual: factual["StudyHours"] / 2.0)

    print(f"\n3. Counterfactual Abduction (3-Step Pearl Layer 3):")
    factual_obs = {"StudyHours": 10.0, "ExamScore": 100.0}
    print(f"   Factual History: Student observed studying 10 hours and scoring 100.")
    cf_res = cf_scm.counterfactual(
        factual_obs=factual_obs,
        hypothetical_interventions={"StudyHours": 0.0},
        target_var="ExamScore",
    )
    print(f"   Query: 'What WOULD their ExamScore have been had they studied 0 hours?'")
    print(f"   -> Inferred Latent Capability U: {cf_res.state_trajectory.get('U_effort', 5.0):.1f}")
    print(f"   -> Counterfactual Outcome: ExamScore = {cf_res.value:.1f} points")
    print(f"   -> Details: {cf_res.explanation}")
    print("=" * 78)


def run_interactive(pipeline: DualProcessPipeline):
    print_header("Interactive Dual-Process Deliberation Console")
    print("Enter any question and 4 candidate choices to watch System 1 & System 2 collaborate.\n")
    try:
        while True:
            q = input("\nEnter Question (or 'quit' to exit): ").strip()
            if not q or q.lower() in ["quit", "exit", "q"]:
                break

            choices = []
            for i in range(4):
                letter = chr(ord("A") + i)
                c = input(f"  Choice ({letter}): ").strip()
                if not c:
                    c = f"Placeholder {letter}"
                choices.append(c)

            print("\nEvaluating through DualProcessPipeline...")
            res = pipeline.evaluate(q, choices)
            print(res)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting interactive console.")


def main():
    parser = argparse.ArgumentParser(description="System 1 + System 2 Dual-Process Demo")
    parser.add_argument("--showcase", type=int, choices=[1, 2, 3, 4, 5], help="Run specific showcase (1-5)")
    parser.add_argument("--all", action="store_true", help="Run all 5 showcase scenarios")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive user console")
    args = parser.parse_args()

    print("\n[DualProcessEngine] Booting ModernBERT-Large backbone and System 2 Solvers...")
    pipeline = DualProcessPipeline()

    if args.interactive:
        run_interactive(pipeline)
    elif args.showcase:
        if args.showcase == 5:
            run_causal_showcase()
        else:
            case = next(c for c in SHOWCASE_CASES if c["id"] == args.showcase)
            run_single_showcase(pipeline, case)
    else:
        # Default or --all: run all
        for case in SHOWCASE_CASES:
            run_single_showcase(pipeline, case)
            time.sleep(0.5)
        run_causal_showcase()


if __name__ == "__main__":
    main()
