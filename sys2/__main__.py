"""__main__.py - Command-Line Interface for System 2 Deliberation & Dual-Process Verification.

Usage:
  # 1. Standard deliberation / perception screening:
  python3 -m sys2 "Under contract law, what element creates mutual assent?" \
    --choices "Offer and acceptance" "Unilateral expectation" "Past consideration" "Promissory estoppel"

  # 2. Dual-Process verification (System 1 + System 2 symbolic / causal verifiers):
  python3 -m sys2 "Which equation correctly relates force, mass, and acceleration?" \
    --choices "F = m * a" "F = m / a" "F = m * a^2" "F = a / m" --dual

  # 3. Dynamic Domain Route Only (< 0.1 ms):
  python3 -m sys2 --route "A vehicle triggers DTC code P0300 with active misfire counts."

  # 4. Interactive Terminal:
  python3 -m sys2 --interactive
"""

import argparse
import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import transformers
transformers.logging.set_verbosity_error()

from sys2.engine import System2Engine
from sys2.pipeline import DualProcessPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="sys2",
        description="System 2 Deliberation & Dual-Process Verification Engine",
    )
    parser.add_argument("prompt", nargs="?", help="Input prompt / question to evaluate or route")
    parser.add_argument("--choices", "-c", nargs="+", help="Candidate choices (2 to 4 choices)")
    parser.add_argument(
        "--domain",
        "-d",
        choices=[
            "science", "code", "med", "electrical", "auto", "culinary",
            "math", "legal", "finance", "physics", "safety", "eval", "logic"
        ],
        help="Manually override domain track",
    )
    parser.add_argument("--dual", action="store_true", help="Execute Dual-Process pipeline (Sys1 screening + Sys2 verification)")
    parser.add_argument("--force-sys2", action="store_true", help="Force System 2 deliberative verification even if Sys1 is confident")
    parser.add_argument("--route", action="store_true", help="Execute two-stage domain routing only without full evaluation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive deliberation REPL")
    parser.add_argument("--device", default=None, help="Device to execute on (mps, cpu, cuda)")

    args = parser.parse_args()

    if args.route:
        if not args.prompt:
            print("Error: --route requires a prompt argument.", file=sys.stderr)
            sys.exit(1)
        from routing.domain_router import DomainRouter
        router = DomainRouter(device="cpu")
        decision = router.route(args.prompt)
        print(f"Routed Domain: [{decision.selected_track.upper()}] ({DomainRouter.TRACK_NAMES.get(decision.selected_track, '')})")
        print(f"Method: {decision.method} | Confidence: {decision.confidence * 100:.1f}% | Latency: {decision.latency_ms:.3f} ms")
        return

    if args.interactive:
        print("========================================================================")
        print(" SYSTEM 2 DUAL-PROCESS DELIBERATION INTERACTIVE TERMINAL")
        print(" (Ctrl+C to exit)")
        print("========================================================================")
        pipeline = DualProcessPipeline(device=args.device)
        letters = ["A", "B", "C", "D"]

        while True:
            try:
                print("\n" + "-" * 72)
                q = input("Question: ").strip()
                if not q:
                    continue
                choices = []
                for i in range(4):
                    c = input(f"Choice ({letters[i]}): ").strip()
                    choices.append(c)

                print("\n[Evaluating via Dual-Process Pipeline (System 1 + System 2)...]")
                result = pipeline.evaluate(q, choices, domain=args.domain, force_sys2=args.force_sys2)
                print(str(result))
            except KeyboardInterrupt:
                print("\nExiting System 2 interactive session.")
                break
        return

    if not args.prompt:
        parser.print_help()
        return

    if not args.choices or len(args.choices) < 2:
        print("Error: Please provide at least 2 candidate choices via --choices / -c", file=sys.stderr)
        sys.exit(1)

    if args.dual or args.force_sys2:
        pipeline = DualProcessPipeline(device=args.device)
        res = pipeline.evaluate(args.prompt, args.choices, domain=args.domain, force_sys2=args.force_sys2)
        print(str(res))
    else:
        engine = System2Engine(device=args.device)
        res = engine.adjudicate(args.prompt, args.choices, domain=args.domain)
        print(str(res))


if __name__ == "__main__":
    main()
