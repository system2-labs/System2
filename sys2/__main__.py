"""__main__.py - Command-Line Interface for System 2 Non-Autoregressive Deliberation.

Usage:
  # 1. Adjudicate a question with 4 choices:
  python3 -m sys2 "Under contract law, what element creates mutual assent?" \
    --choices "Offer and acceptance" "Unilateral expectation" "Past consideration" "Promissory estoppel"

  # 2. Interactive Deliberation Terminal:
  python3 -m sys2 --interactive

  # 3. Dynamic Domain Route Only (< 0.1 ms):
  python3 -m sys2 --route "A vehicle triggers DTC code P0300 with active misfire counts."
"""

import argparse
import os
import sys

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import transformers
transformers.logging.set_verbosity_error()

from sys2.engine import System2Engine


def main():
    parser = argparse.ArgumentParser(
        prog="sys2",
        description="System 1 Non-Autoregressive Feedforward Perception & Dynamic Domain Router",
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
        help="Manually override domain track"
    )
    parser.add_argument("--route", action="store_true", help="Execute two-stage domain routing only without full evaluation")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive perception REPL")
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
        print(" SYSTEM 1 PARALLEL PERCEPTION INTERACTIVE TERMINAL")
        print(" (Ctrl+C to exit)")
        print("========================================================================")
        engine = System2Engine(device=args.device)
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

                print("\n[Deliberating across 5 ThoughtHeads + Judger...]")
                result = engine.adjudicate(q, choices, domain=args.domain)
                print(str(result))
            except (KeyboardInterrupt, EOFError):
                print("\nExiting System 2 Terminal.")
                break
        return

    if args.prompt and args.choices:
        if len(args.choices) < 2:
            print("Error: At least 2 choices required for contrastive deliberation.", file=sys.stderr)
            sys.exit(1)
        # Pad to 4 choices if needed
        choices = list(args.choices)
        while len(choices) < 4:
            choices.append("N/A")

        engine = System2Engine(device=args.device)
        result = engine.adjudicate(args.prompt, choices, domain=args.domain)
        print(str(result))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
