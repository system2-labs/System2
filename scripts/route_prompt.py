#!/usr/bin/env python3
"""route_prompt.py - Fast CLI Domain Routing Dispatcher for System2.

Usage:
  python3 scripts/route_prompt.py "Your prompt or question text here"
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from routing.domain_router import DomainRouter


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/route_prompt.py \"<prompt text>\"")
        sys.exit(1)

    prompt = sys.argv[1]
    router = DomainRouter(device="cpu")
    decision = router.route(prompt)

    print("========================================================================")
    print(" SYSTEM2 DYNAMIC DOMAIN ROUTING VERDICT")
    print("========================================================================")
    print(f" Prompt: {prompt[:120]}...")
    print(f"\n Dispatched Track: [{decision.selected_track.upper()}] {DomainRouter.TRACK_NAMES.get(decision.selected_track, '')}")
    print(f" Calibrated Confidence: {decision.confidence * 100:.1f}%")
    print(f" Routing Mechanism: {decision.method.upper()}")
    print(f" Routing Latency: {decision.latency_ms:.3f} ms")
    print("\n Domain Probability Distribution:")
    for track, prob in sorted(decision.probabilities.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"   {track:<12}: {prob * 100:5.1f}%  {bar}")
    print("========================================================================")


if __name__ == "__main__":
    main()
