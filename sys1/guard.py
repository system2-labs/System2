"""sys1.guard - Fast Security Gatekeeper for Agent Commands."""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sys2.guard import (
    AgentGatekeeper,
    GuardVerdict,
    GuardResult,
    check_command,
    main,
)

__all__ = [
    "AgentGatekeeper",
    "GuardVerdict",
    "GuardResult",
    "check_command",
    "main",
]

if __name__ == "__main__":
    main()
