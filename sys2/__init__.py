"""sys2 - Non-autoregressive System 2 Deliberation Engine & CLI.

Exports:
- System2Engine: Unified multi-domain deliberation engine with adapter hot-swapping.
- DeliberationResult: Structured result container with calibrated confidence and head consensus.
"""

from sys2.engine import System2Engine, DeliberationResult

__all__ = [
    "System2Engine",
    "DeliberationResult",
    "AgentGatekeeper",
    "GuardVerdict",
    "GuardResult",
    "check_command",
    "create_app",
]


def __getattr__(name: str):
    if name in ("AgentGatekeeper", "GuardVerdict", "GuardResult", "check_command"):
        import sys2.guard as guard_mod
        return getattr(guard_mod, name)
    if name == "create_app":
        import sys2.server as server_mod
        return getattr(server_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

