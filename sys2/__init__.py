"""sys2 - Non-autoregressive System 2 Deliberation Engine & CLI.

Exports:
- System2Engine: Unified multi-domain deliberation engine with adapter hot-swapping.
- DeliberationResult: Structured result container with calibrated confidence and head consensus.
"""

from sys2.engine import System2Engine, DeliberationResult

__all__ = ["System2Engine", "DeliberationResult"]
