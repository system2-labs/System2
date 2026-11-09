"""sys1.engine - System 1 Fast Perceptual Screening Engine.

Provides <20ms feedforward candidate evaluation using frozen ModernBERT-Large
and domain-adapted lightweight adapters.
"""

import sys
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sys2.engine import (
    System2Engine as System1Engine,
    DeliberationResult as PerceptionResult,
    CHECKPOINT_MAP,
    sanitize_authority_preamble,
)

# Backward compatibility alias
System2Engine = System1Engine
DeliberationResult = PerceptionResult

__all__ = [
    "System1Engine",
    "PerceptionResult",
    "System2Engine",
    "DeliberationResult",
    "CHECKPOINT_MAP",
    "sanitize_authority_preamble",
]
