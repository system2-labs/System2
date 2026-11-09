"""sys1 - Non-autoregressive System 1 Fast Perceptual Screening & Routing Engine.

Provides ultra-fast (<20ms) perceptual evaluation:
- 13-track domain routing via centroid distance + lexical heuristics
- Pairwise contrastive candidate scoring over ModernBERT-Large representations
- Outlier filtering and perceptual confidence estimation
- Agent safety gatekeeping (<15ms shell command classification)

Exports:
- System1Engine: Fast perceptual scoring and domain routing engine.
- PerceptionResult: Structured result container with calibrated confidence and head consensus.
- AgentGatekeeper: Sub-20ms security verification for agent tool calls.
- GuardVerdict, GuardResult, check_command: Agent security primitives.
- DomainRouter: 13-domain semantic router.
"""

from sys2.engine import DeliberationResult as PerceptionResult
from sys2.engine import System2Engine as System1Engine
from routing.domain_router import DomainRouter
from sys2.guard import AgentGatekeeper, GuardVerdict, GuardResult, check_command

# Backward compatibility alias
System2Engine = System1Engine
DeliberationResult = PerceptionResult

__all__ = [
    "System1Engine",
    "PerceptionResult",
    "System2Engine",
    "DeliberationResult",
    "AgentGatekeeper",
    "GuardVerdict",
    "GuardResult",
    "check_command",
    "DomainRouter",
]
