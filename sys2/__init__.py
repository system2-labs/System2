"""sys2 - True System 2 Deliberation & Verification Stack.

Components:
- DualProcessPipeline: End-to-end cognitive pipeline integrating System 1 perception with System 2 verification.
- UnitVerifier: Physical dimensional analysis and Fourier homogeneity checks (Pint).
- AlgebraVerifier: Computer algebra equation solving and equivalence (SymPy).
- LogicVerifier: Propositional/SMT constraint satisfaction and entailment (Z3).
- StructuralCausalModel: Structural Causal Models with Pearl's do-calculus and counterfactuals.
- MCTSDeliberationEngine: Dynamic compute scaling via Monte Carlo Tree Search.

Backward Compatibility:
- System2Engine: Fast feedforward perception engine (aliased from sys1).
- DeliberationResult: Result container for perception passes.
"""

from sys2.engine import System2Engine, DeliberationResult
from sys2.pipeline import DualProcessPipeline, DualProcessResult
from sys2.symbolic.units import UnitVerifier, UnitVerificationResult
from sys2.symbolic.algebra import AlgebraVerifier, AlgebraVerificationResult
from sys2.symbolic.logic import LogicVerifier, LogicVerificationResult
from sys2.causal.scm import StructuralCausalModel, CausalNode, CausalQueryResult
from sys2.rollout.mcts import MCTSDeliberationEngine, DeliberationNode, MCTSSearchResult

__all__ = [
    "DualProcessPipeline",
    "DualProcessResult",
    "System2Engine",
    "DeliberationResult",
    "UnitVerifier",
    "UnitVerificationResult",
    "AlgebraVerifier",
    "AlgebraVerificationResult",
    "LogicVerifier",
    "LogicVerificationResult",
    "StructuralCausalModel",
    "CausalNode",
    "CausalQueryResult",
    "MCTSDeliberationEngine",
    "DeliberationNode",
    "MCTSSearchResult",
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
