"""sys2.causal - Structural Causal Models & Counterfactual Engine.

Exports:
- StructuralCausalModel: DAG-based SCM supporting Pearl's do-calculus.
- CausalNode: Graph node container.
- CausalQueryResult: Outcome of causal or counterfactual queries.
"""

from sys2.causal.scm import StructuralCausalModel, CausalNode, CausalQueryResult

__all__ = [
    "StructuralCausalModel",
    "CausalNode",
    "CausalQueryResult",
]
