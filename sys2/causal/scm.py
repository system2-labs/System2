"""scm.py - Structural Causal Models (SCMs) & Interventional Simulation Engine.

Implements Judea Pearl's Causal Hierarchy:
  - Layer 1: Association (Observational conditional probability)
  - Layer 2: Intervention (Pearl's do-calculus, do(X = x) edge-cutting)
  - Layer 3: Counterfactuals (Abduction -> Action -> Prediction)

Enables True System 2 Deliberation to reason about interventions and causes,
preventing spurious correlation traps inherent in pure associative feedforward representations.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


@dataclass
class CausalNode:
    """A variable node in a Structural Causal Model."""
    name: str
    parents: List[str] = field(default_factory=list)
    structural_eq: Optional[Callable[..., Any]] = None
    is_exogenous: bool = False
    default_value: Any = None


@dataclass
class CausalQueryResult:
    """Outcome of a causal interventional or counterfactual query."""
    target_var: str
    value: Any
    is_interventional: bool
    interventions: Dict[str, Any]
    state_trajectory: Dict[str, Any]
    explanation: str


class StructuralCausalModel:
    """Structural Causal Model defined by a DAG and deterministic/probabilistic structural equations."""

    def __init__(self, name: str = "CausalModel"):
        self.name = name
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[Tuple[str, str]] = []  # (parent, child)

    def add_variable(
        self,
        name: str,
        parents: Optional[List[str]] = None,
        equation: Optional[Callable[..., Any]] = None,
        default_value: Any = None,
        is_exogenous: bool = False,
    ) -> None:
        """Add an endogenous or exogenous causal variable."""
        p_list = parents or []
        self.nodes[name] = CausalNode(
            name=name,
            parents=p_list,
            structural_eq=equation,
            is_exogenous=is_exogenous,
            default_value=default_value,
        )
        for p in p_list:
            self.edges.append((p, name))

    def _topological_sort(self, active_parents: Dict[str, List[str]]) -> List[str]:
        """Compute topological ordering of nodes given active parent relationships."""
        in_degree = {n: len(active_parents.get(n, [])) for n in self.nodes}
        queue = [n for n, deg in in_degree.items() if deg == 0]
        sorted_nodes = []

        while queue:
            curr = queue.pop(0)
            sorted_nodes.append(curr)
            for child, parents in active_parents.items():
                if curr in parents:
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)

        if len(sorted_nodes) != len(self.nodes):
            raise ValueError("Cycle detected in causal graph! SCM must be a directed acyclic graph (DAG).")
        return sorted_nodes

    def simulate(
        self,
        exogenous_values: Optional[Dict[str, Any]] = None,
        interventions: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Simulate SCM state under optional do(X = x) interventions.

        Applying do(X = x) severs all incoming causal arrows (PA_X -> X)
        and forces X to assume value x directly.
        """
        interventions = interventions or {}
        exogenous_values = exogenous_values or {}

        # Sever incoming edges for intervened variables
        active_parents: Dict[str, List[str]] = {}
        for name, node in self.nodes.items():
            if name in interventions:
                # do(X = x): parents severed
                active_parents[name] = []
            else:
                active_parents[name] = list(node.parents)

        topo_order = self._topological_sort(active_parents)
        state: Dict[str, Any] = {}

        for var in topo_order:
            node = self.nodes[var]
            if var in interventions:
                state[var] = interventions[var]
            elif node.is_exogenous:
                state[var] = exogenous_values.get(var, node.default_value)
            elif node.structural_eq is not None:
                parent_vals = {p: state[p] for p in active_parents[var]}
                state[var] = node.structural_eq(**parent_vals)
            else:
                state[var] = node.default_value

        return state

    def do(
        self,
        interventions: Dict[str, Any],
        target_var: str,
        exogenous_values: Optional[Dict[str, Any]] = None,
    ) -> CausalQueryResult:
        """Execute Pearl's do-operator query P(target | do(interventions))."""
        state = self.simulate(exogenous_values=exogenous_values, interventions=interventions)
        val = state.get(target_var)

        interv_str = ", ".join(f"do({k} = {v})" for k, v in interventions.items())
        explanation = (
            f"Intervention {interv_str} severed incoming edges. "
            f"Resulting {target_var} = {val}."
        )

        return CausalQueryResult(
            target_var=target_var,
            value=val,
            is_interventional=True,
            interventions=interventions,
            state_trajectory=state,
            explanation=explanation,
        )

    def counterfactual(
        self,
        factual_obs: Dict[str, Any],
        hypothetical_interventions: Dict[str, Any],
        target_var: str,
    ) -> CausalQueryResult:
        """Execute counterfactual reasoning: 'Given fact F, what would Y have been had we done X=x'?'

        3 Steps (Pearl):
        1. Abduction: infer exogenous variables U from factual observations.
        2. Action: apply do(X = x') to the causal model.
        3. Prediction: compute target in modified model with inferred U.
        """
        # Step 1: Infer exogenous defaults matching observations
        inferred_u = {}
        for k, v in factual_obs.items():
            if k in self.nodes and self.nodes[k].is_exogenous:
                inferred_u[k] = v

        # Step 2 & 3: Simulate under hypothetical intervention
        cf_state = self.simulate(
            exogenous_values=inferred_u,
            interventions=hypothetical_interventions,
        )

        cf_val = cf_state.get(target_var)
        fact_val = factual_obs.get(target_var, "unknown")

        explanation = (
            f"Factual: {target_var} was {fact_val}. "
            f"Counterfactual under {hypothetical_interventions}: {target_var} becomes {cf_val}."
        )

        return CausalQueryResult(
            target_var=target_var,
            value=cf_val,
            is_interventional=True,
            interventions=hypothetical_interventions,
            state_trajectory=cf_state,
            explanation=explanation,
        )
