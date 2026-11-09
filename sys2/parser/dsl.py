"""dsl.py - Formal Verification Domain-Specific Language (DSL) Claims.

Typed intermediate representations (ASTs) for System 2 formal verification engines:
- PhysicalClaim -> Pint (Dimensional Analysis & Homogeneity)
- AlgebraicClaim -> SymPy (Algebraic Equivalence & Variable Solving)
- LogicalClaim -> Z3 (SMT Entailment & Negation Consistency)
- CausalClaim -> StructuralCausalModel (Pearl's Interventions & Counterfactuals)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PhysicalClaim:
    """Represents a physical or dimensional claim extracted from natural text."""
    target_concept: Optional[str] = None
    target_dimension: Optional[str] = None
    extracted_quantities: List[Any] = field(default_factory=list)
    claimed_relation: Optional[str] = None


@dataclass
class AlgebraicClaim:
    """Represents an algebraic equality or relational claim."""
    lhs: str
    rhs: str
    relation: str = "=="
    target_var: Optional[str] = None


@dataclass
class LogicalClaim:
    """Represents a propositional or natural-language logical claim."""
    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    has_negation: bool = False
    negated_terms: List[str] = field(default_factory=list)


@dataclass
class CausalClaim:
    """Represents a causal hypothesis, intervention, or counterfactual claim."""
    causes: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    interventions: Dict[str, Any] = field(default_factory=dict)
    is_counterfactual: bool = False
