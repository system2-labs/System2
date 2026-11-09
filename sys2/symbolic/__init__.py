"""sys2.symbolic - Exact External Symbolic Verification Engines.

Modules:
- units: Physical dimensional analysis & Fourier homogeneity checks (via Pint).
- algebra: Exact computer algebra simplification & equation solving (via SymPy).
- logic: Propositional/SMT constraint satisfaction & entailment verification (via Z3).
"""

from sys2.symbolic.units import UnitVerifier, UnitVerificationResult
from sys2.symbolic.algebra import AlgebraVerifier, AlgebraVerificationResult
from sys2.symbolic.logic import LogicVerifier, LogicVerificationResult

__all__ = [
    "UnitVerifier",
    "UnitVerificationResult",
    "AlgebraVerifier",
    "AlgebraVerificationResult",
    "LogicVerifier",
    "LogicVerificationResult",
]
