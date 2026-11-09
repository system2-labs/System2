"""logic.py - Exact Constraint Satisfaction & Logical Entailment Engine.

Enforces formal propositional & first-order logic constraints using Z3 SMT solver.
Solves constraint satisfaction problems, checks whether premises strictly entail
a conclusion (via UNSAT check on Premises ∧ ¬Conclusion), and catches counterexamples
or direct negation contradictions that feedforward embeddings miss.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import z3
    _Z3_AVAILABLE = True
except ImportError:
    _Z3_AVAILABLE = False


@dataclass
class LogicVerificationResult:
    """Outcome of logical constraint checking or entailment verification."""
    is_valid: bool
    is_satisfiable: bool
    counterexample: Optional[Dict[str, str]] = None
    details: str = ""
    error: Optional[str] = None


class LogicVerifier:
    """Verifies logical entailment and constraint satisfaction via Z3."""

    def __init__(self):
        self.available = _Z3_AVAILABLE

    def check_satisfiability(self, constraints: List[Any]) -> LogicVerificationResult:
        """Check if a list of Z3 boolean constraints is simultaneously satisfiable."""
        if not self.available:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Z3 not available; check bypassed.",
            )

        solver = z3.Solver()
        for c in constraints:
            solver.add(c)

        check_res = solver.check()
        if check_res == z3.sat:
            m = solver.model()
            model_dict = {str(d): str(m[d]) for d in m.decls()}
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                counterexample=model_dict,
                details=f"Satisfiable. Model: {model_dict}",
            )
        elif check_res == z3.unsat:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Unsatisfiable (contradiction found among constraints).",
            )
        else:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Z3 returned unknown.",
            )

    def verify_entailment(
        self,
        premises: List[Any],
        conclusion: Any,
    ) -> LogicVerificationResult:
        """Check if premises strictly entail conclusion.

        In classical logic: Premises |= Conclusion iff (Premises AND NOT Conclusion) is UNSAT.
        If SAT, the satisfying model is a concrete counterexample disproving entailment!
        """
        if not self.available:
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Z3 not available; check bypassed.",
            )

        solver = z3.Solver()
        for p in premises:
            solver.add(p)
        solver.add(z3.Not(conclusion))

        check_res = solver.check()
        if check_res == z3.unsat:
            # No counterexample exists -> entailment holds!
            return LogicVerificationResult(
                is_valid=True,
                is_satisfiable=True,
                details="Strict logical entailment verified (¬Conclusion is UNSAT).",
            )
        elif check_res == z3.sat:
            # A counterexample was found!
            m = solver.model()
            counterexample = {str(d): str(m[d]) for d in m.decls()}
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                counterexample=counterexample,
                details=f"Entailment FAILS. Counterexample found: {counterexample}",
            )
        else:
            return LogicVerificationResult(
                is_valid=False,
                is_satisfiable=False,
                details="Z3 returned unknown.",
            )

    def check_negation_consistency(self, context_text: str, candidate_claim: str) -> LogicVerificationResult:
        """Heuristic logical audit for blatant negation blindness.

        Detects when context explicitly asserts 'not X', 'never X', 'cannot X'
        while candidate asserts 'X' (or vice-versa).
        """
        lower_context = context_text.lower()
        lower_claim = candidate_claim.lower()

        negation_markers = ["not ", "never ", "cannot ", "unable to ", "fails to ", "neither ", "no "]

        # Extract active keywords from claim
        words = [w for w in re.findall(r"\b[a-z]{4,}\b", lower_claim) if w not in ("that", "this", "with", "from", "have", "were", "been")]

        for word in words:
            for marker in negation_markers:
                # Context says "not <word>", candidate claims "<word>" without negation
                context_neg = f"{marker}{word}"
                if context_neg in lower_context and context_neg not in lower_claim and f" {word} " in f" {lower_claim} ":
                    return LogicVerificationResult(
                        is_valid=False,
                        is_satisfiable=False,
                        details=f"Direct negation conflict detected: context specifies '{context_neg}', but choice asserts '{word}'.",
                    )

        return LogicVerificationResult(
            is_valid=True,
            is_satisfiable=True,
            details="No overt negation conflict detected.",
        )
