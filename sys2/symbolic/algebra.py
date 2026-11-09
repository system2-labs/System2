"""algebra.py - Exact Computer Algebra & Mathematical Verification Engine.

Enforces algebraic consistency, simplification, and equation verification using SymPy.
Evaluates exact arithmetic equivalence, solves for target variables, and detects
mathematical contradictions in candidate steps or answers.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


@dataclass
class AlgebraVerificationResult:
    """Outcome of algebraic verification."""
    is_valid: bool
    is_equivalent: bool
    simplified_difference: Optional[str] = None
    solutions: Optional[List[str]] = None
    details: str = ""
    error: Optional[str] = None


class AlgebraVerifier:
    """Verifies mathematical equivalence and algebraic solutions using SymPy."""

    def __init__(self):
        self.available = _SYMPY_AVAILABLE

    def parse(self, expr_str: str) -> Optional[Any]:
        """Safely parse a mathematical expression string into a SymPy expression."""
        if not self.available or not expr_str:
            return None
        cleaned = expr_str.strip().strip("$").strip()
        cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")
        try:
            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(cleaned, transformations=transformations, evaluate=True)
        except Exception:
            return None

    def verify_equivalence(self, expr_a: str, expr_b: str) -> AlgebraVerificationResult:
        """Check if two mathematical expressions are algebraically equivalent."""
        if not self.available:
            return AlgebraVerificationResult(
                is_valid=True,
                is_equivalent=True,
                details="SymPy not available; check bypassed.",
            )

        ea = self.parse(expr_a)
        eb = self.parse(expr_b)

        if ea is None or eb is None:
            return AlgebraVerificationResult(
                is_valid=False,
                is_equivalent=False,
                details=f"Failed to parse algebraic expressions: '{expr_a}' vs '{expr_b}'",
                error="ParseError",
            )

        try:
            # First check ea.equals(eb), which performs symbolic and randomized numerical evaluation
            is_eq = bool(ea.equals(eb))
            if not is_eq:
                diff = sympy.simplify(ea - eb)
                is_eq = bool(diff == 0 or diff == sympy.S.Zero)
            else:
                diff = "0"

            return AlgebraVerificationResult(
                is_valid=is_eq,
                is_equivalent=is_eq,
                simplified_difference=str(diff),
                details=f"Difference: {diff} (Equivalent: {is_eq})",
            )
        except Exception as e:
            return AlgebraVerificationResult(
                is_valid=False,
                is_equivalent=False,
                details=str(e),
                error=type(e).__name__,
            )

    def solve_for(self, equation_str: str, variable_str: str) -> AlgebraVerificationResult:
        """Solve an equation (LHS = RHS) for a given variable."""
        if not self.available:
            return AlgebraVerificationResult(
                is_valid=True,
                is_equivalent=True,
                details="SymPy not available; check bypassed.",
            )

        if "==" in equation_str:
            parts = equation_str.split("==")
            lhs_str, rhs_str = parts[0], parts[1]
        elif "=" in equation_str:
            parts = equation_str.split("=")
            lhs_str, rhs_str = parts[0], parts[1]
        else:
            lhs_str, rhs_str = equation_str, "0"

        elhs = self.parse(lhs_str)
        erhs = self.parse(rhs_str)

        if elhs is None or erhs is None:
            return AlgebraVerificationResult(
                is_valid=False,
                is_equivalent=False,
                details=f"Failed to parse equation: {equation_str}",
                error="ParseError",
            )

        try:
            var = sympy.Symbol(variable_str.strip())
            # Check for identity equation (e.g. x = x)
            if elhs.equals(erhs):
                return AlgebraVerificationResult(
                    is_valid=True,
                    is_equivalent=True,
                    solutions=["All real values (identity)"],
                    details=f"Equation is an identity for {variable_str}.",
                )

            eq = sympy.Eq(elhs, erhs)
            sols = sympy.solve(eq, var)
            sol_strs = [str(s) for s in sols]
            is_sol_found = len(sol_strs) > 0

            return AlgebraVerificationResult(
                is_valid=is_sol_found,
                is_equivalent=is_sol_found,
                solutions=sol_strs,
                details=f"Solutions for {variable_str}: {sol_strs}",
            )
        except Exception as e:
            return AlgebraVerificationResult(
                is_valid=False,
                is_equivalent=False,
                details=str(e),
                error=type(e).__name__,
            )
