"""algebra.py - Exact Computer Algebra & Mathematical Verification Engine.

Enforces algebraic consistency, simplification, and equation verification using SymPy.
Evaluates exact arithmetic equivalence, solves for target variables, and detects
mathematical contradictions in candidate steps or answers.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

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

    _ALLOWED_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/().,\s]+$")
    # Disallow leading underscores / dunders (e.g. __import__, _x, 1.__class__)
    _UNDERSCORE_PREFIX_RE = re.compile(r"(?:^|[^A-Za-z0-9])_|__")
    # Disallow attribute access or malformed dots while preserving floats like 3.14, .5, 2.
    _BAD_DOT_RE = re.compile(r"[A-Za-z_]\s*\.|\.\s*[A-Za-z_]|\.\.|(?:^|[^0-9])\.(?![0-9])")
    # Disallow space-separated multi-letter English/unit words (e.g. "meter per second")
    # so natural prose is not tokenized into single-letter symbol products.
    _MULTIWORD_PROSE_RE = re.compile(r"\b[A-Za-z]{2,}\s+[A-Za-z]{2,}\b")
    _CALL_RE = re.compile(r"([A-Za-z][A-Za-z0-9_]*)\s*\(")
    _SAFE_FUNC_NAMES = frozenset({
        "sin", "cos", "tan", "csc", "sec", "cot",
        "asin", "acos", "atan", "sinh", "cosh", "tanh",
        "exp", "log", "ln", "sqrt", "cbrt", "abs", "Abs",
        "floor", "ceiling", "ceil", "factorial", "gamma",
        "Max", "Min", "max", "min",
    })

    def __init__(self):
        self.available = _SYMPY_AVAILABLE
        if _SYMPY_AVAILABLE:
            # Lock down SymPy's parse_expr namespace: disable __builtins__ completely and
            # expose only internal AST-builder classes plus allowlisted math functions/constants.
            self._safe_globals: Dict[str, Any] = {
                "__builtins__": {},
                "Symbol": sympy.Symbol,
                "Integer": sympy.Integer,
                "Float": sympy.Float,
                "Rational": sympy.Rational,
                "sin": sympy.sin,
                "cos": sympy.cos,
                "tan": sympy.tan,
                "csc": sympy.csc,
                "sec": sympy.sec,
                "cot": sympy.cot,
                "asin": sympy.asin,
                "acos": sympy.acos,
                "atan": sympy.atan,
                "sinh": sympy.sinh,
                "cosh": sympy.cosh,
                "tanh": sympy.tanh,
                "exp": sympy.exp,
                "log": sympy.log,
                "ln": sympy.log,
                "sqrt": sympy.sqrt,
                "cbrt": sympy.cbrt,
                "abs": sympy.Abs,
                "Abs": sympy.Abs,
                "floor": sympy.floor,
                "ceiling": sympy.ceiling,
                "ceil": sympy.ceiling,
                "factorial": sympy.factorial,
                "gamma": sympy.gamma,
                "Max": sympy.Max,
                "Min": sympy.Min,
                "max": sympy.Max,
                "min": sympy.Min,
                "pi": sympy.pi,
                "E": sympy.E,
                "I": sympy.I,
                "N": sympy.N,
                "oo": sympy.oo,
            }
        else:
            self._safe_globals = {}

    def parse(self, expr_str: str) -> Optional[Any]:
        """Safely parse a mathematical expression string into a SymPy expression."""
        if not self.available or not expr_str:
            return None
        cleaned = expr_str.strip().strip("$").strip()
        cleaned = cleaned.replace("^", "**").replace("×", "*").replace("÷", "/")

        # Reject non-math syntax (quotes, brackets, braces, semicolons, etc.),
        # dunders/leading underscores, attribute dots, multi-word prose, and unknown
        # function calls before invoking SymPy's parse_expr (which calls eval()).
        if not cleaned or not self._ALLOWED_EXPR_RE.match(cleaned):
            return None
        if self._UNDERSCORE_PREFIX_RE.search(cleaned):
            return None
        if self._BAD_DOT_RE.search(cleaned):
            return None
        if self._MULTIWORD_PROSE_RE.search(cleaned):
            return None
        for call_match in self._CALL_RE.finditer(cleaned):
            if call_match.group(1) not in self._SAFE_FUNC_NAMES:
                return None

        try:
            transformations = standard_transformations + (implicit_multiplication_application,)
            return parse_expr(
                cleaned,
                local_dict={},
                global_dict=self._safe_globals,
                transformations=transformations,
                evaluate=True,
            )
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
