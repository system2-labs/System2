"""units.py - Physical Dimensional Analysis & Unit Verification Engine.

Enforces Fourier's principle of dimensional homogeneity using Pint.
Verifies that physical equations and candidate claims have matching dimensionalities
(e.g., [force] == [mass] * [length] / [time]^2) and prevents category errors
like equating velocity to acceleration or adding incompatible dimensions.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import pint
    _UREF = pint.UnitRegistry()
    _PINT_AVAILABLE = True
except ImportError:
    _UREF = None
    _PINT_AVAILABLE = False


@dataclass
class UnitVerificationResult:
    """Outcome of dimensional verification."""
    is_valid: bool
    is_dimensionally_homogeneous: bool
    dimensions_lhs: Optional[str] = None
    dimensions_rhs: Optional[str] = None
    details: str = ""
    error: Optional[str] = None


class UnitVerifier:
    """Verifies dimensional homogeneity and physical unit consistency."""

    # Common physical units and symbols regex
    UNIT_PATTERN = re.compile(
        r"\b(?:meters?|metres?|m|seconds?|s|hours?|h|kg|kilograms?|grams?|g|newtons?|N|"
        r"joules?|J|watts?|W|volts?|V|amperes?|amps?|A|ohms?|Ω|pascals?|Pa|kelvins?|K|"
        r"m/s|m/s\^2|km/h|mph|N\*m|rad/s|work|energy|force|power|velocity|acceleration)\b",
        re.IGNORECASE,
    )

    CONCEPT_DIMENSIONS: Dict[str, str] = {
        "work": "joule",
        "energy": "joule",
        "kinetic energy": "joule",
        "potential energy": "joule",
        "force": "newton",
        "power": "watt",
        "velocity": "meter / second",
        "speed": "meter / second",
        "acceleration": "meter / second^2",
        "pressure": "pascal",
        "voltage": "volt",
        "current": "ampere",
        "resistance": "ohm",
        "mass": "kilogram",
        "distance": "meter",
        "length": "meter",
        "time": "second",
    }

    def __init__(self):
        if _PINT_AVAILABLE and _UREF is not None:
            self.ureg = _UREF
        else:
            self.ureg = None

    def has_unit_claims(self, text: str) -> bool:
        """Check if the text contains physical quantities, units, or dimensional concepts."""
        return bool(self.UNIT_PATTERN.search(text))

    def parse_quantity(self, expr_str: str) -> Optional[Any]:
        """Safely parse an expression or physical concept into a Pint Quantity."""
        if not self.ureg:
            return None
        cleaned = expr_str.strip().strip("$").strip()
        # Clean common mathematical notation
        cleaned = cleaned.replace("×", "*").replace("·", "*")

        # Check if the string is a pure dimensional concept name
        lower_clean = cleaned.lower()
        if lower_clean in self.CONCEPT_DIMENSIONS:
            unit_name = self.CONCEPT_DIMENSIONS[lower_clean]
            try:
                return self.ureg(f"1 * {unit_name}")
            except Exception:
                pass

        # Handle simple division like "/ 5 m" -> "/ (5 * meter)"
        # to prevent Pint treating / 5 m as (/ 5) * m
        cleaned = re.sub(r"/\s*(\d+)\s*([a-zA-Z]+)", r"/ (\1 * \2)", cleaned)

        try:
            qty = self.ureg(cleaned)
            return qty
        except Exception:
            return None

    def verify_equation(self, lhs_str: str, rhs_str: str) -> UnitVerificationResult:
        """Verify if lhs and rhs have identical physical dimensions."""
        if not self.ureg:
            return UnitVerificationResult(
                is_valid=True,
                is_dimensionally_homogeneous=True,
                details="Pint not available; verification bypassed.",
            )

        q_lhs = self.parse_quantity(lhs_str)
        q_rhs = self.parse_quantity(rhs_str)

        if q_lhs is None or q_rhs is None:
            return UnitVerificationResult(
                is_valid=False,
                is_dimensionally_homogeneous=False,
                details=f"Could not parse physical units from: '{lhs_str}' or '{rhs_str}'",
                error="ParseError",
            )

        try:
            dim_lhs = str(q_lhs.dimensionality)
            dim_rhs = str(q_rhs.dimensionality)
            is_homogeneous = bool(q_lhs.dimensionality == q_rhs.dimensionality)

            return UnitVerificationResult(
                is_valid=is_homogeneous,
                is_dimensionally_homogeneous=is_homogeneous,
                dimensions_lhs=dim_lhs,
                dimensions_rhs=dim_rhs,
                details=(
                    f"LHS dimensions: {dim_lhs} | RHS dimensions: {dim_rhs} | "
                    f"Homogeneous: {is_homogeneous}"
                ),
            )
        except Exception as e:
            return UnitVerificationResult(
                is_valid=False,
                is_dimensionally_homogeneous=False,
                details=str(e),
                error=type(e).__name__,
            )

    def verify_statement(self, text: str) -> UnitVerificationResult:
        """Scan text for an equation or relation with units and verify dimensional consistency."""
        if "=" not in text and "==" not in text:
            return UnitVerificationResult(
                is_valid=True,
                is_dimensionally_homogeneous=True,
                details="No explicit equality found to verify.",
            )

        # Split across all '=' tokens
        parts = [p.strip() for p in re.split(r"==|=", text) if p.strip()]
        if len(parts) < 2:
            return UnitVerificationResult(
                is_valid=True,
                is_dimensionally_homogeneous=True,
                details="Not enough parts around equality.",
            )

        # Check all adjacent pairs for dimensional homogeneity
        for i in range(len(parts) - 1):
            lhs_part = parts[i]
            rhs_part = parts[i + 1]
            res = self.verify_equation(lhs_part, rhs_part)
            if not res.is_dimensionally_homogeneous:
                return res

        return UnitVerificationResult(
            is_valid=True,
            is_dimensionally_homogeneous=True,
            details=f"All {len(parts)} parts are dimensionally homogeneous.",
        )
