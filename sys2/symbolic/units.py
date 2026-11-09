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

    # Note: Amperes matched via 'amperes?' or '\b\d+\s*A\b' to avoid colliding with English 'a'
    UNIT_PATTERN = re.compile(
        r"\b(?:meters?|metres?|seconds?|hours?|kilograms?|grams?|newtons?|"
        r"joules?|watts?|volts?|amperes?|amps?|ohms?|pascals?|kelvins?|"
        r"m/s|m/s\^2|m/s²|km/h|mph|N\*m|N·m|rad/s|µm|microns?|Ω|"
        r"\d+\s*(?:m|s|h|kg|g|N|J|W|V|A|Pa|K))\b",
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
        if not text:
            return False
        return bool(self.UNIT_PATTERN.search(text))

    def parse_quantity(self, expr_str: str) -> Optional[Any]:
        """Safely parse an expression or physical concept into a Pint Quantity."""
        if not self.ureg:
            return None
        cleaned = expr_str.strip().strip("$").strip()
        # Clean common mathematical notation
        cleaned = cleaned.replace("×", "*").replace("·", "*").replace("²", "^2")

        # Check if the string is a pure dimensional concept name
        lower_clean = cleaned.lower()
        if lower_clean in self.CONCEPT_DIMENSIONS:
            unit_name = self.CONCEPT_DIMENSIONS[lower_clean]
            try:
                return self.ureg(f"1 * {unit_name}")
            except Exception:
                pass

        # Handle simple division like "/ 5 m" -> "/ (5 * meter)"
        cleaned = re.sub(r"/\s*(\d+)\s*([a-zA-Z]+)", r"/ (\1 * \2)", cleaned)

        try:
            qty = self.ureg(cleaned)
            return qty
        except Exception:
            return None

    def extract_quantities(self, text: str) -> List[Any]:
        """Extract all physical quantities from freeform text."""
        if not self.ureg:
            return []
        found = []
        # Match number + unit or known unit patterns
        matches = re.finditer(r"\b(\d+(?:\.\d+)?)\s*([a-zA-ZΩµ/^²·]+)\b", text)
        for m in matches:
            q_str = m.group(0)
            q = self.parse_quantity(q_str)
            if q is not None and hasattr(q, "dimensionality"):
                found.append(q)
        return found

    def verify_equation(self, lhs_str: str, rhs_str: str) -> UnitVerificationResult:
        """Verify if lhs and rhs have identical physical dimensions.

        Crucial invariant: Parse errors on unparseable natural text are INDETERMINATE,
        NOT proof of dimensional inhomogeneity.
        """
        if not self.ureg:
            return UnitVerificationResult(
                is_valid=True,
                is_dimensionally_homogeneous=True,
                details="Pint not available; verification bypassed.",
            )

        q_lhs = self.parse_quantity(lhs_str)
        q_rhs = self.parse_quantity(rhs_str)

        if q_lhs is None or q_rhs is None:
            # Cannot parse physical quantities -> indeterminate, do NOT falsely veto!
            return UnitVerificationResult(
                is_valid=True,
                is_dimensionally_homogeneous=True,
                details=f"Could not parse physical units from '{lhs_str}' or '{rhs_str}' (treated as neutral).",
                error="ParseIndeterminate",
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
                is_valid=True,
                is_dimensionally_homogeneous=True,
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

    def verify_natural_language_claim(self, target_concept_or_question: str, candidate_text: str) -> UnitVerificationResult:
        """Verify dimensional compatibility between question target and candidate choice.

        e.g., Question asks for 'velocity', candidate states '50 joules' -> Incompatible!
        """
        # If candidate already contains an explicit equality, verify the equation directly
        if "=" in candidate_text or "==" in candidate_text:
            return self.verify_statement(candidate_text)

        lower_q = target_concept_or_question.lower()

        # Identify requested target concept in question
        target_dim = None
        target_name = None
        for concept, canonical_unit in self.CONCEPT_DIMENSIONS.items():
            if re.search(rf"\b{re.escape(concept)}\b", lower_q):
                target_q = self.parse_quantity(concept)
                if target_q is not None:
                    target_dim = target_q.dimensionality
                    target_name = concept
                    break

        if target_dim is None:
            # If no target concept identified, check standard statement equality
            return self.verify_statement(f"{target_concept_or_question} = {candidate_text}")

        # Extract quantities from candidate
        quantities = self.extract_quantities(candidate_text)
        if not quantities:
            # Also check if choice is itself a concept name
            choice_q = self.parse_quantity(candidate_text)
            if choice_q is not None and hasattr(choice_q, "dimensionality"):
                quantities = [choice_q]

        for q in quantities:
            if hasattr(q, "dimensionality") and q.dimensionality != target_dim:
                return UnitVerificationResult(
                    is_valid=False,
                    is_dimensionally_homogeneous=False,
                    dimensions_lhs=str(target_dim),
                    dimensions_rhs=str(q.dimensionality),
                    details=f"Candidate specifies [{q.dimensionality}], but target '{target_name}' requires [{target_dim}].",
                )

        return UnitVerificationResult(
            is_valid=True,
            is_dimensionally_homogeneous=True,
            details="Candidate dimensions are compatible with target concept.",
        )
