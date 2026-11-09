"""Unit tests for System 2 Symbolic Verification Engines (Pint, SymPy, Z3)."""

import pytest
from sys2.symbolic.units import UnitVerifier
from sys2.symbolic.algebra import AlgebraVerifier
from sys2.symbolic.logic import LogicVerifier


def test_unit_verifier_homogeneity():
    verifier = UnitVerifier()
    # Valid equation: Force = mass * acceleration
    res_valid = verifier.verify_equation("10 N", "2 kg * 5 m / s^2")
    assert res_valid.is_dimensionally_homogeneous is True
    assert res_valid.is_valid is True

    # Invalid equation: Length = Time
    res_invalid = verifier.verify_equation("10 m", "10 s")
    assert res_invalid.is_dimensionally_homogeneous is False
    assert res_invalid.is_valid is False

    # Statement scanning
    stmt_res = verifier.verify_statement("Energy = 100 J = 10 N * 10 m")
    assert stmt_res.is_dimensionally_homogeneous is True


def test_algebra_verifier_equivalence():
    verifier = AlgebraVerifier()
    # (x + 1)^2 == x^2 + 2x + 1
    res = verifier.verify_equivalence("(x + 1)^2", "x^2 + 2*x + 1")
    assert res.is_equivalent is True
    assert res.is_valid is True

    # Inequivalent expressions
    res_neq = verifier.verify_equivalence("x^2 + 1", "x^2 - 1")
    assert res_neq.is_equivalent is False


def test_algebra_verifier_solve():
    verifier = AlgebraVerifier()
    # Solve 2x + 6 = 12 -> x = 3
    res = verifier.solve_for("2*x + 6 = 12", "x")
    assert res.is_valid is True
    assert "3" in res.solutions


def test_logic_verifier_entailment_and_counterexample():
    import z3
    verifier = LogicVerifier()

    # P implies Q, P is true => Q must be true (Modus Ponens)
    p = z3.Bool("P")
    q = z3.Bool("Q")

    premises = [z3.Implies(p, q), p]
    conclusion = q

    res = verifier.verify_entailment(premises, conclusion)
    assert res.is_valid is True
    assert res.is_satisfiable is True

    # Fallacy: Affirming the consequent (P implies Q, Q is true => P is true? NO!)
    premises_fallacy = [z3.Implies(p, q), q]
    res_fallacy = verifier.verify_entailment(premises_fallacy, p)
    assert res_fallacy.is_valid is False
    assert res_fallacy.counterexample is not None

    # Contradictory premises (ex falso quodlibet): P and Not(P) -> must NOT be accepted as valid entailment
    premises_contra = [p, z3.Not(p)]
    res_contra = verifier.verify_entailment(premises_contra, q)
    assert res_contra.is_valid is False
    assert "ex falso quodlibet" in res_contra.details


def test_logic_verifier_timeout():
    import z3
    # Use very short timeout (50ms) to ensure solver does not hang
    verifier = LogicVerifier(timeout_ms=50)
    x = z3.Int("x")
    y = z3.Int("y")
    z = z3.Int("z")
    # Non-linear Diophantine constraint
    constraints = [x > 0, y > 0, z > 0, x*x*x + y*y*y == z*z*z]
    res = verifier.check_satisfiability(constraints)
    # Must terminate without hanging indefinitely
    assert res.is_valid is False or res.is_satisfiable is False


def test_logic_verifier_negation_consistency():
    verifier = LogicVerifier()
    context = "The material does not conduct electricity under room temperature."
    invalid_claim = "The material is conductive and will conduct electricity."

    res = verifier.check_negation_consistency(context, invalid_claim)
    assert res.is_valid is False
    assert "negation conflict" in res.details.lower()
