"""Unit tests for sys2.parser (SemanticParser and DSL claims)."""

import pytest
from sys2.parser import (
    SemanticParser,
    PhysicalClaim,
    AlgebraicClaim,
    LogicalClaim,
    CausalClaim,
)
from sys2.pipeline import DualProcessPipeline


def test_dsl_claim_dataclasses():
    """Verify typed DSL claim representations can be instantiated and hold slots."""
    phys = PhysicalClaim(
        target_concept="velocity",
        target_dimension="meter / second",
        extracted_quantities=["10 m/s"],
        claimed_relation="=",
    )
    assert phys.target_concept == "velocity"
    assert phys.target_dimension == "meter / second"
    assert phys.claimed_relation == "="
    assert len(phys.extracted_quantities) == 1

    alg = AlgebraicClaim(
        lhs="(x + 1)^2",
        rhs="x^2 + 2*x + 1",
        relation="==",
        target_var="x",
    )
    assert alg.lhs == "(x + 1)^2"
    assert alg.rhs == "x^2 + 2*x + 1"
    assert alg.relation == "=="
    assert alg.target_var == "x"

    log = LogicalClaim(
        premises=["All humans are mortal", "Socrates is human"],
        conclusion="Socrates is mortal",
        has_negation=False,
        negated_terms=[],
    )
    assert len(log.premises) == 2
    assert log.conclusion == "Socrates is mortal"
    assert log.has_negation is False

    causal = CausalClaim(
        causes=["Switch", "Wire"],
        effects=["Wire", "Light"],
        interventions={"Wire": 1},
        is_counterfactual=False,
    )
    assert causal.causes == ["Switch", "Wire"]
    assert causal.effects == ["Wire", "Light"]
    assert causal.interventions == {"Wire": 1}
    assert causal.is_counterfactual is False


def test_target_concept_resolution():
    """Verify resolution of key target concepts and their canonical dimensions."""
    parser = SemanticParser()

    concepts = [
        ("What is the velocity of the projectile?", "velocity", "meter / second"),
        ("Calculate the kinetic energy of the car.", "kinetic energy", "joule"),
        ("What is the net acceleration of the object?", "acceleration", "meter / second^2"),
        ("Determine the gravitational force acting on the mass.", "force", "newton"),
        ("What is the power output of the engine?", "power", "watt"),
        ("Find the total mass in kilograms.", "mass", "kilogram"),
        ("What is the electrical resistance of the circuit?", "resistance", "ohm"),
    ]

    for question, expected_concept, expected_unit in concepts:
        res = parser.resolve_target_concept(question)
        assert res is not None, f"Failed to resolve concept for: {question}"
        concept_name, canonical_unit = res
        assert concept_name == expected_concept
        assert canonical_unit == expected_unit


def test_compound_unit_normalization():
    """Verify normalization of verbose compound unit names."""
    parser = SemanticParser()

    cases = [
        ("10 kilogram meter per second squared", "10 N"),
        ("The speed was 25 meters per second", "The speed was 25 m/s"),
        ("acceleration of 9.8 meters per second squared", "acceleration of 9.8 meter / second^2"),
        ("energy of 100 kilogram meters squared per second squared", "energy of 100 joule"),
        ("pressure of 500 newtons per square meter", "pressure of 500 pascal"),
        ("power of 50 joules per second", "power of 50 watt"),
    ]

    for input_text, expected_substring in cases:
        norm = parser.normalize_units(input_text)
        assert expected_substring in norm, f"Failed on '{input_text}': got '{norm}'"


def test_parse_physical_claims():
    """Verify extraction of PhysicalClaim objects from questions and options."""
    parser = SemanticParser()

    question = "Which option represents valid velocity?"
    choices = [
        "The velocity is 50 meters per second",
        "The velocity is 50 joules",
        "Velocity = 100 m/s",
    ]

    target_concept, claims = parser.parse_physical_claims(question, choices)
    assert target_concept == "velocity"
    assert len(claims) == 3

    assert claims[0].target_concept == "velocity"
    assert claims[0].target_dimension == "meter / second"
    assert len(claims[0].extracted_quantities) >= 1

    assert claims[1].target_concept == "velocity"
    assert len(claims[1].extracted_quantities) >= 1

    assert claims[2].claimed_relation == "="
    assert claims[2].target_concept == "velocity"


def test_parse_algebraic_claims():
    """Verify extraction of AlgebraicClaim objects."""
    parser = SemanticParser()

    # Case 1: Explicit equation in choices
    question = "Which identity is valid?"
    choices = [
        "(x + 1)^2 == x^2 + 2*x + 1",
        "y = 2*x + 5",
        "Non-algebraic claim",
    ]
    claims = parser.parse_algebraic_claims(question, choices)
    assert len(claims) == 3
    assert claims[0] is not None
    assert claims[0].lhs == "(x + 1)^2"
    assert claims[0].rhs == "x^2 + 2*x + 1"
    assert claims[0].relation == "=="

    assert claims[1] is not None
    assert claims[1].lhs == "y"
    assert claims[1].rhs == "2*x + 5"
    assert claims[1].relation == "="

    assert claims[2] is None

    # Case 2: Equation in question with target variable
    q_solve = "Solve for x: 2*x + 6 = 12"
    choices_solve = ["3", "5"]
    claims_solve = parser.parse_algebraic_claims(q_solve, choices_solve)
    assert len(claims_solve) == 2
    assert claims_solve[0] is not None
    assert claims_solve[0].lhs == "x"
    assert claims_solve[0].rhs == "3"
    assert claims_solve[0].target_var == "x"


def test_parse_logical_claims():
    """Verify extraction of LogicalClaim objects with premises and negation detection."""
    parser = SemanticParser()

    question = (
        "The material does not conduct electricity under room temperature. "
        "Which statement is consistent?"
    )
    choices = [
        "The material is conductive and will conduct electricity.",
        "The material is an insulator and does not conduct current.",
    ]

    claims = parser.parse_logical_claims(question, choices)
    assert len(claims) == 2

    # Premise extraction
    assert any("does not conduct electricity" in p for p in claims[0].premises)

    # Negation slots
    assert claims[0].has_negation is True
    assert "conduct" in claims[0].negated_terms

    assert claims[1].has_negation is True
    assert claims[1].conclusion == choices[1]


def test_parse_causal_claims_interventions():
    """Verify extraction of interventions and causal graph structure."""
    parser = SemanticParser()

    question = "In a causal model where Switch -> Wire -> Light, what occurs under do(Wire = 1)?"
    choices = [
        "Light becomes 1 because Wire is 1",
        "Light remains 0",
    ]

    q_claim, c_claims = parser.parse_causal_claims(question, choices)
    assert q_claim is not None
    assert "Switch" in q_claim.causes
    assert "Wire" in q_claim.causes
    assert "Light" in q_claim.effects
    assert q_claim.interventions == {"Wire": 1}
    assert q_claim.is_counterfactual is False

    assert len(c_claims) == 2
    assert c_claims[0].interventions == {"Wire": 1}


def test_parse_causal_claims_counterfactual():
    """Verify counterfactual detection in causal queries."""
    parser = SemanticParser()

    question = "Factual Light was 0 because Switch was 0. What would Light have been if Switch had been 1?"
    choices = ["Light would have been 1", "Light would remain 0"]

    q_claim, c_claims = parser.parse_causal_claims(question, choices)
    assert q_claim is not None
    assert q_claim.is_counterfactual is True
    assert q_claim.interventions == {"Switch": 1}
    assert any(c.is_counterfactual for c in c_claims)


def test_dual_process_pipeline_with_semantic_parser():
    """Verify end-to-end integration of SemanticParser inside DualProcessPipeline."""
    pipeline = DualProcessPipeline(device="cpu")

    # Physical claim with compound units
    question = "Which option specifies a valid force?"
    choices = [
        "Force = 10 kilogram meter per second squared",
        "Force = 10 meter per second",
    ]

    res = pipeline.evaluate(question, choices, domain="physics", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 0
    assert 1 in res.vetoed_choices
