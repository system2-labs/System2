"""Unit and end-to-end tests for System 2 causal SCM integration and symbolic verification fixes.

Covers:
  1. StructuralCausalModel wiring in DualProcessPipeline.evaluate() and MCTS combined_verifier
     (Pearl do() interventions, numeric structural equations, counterfactuals, upstream
     edge-cutting vetoes, and conceptual natural-language fail-open).
  2. Tightened mathematical equation extraction (_extract_equation) and algebraic identity
     filtering (ignoring surrounding prose and physics unit strings like 'Work = 10 N * 5 m').
  3. Alignment of Step 2 and MCTS combined_verifier on symbolic physics variable substitution
     (_substitute_physics_vars).
  4. LogicVerifier.check_negation_consistency handling interrogative prompts and explanatory /
     preventive phrasing without false vetoes, while still catching declarative contradictions.
"""

from typing import List, Tuple

import pytest

from sys2.causal.scm import StructuralCausalModel
from sys2.pipeline import DualProcessPipeline
from sys2.symbolic.logic import LogicVerifier


@pytest.fixture(scope="module")
def pipeline() -> DualProcessPipeline:
    """Module-scoped CPU pipeline so ModernBERT loads once across end-to-end tests."""
    return DualProcessPipeline(device="cpu")


# -----------------------------------------------------------------------------
# 1. StructuralCausalModel Unit & Pipeline Integration Tests
# -----------------------------------------------------------------------------
def test_scm_ancestors_and_descendants():
    """Verify DAG ancestor/descendant traversal supporting Pearl edge-cutting."""
    scm = StructuralCausalModel("Chain")
    scm.add_variable("Switch", default_value=0, is_exogenous=True)
    scm.add_variable("Wire", parents=["Switch"], equation=lambda Switch: Switch)
    scm.add_variable("Light", parents=["Wire"], equation=lambda Wire: Wire)

    assert scm.ancestors("Switch") == set()
    assert scm.ancestors("Wire") == {"Switch"}
    assert scm.ancestors("Light") == {"Switch", "Wire"}

    assert scm.descendants("Switch") == {"Wire", "Light"}
    assert scm.descendants("Wire") == {"Light"}
    assert scm.descendants("Light") == set()


def test_pipeline_scm_do_intervention_switch_wire_light(pipeline: DualProcessPipeline):
    """Verify DualProcessPipeline wires SCM do(Wire = 1) on Switch -> Wire -> Light.

    With Switch = 0 and do(Wire = 1), Pearl edge-cutting keeps Switch = 0 while
    propagating Light = 1. Contradictory target values and upstream-mutation claims
    must be vetoed, and the SCM must appear in the audit trail.
    """
    question = (
        "In a causal model where Switch -> Wire -> Light with Switch = 0, "
        "what occurs under do(Wire = 1)?"
    )
    choices = [
        "Light = 0 and Switch = 0",
        "Light = 1 and Switch = 1",
        "Light = 1 and Switch = 0",
        "Switch becomes 1 because Wire is 1",
    ]

    res = pipeline.evaluate(question, choices, domain="science", force_sys2=True)
    assert res.sys2_verified is True
    assert res.all_falsified is False
    assert res.winning_choice_idx == 2
    assert 0 in res.vetoed_choices
    assert 1 in res.vetoed_choices
    assert 3 in res.vetoed_choices
    assert 2 not in res.vetoed_choices
    assert any("SCM" in entry for entry in res.audit_trail)


def test_pipeline_scm_numeric_structural_equation(pipeline: DualProcessPipeline):
    """Verify SCM executes explicit numeric structural equations (Y = 2*X, do(X = 5) -> Y = 10)."""
    question = "In a causal graph X -> Y where Y = 2*X and X = 1, what is Y under do(X = 5)?"
    choices = [
        "Y = 2",
        "Y = 10",
        "Y = 5",
    ]

    res = pipeline.evaluate(question, choices, domain="math", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 2 in res.vetoed_choices
    assert 1 not in res.vetoed_choices
    assert any("SCM" in entry for entry in res.audit_trail)


def test_pipeline_scm_multistep_numeric_preserves_upstream(pipeline: DualProcessPipeline):
    """Verify multi-step numeric equations (Y = 2*X, Z = Y + 3) preserve upstream X under do(Y = 10)."""
    question = (
        "In a causal model X -> Y -> Z where Y = 2*X and Z = Y + 3, with X = 2, "
        "what happens under do(Y = 10)?"
    )
    choices = [
        "Z = 7 and X = 2",
        "Z = 13 and X = 5",
        "Z = 13 and X = 2",
    ]

    res = pipeline.evaluate(question, choices, domain="math", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 2
    assert 0 in res.vetoed_choices
    assert 1 in res.vetoed_choices
    assert 2 not in res.vetoed_choices


def test_pipeline_scm_counterfactual_abduction(pipeline: DualProcessPipeline):
    """Verify Pearl Layer 3 counterfactual evaluation inside DualProcessPipeline."""
    question = "Factual Light was 0 because Switch was 0. What would Light have been if Switch had been 1?"
    choices = [
        "Light would remain 0",
        "Light would have been 1",
    ]

    res = pipeline.evaluate(question, choices, domain="science", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 1 not in res.vetoed_choices
    assert any("Counterfactual" in entry or "SCM" in entry for entry in res.audit_trail)


def test_pipeline_scm_counterfactual_with_equation(pipeline: DualProcessPipeline):
    """Verify counterfactual prediction with explicit numeric equation Y = 2.5*X + 1."""
    question = (
        "In a causal model X -> Y where Y = 2.5*X + 1, factually X = 2 and Y = 6. "
        "What would Y have been if X had been 4?"
    )
    choices = [
        "Y would have been 6",
        "Y would have been 11",
    ]

    res = pipeline.evaluate(question, choices, domain="math", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 1 not in res.vetoed_choices


def test_pipeline_scm_dag_edge_cutting_veto_and_single_letter_safety(pipeline: DualProcessPipeline):
    """Verify DAG A -> B -> C with do(B = 1) vetoes upstream mutation claims on A.

    Also verifies single-letter node 'A' does not collide with lowercase English article 'a'
    in valid choices like 'causes a change in C'.
    """
    question = "In a causal DAG A -> B -> C, if we apply do(B = 1), which statement is valid?"
    choices = [
        "do(B = 1) causes A to change to 1.",
        "Setting B = 1 causes a change in C, while A remains unaffected because A -> B is severed.",
        "Intervening with do(B = 1) changes upstream variable A.",
    ]

    res = pipeline.evaluate(question, choices, domain="science", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 2 in res.vetoed_choices
    assert 1 not in res.vetoed_choices


def test_pipeline_scm_conceptual_prose_fail_open(pipeline: DualProcessPipeline):
    """Purely conceptual natural-language causal questions must fail open (zero false vetoes)."""
    question = "What causes ocean tides on Earth?"
    choices = [
        "Earth's rotation",
        "Moon's gravitational pull",
        "Wind currents",
        "Sun's magnetic field",
    ]

    res = pipeline.evaluate(question, choices, domain="science", force_sys2=True)
    assert res.all_falsified is False
    assert res.vetoed_choices == []

    # Even with verb-edge + 'if' clause in prose, unknown non-formal mechanisms fail open.
    question2 = "Heat causes melting. If heat is increased, what happens to ice?"
    choices2 = [
        "The ice melts faster into liquid water.",
        "The ice freezes harder.",
    ]
    res2 = pipeline.evaluate(question2, choices2, domain="science", force_sys2=True)
    assert res2.all_falsified is False
    assert res2.vetoed_choices == []


def test_mcts_combined_verifier_runs_scm(pipeline: DualProcessPipeline, monkeypatch: pytest.MonkeyPatch):
    """Confirm MCTS combined_verifier itself enforces SCM vetoes, not only Step 2."""
    captured: List[Tuple[str, bool, str]] = []
    orig_search = pipeline.mcts_engine.search

    def _spy_search(question, choices, prior_scores=None, verifier_fn=None):
        if verifier_fn is not None:
            for c in choices:
                ok, reason = verifier_fn(question, c)
                captured.append((c, ok, reason))
        return orig_search(question, choices, prior_scores=prior_scores, verifier_fn=verifier_fn)

    monkeypatch.setattr(pipeline.mcts_engine, "search", _spy_search)

    question = "In Switch -> Wire -> Light with Switch = 0, what occurs under do(Wire = 1)?"
    choices = ["Light = 1 and Switch = 0", "Light = 0 and Switch = 0"]
    pipeline.evaluate(question, choices, domain="science", force_sys2=True)

    verdicts = {c: ok for c, ok, _ in captured}
    assert verdicts["Light = 1 and Switch = 0"] is True
    assert verdicts["Light = 0 and Switch = 0"] is False


# -----------------------------------------------------------------------------
# 2. Equation Extraction & Algebraic Identity Tightening Tests
# -----------------------------------------------------------------------------
def test_extract_equation_ignores_surrounding_prose_and_units(pipeline: DualProcessPipeline):
    """Verify _extract_equation extracts only clean math equations, never prose or units."""
    assert pipeline._extract_equation("Solve the equation 2*x + 4 = 10 for x") == "2*x + 4 = 10"
    assert pipeline._extract_equation("Find x when x^2 - 4 = 0.") == "x^2 - 4 = 0"
    assert pipeline._extract_equation("What is x if 3x - 7 = 2?") == "3x - 7 = 2"
    assert pipeline._extract_equation("Solve 2(x - 1) = 8 for x") == "2(x - 1) = 8"
    assert pipeline._extract_equation("Check (x + 1)^2 == x^2 + 2*x + 1") == "(x + 1)^2 == x^2 + 2*x + 1"

    # Must reject physics unit claims, causal assignments, prose, and inequalities.
    assert pipeline._extract_equation("Work = 10 N * 5 m") is None
    assert pipeline._extract_equation("W = 10 N * 5 m") is None
    assert pipeline._extract_equation("Velocity = 10 m/s") is None
    assert pipeline._extract_equation("The mass m = 5 kg and velocity v = 10 m/s") is None
    assert pipeline._extract_equation("The total cost = fifty dollars") is None
    assert pipeline._extract_equation("Light = 1 and Switch = 0") is None
    assert pipeline._extract_equation("x != 5 and y <= 10") is None


def test_pipeline_equation_solving_with_prose_prompt(pipeline: DualProcessPipeline):
    """Verify DualProcessPipeline solves equations embedded in English prose."""
    question = "Solve the equation 2*x + 4 = 10 for x"
    choices = ["x = 5", "x = 3", "x = 7"]

    res = pipeline.evaluate(question, choices, domain="math", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 2 in res.vetoed_choices
    assert 1 not in res.vetoed_choices

    # Bare numeric answers with explicit target variable ("for x") must also confirm/veto cleanly.
    res_bare = pipeline.evaluate(question, ["5", "3", "7"], domain="math", force_sys2=True)
    assert res_bare.sys2_verified is True
    assert res_bare.winning_choice_idx == 1
    assert 0 in res_bare.vetoed_choices
    assert 2 in res_bare.vetoed_choices
    assert 1 not in res_bare.vetoed_choices


def test_pipeline_physics_unit_strings_not_vetoed_by_algebra_verifier(pipeline: DualProcessPipeline):
    """Verify unit-bearing strings like 'Work = 10 N * 5 m' and 'Velocity = 10 m/s' are never
    mis-vetoed by SymPy AlgebraVerifier.
    """
    res_work = pipeline.evaluate(
        "Which option represents valid work or energy?",
        ["Work = 10 N * 5 m", "Work = 10 N / 5 m"],
        domain="physics",
        force_sys2=True,
    )
    assert res_work.sys2_verified is True
    assert res_work.winning_choice_idx == 0
    assert 0 not in res_work.vetoed_choices
    assert 1 in res_work.vetoed_choices
    assert not any("AlgebraVerifier" in entry for entry in res_work.audit_trail)

    # 'Velocity = 10 m/s' previously caused SymPy to split 'Velocity' into V*e*l*o*c*i*t*y and veto!
    res_vel = pipeline.evaluate(
        "Which option represents valid velocity?",
        ["Velocity = 10 m/s", "Velocity = 10 joules"],
        domain="physics",
        force_sys2=True,
    )
    assert res_vel.sys2_verified is True
    assert res_vel.winning_choice_idx == 0
    assert 0 not in res_vel.vetoed_choices
    assert 1 in res_vel.vetoed_choices
    assert not any("AlgebraVerifier" in entry for entry in res_vel.audit_trail)


def test_pipeline_algebraic_identity_check_preserved(pipeline: DualProcessPipeline):
    """Genuine false algebraic identities must still be vetoed while valid identities pass."""
    question = "Which algebraic identity holds for all real x?"
    choices = [
        "(x + 1)^2 = x^2 + 1",
        "(x + 1)^2 = x^2 + 2*x + 1",
    ]
    res = pipeline.evaluate(question, choices, domain="math", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 1
    assert 0 in res.vetoed_choices
    assert 1 not in res.vetoed_choices


# -----------------------------------------------------------------------------
# 3. Step 2 and MCTS Physics Variable Substitution Alignment Tests
# -----------------------------------------------------------------------------
def test_substitute_physics_vars_helper(pipeline: DualProcessPipeline):
    """Verify _substitute_physics_vars rewrites symbolic physics vars while preserving explicit units."""
    q_ke = "What is the formula for kinetic energy of mass m moving at velocity v?"
    assert pipeline._substitute_physics_vars(q_ke, "0.5 * m * v^2") == "0.5 * kg * (m/s)^2"
    assert pipeline._substitute_physics_vars(q_ke, "1/2 * m * v^2") == "1/2 * kg * (m/s)^2"
    assert pipeline._substitute_physics_vars(q_ke, "m * v") == "kg * (m/s)"

    # Explicit meter units ('5 m', '10 m/s') must NOT be rewritten to 'kg'.
    assert pipeline._substitute_physics_vars(q_ke, "Work = 10 N * 5 m") == "Work = 10 N * 5 m"
    assert pipeline._substitute_physics_vars(q_ke, "10 m/s") == "10 m/s"


def test_step2_and_mcts_agree_on_symbolic_physics_formulas(
    pipeline: DualProcessPipeline,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify Step 2 and MCTS combined_verifier BOTH accept '0.5 * m * v^2' and reject 'm * v',
    and BOTH accept 'm * a' for force while rejecting 'm / a'.
    """
    mcts_verdicts: List[Tuple[str, bool, str]] = []
    orig_search = pipeline.mcts_engine.search

    def _spy_search(question, choices, prior_scores=None, verifier_fn=None):
        if verifier_fn is not None:
            for c in choices:
                ok, reason = verifier_fn(question, c)
                mcts_verdicts.append((c, ok, reason))
        return orig_search(question, choices, prior_scores=prior_scores, verifier_fn=verifier_fn)

    monkeypatch.setattr(pipeline.mcts_engine, "search", _spy_search)

    # Kinetic energy: 0.5 * m * v^2 valid, m * v invalid
    q_ke = "What is the formula for kinetic energy of an object with mass m and velocity v?"
    choices_ke = ["0.5 * m * v^2", "m * v"]
    res_ke = pipeline.evaluate(q_ke, choices_ke, domain="physics", force_sys2=True)

    assert res_ke.sys2_verified is True
    assert res_ke.winning_choice_idx == 0
    assert 0 not in res_ke.vetoed_choices
    assert 1 in res_ke.vetoed_choices
    ke_map = {c: ok for c, ok, _ in mcts_verdicts}
    assert ke_map["0.5 * m * v^2"] is True
    assert ke_map["m * v"] is False

    # Force: m * a valid (previously falsely invalidated in MCTS because Pint parsed 'm' as meter
    # and 'a' as annum without physics_var_subs!), m / a invalid
    mcts_verdicts.clear()
    q_force = "Calculate the force for an object with mass m and acceleration a."
    choices_force = ["m * a", "m / a"]
    res_force = pipeline.evaluate(q_force, choices_force, domain="physics", force_sys2=True)

    assert res_force.sys2_verified is True
    assert res_force.winning_choice_idx == 0
    assert 0 not in res_force.vetoed_choices
    assert 1 in res_force.vetoed_choices
    force_map = {c: ok for c, ok, _ in mcts_verdicts}
    assert force_map["m * a"] is True
    assert force_map["m / a"] is False


# -----------------------------------------------------------------------------
# 4. LogicVerifier.check_negation_consistency False-Positive & Contradiction Tests
# -----------------------------------------------------------------------------
def test_negation_consistency_accepts_interrogative_and_explanatory_answers():
    """Negated questions and explanatory/preventive choices must NOT be falsely vetoed."""
    lv = LogicVerifier()

    safe_pairs = [
        ("Why does ice not sink in water?", "Ice floats because its lower density prevents sinking."),
        ("Which gas is NOT produced during photosynthesis?", "Carbon dioxide is consumed rather than produced."),
        ("What material cannot conduct electricity?", "Rubber is an insulator and does not conduct electricity."),
        ("Under the rule, illegally obtained evidence CANNOT be:", "Admitted unconditionally at trial."),
        ("Ice does not sink in water.", "Ice floats because its lower density prevents sinking."),
        ("The system must never delete production data.", "The system avoids deleting production data by archiving instead."),
        ("The patient cannot take penicillin.", "Prescribe azithromycin instead of penicillin."),
        ("The patient cannot take penicillin.", "Use an alternative antibiotic rather than penicillin."),
        ("The patient cannot take penicillin.", "The patient lacks tolerance for penicillin and must avoid it."),
        ("The switch is not closed.", "The switch is no longer closed and remains open."),
        ("The switch is not closed.", "The switch fails to close, leaving the circuit open."),
        ("The material does not conduct electricity under room temperature.", "It doesn't conduct electricity."),
        ("The material does not conduct electricity under room temperature.", "The material conducts heat effectively."),
    ]
    for ctx, claim in safe_pairs:
        res = lv.check_negation_consistency(ctx, claim)
        assert res.is_valid is True, f"False veto on ({ctx!r}, {claim!r}): {res.details}"


def test_negation_consistency_catches_genuine_declarative_contradictions():
    """Declarative prohibitions/negations must still veto affirmatively contradictory choices."""
    lv = LogicVerifier()

    contradictory_pairs = [
        ("The system must never delete production data.", "The system should delete production data."),
        ("The system must never delete production data.", "Delete the production data immediately."),
        ("The patient cannot take penicillin.", "The patient should take penicillin."),
        ("The patient cannot take penicillin.", "Administer penicillin to the patient."),
        ("The switch is not closed.", "The switch is closed."),
        ("The switch is not closed. What is the circuit state?", "The switch remains closed."),
        (
            "The material does not conduct electricity under room temperature.",
            "The material is conductive and will conduct electricity.",
        ),
    ]
    for ctx, claim in contradictory_pairs:
        res = lv.check_negation_consistency(ctx, claim)
        assert res.is_valid is False, f"Missed contradiction on ({ctx!r}, {claim!r})"
        assert "negation conflict" in res.details.lower()


def test_pipeline_negation_end_to_end(pipeline: DualProcessPipeline):
    """End-to-end pipeline checks: interrogative negated question passes, declarative violation vetoes."""
    # Interrogative negated question: valid explanation must NOT be vetoed.
    q_why = "Why does ice not sink in water?"
    choices_why = [
        "Ice floats because its lower density prevents sinking.",
        "Ice sinks rapidly because it is denser than liquid water.",
    ]
    res_why = pipeline.evaluate(q_why, choices_why, domain="science", force_sys2=True)
    assert 0 not in res_why.vetoed_choices
    assert res_why.all_falsified is False

    # Declarative prohibition: violating choice MUST be vetoed.
    q_decl = "The system must never delete production data. Which action is safe?"
    choices_decl = [
        "The system should delete production data.",
        "Archive production data without deleting it.",
    ]
    res_decl = pipeline.evaluate(q_decl, choices_decl, domain="safety", force_sys2=True)
    assert res_decl.sys2_verified is True
    assert 0 in res_decl.vetoed_choices
    assert 1 not in res_decl.vetoed_choices
    assert res_decl.winning_choice_idx == 1
