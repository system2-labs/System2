"""Unit tests for System 2 Structural Causal Models (SCMs) & Pearl's do-calculus."""

import pytest
from sys2.causal.scm import StructuralCausalModel


def test_scm_simulation_and_intervention():
    """Test Pearl's classic firing squad or switch-light causal model.

    Switch (S) -> Wire (W) -> Light (L)
    """
    scm = StructuralCausalModel("SwitchLightModel")
    scm.add_variable("Switch", default_value=0, is_exogenous=True)
    scm.add_variable(
        "Wire",
        parents=["Switch"],
        equation=lambda Switch: Switch,
    )
    scm.add_variable(
        "Light",
        parents=["Wire"],
        equation=lambda Wire: 1 if Wire == 1 else 0,
    )

    # 1. Observational simulation: Switch=0 -> Light=0
    obs_0 = scm.simulate(exogenous_values={"Switch": 0})
    assert obs_0["Light"] == 0

    obs_1 = scm.simulate(exogenous_values={"Switch": 1})
    assert obs_1["Light"] == 1

    # 2. Interventional query: do(Wire = 1) even if Switch = 0!
    # In observational correlation, Wire=1 implies Switch=1.
    # In Pearl's do-calculus, do(Wire = 1) cuts the arrow Switch -> Wire.
    interv_res = scm.do(
        interventions={"Wire": 1},
        target_var="Light",
        exogenous_values={"Switch": 0},
    )
    assert interv_res.value == 1
    assert interv_res.state_trajectory["Switch"] == 0
    assert interv_res.state_trajectory["Wire"] == 1
    assert interv_res.state_trajectory["Light"] == 1


def test_scm_counterfactual():
    """Test counterfactual: Factual Light was 0 because Switch was 0.

    Counterfactual: What would Light have been if Switch had been 1?
    """
    scm = StructuralCausalModel("CounterfactualModel")
    scm.add_variable("Switch", default_value=0, is_exogenous=True)
    scm.add_variable(
        "Light",
        parents=["Switch"],
        equation=lambda Switch: Switch,
    )

    factual = {"Switch": 0, "Light": 0}
    cf_res = scm.counterfactual(
        factual_obs=factual,
        hypothetical_interventions={"Switch": 1},
        target_var="Light",
    )
    assert cf_res.value == 1
    assert "Counterfactual under {'Switch': 1}: Light becomes 1" in cf_res.explanation
