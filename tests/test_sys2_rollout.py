"""Unit tests for System 2 MCTS Dynamic Compute Deliberation Engine."""

import pytest
from sys2.rollout.mcts import MCTSDeliberationEngine, DeliberationNode


def test_mcts_deliberation_dynamic_scaling():
    engine = MCTSDeliberationEngine(base_rollouts=10, max_rollouts=30)

    # 1. Clear margin -> uses base rollouts
    res_clear = engine.search(
        question="What is the capital of France?",
        choices=["Paris", "Berlin", "Rome", "Madrid"],
        prior_scores=[0.85, 0.05, 0.05, 0.05],
    )
    assert res_clear.best_choice_idx == 0
    assert res_clear.computation_scaled is False
    assert res_clear.total_rollouts >= 10

    # 2. Ambiguous margin -> dynamically scales compute
    res_ambiguous = engine.search(
        question="Which interpretation applies?",
        choices=["Option A", "Option B"],
        prior_scores=[0.51, 0.49],
    )
    assert res_ambiguous.computation_scaled is True
    assert res_ambiguous.total_rollouts >= 30


def test_mcts_verifier_pruning():
    engine = MCTSDeliberationEngine(base_rollouts=20, max_rollouts=40)

    # Mock verifier that invalidates choice 0 ("Violation")
    def mock_verifier(q, c):
        if "Violation" in c:
            return False, "Detected invariant violation"
        return True, "Valid"

    res = engine.search(
        question="Select valid option",
        choices=["Violation Claim", "Consistent Claim"],
        prior_scores=[0.9, 0.1],  # Choice 0 had higher prior, but is invalid!
        verifier_fn=mock_verifier,
    )

    # Choice 1 should win despite lower prior because Choice 0 was vetoed
    assert res.best_choice_idx == 1
    assert res.best_choice == "Consistent Claim"
