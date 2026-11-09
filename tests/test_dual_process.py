"""Unit tests for the Unified Dual-Process Cognitive Pipeline."""

import pytest
from sys2.pipeline import DualProcessPipeline, DualProcessResult


def test_dual_process_components():
    pipeline = DualProcessPipeline(device="cpu")
    assert pipeline.sys1 is not None
    assert pipeline.unit_verifier is not None
    assert pipeline.algebra_verifier is not None
    assert pipeline.logic_verifier is not None
    assert pipeline.mcts_engine is not None


def test_dual_process_symbolic_unit_veto():
    """Test that System 2 correctly vetoes dimensionally invalid choices."""
    pipeline = DualProcessPipeline(device="cpu")
    question = "Which option represents valid work or energy?"
    # Choice A: 10 N * 5 m (valid energy [M L^2 T^-2])
    # Choice B: 10 N / 5 m (invalid energy [M T^-2])
    choices = [
        "Work = 10 N * 5 m",
        "Work = 10 N / 5 m",
    ]

    res = pipeline.evaluate(question, choices, domain="physics", force_sys2=True)
    assert res.sys2_verified is True
    assert res.winning_choice_idx == 0
    assert 1 in res.vetoed_choices
