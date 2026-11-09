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


def test_dual_process_fail_closed():
    """Test that pipeline fails closed when ALL choices are dimensionally invalid."""
    pipeline = DualProcessPipeline(device="cpu")
    question = "Which option represents valid speed or velocity?"
    # Both choices have wrong dimensions for velocity (joules and kilograms)
    choices = [
        "The speed is 50 joules",
        "The speed is 50 kilograms",
    ]

    res = pipeline.evaluate(question, choices, domain="physics", force_sys2=True)
    assert res.all_falsified is True
    assert res.confidence == 0.0
    assert res.sys2_verified is False
    assert len(res.vetoed_choices) == 2


def test_dual_process_five_plus_choices():
    """Test that pipeline cleanly supports 5+ choices without IndexError."""
    pipeline = DualProcessPipeline(device="cpu")
    question = "Select the single valid choice:"
    choices = [f"Choice {i}" for i in range(6)]

    res = pipeline.evaluate(question, choices, domain="science")
    assert res.winning_choice_idx in range(6)
    assert res.winning_choice_char in ["A", "B", "C", "D", "E", "F"]
