"""test_engine.py - Unit tests for System2Engine and Dynamic Adapter Hot-Swapping."""

import pytest
from sys2.engine import System2Engine, DeliberationResult


@pytest.fixture(scope="module")
def engine():
    # Use CPU for deterministic and lightweight test execution
    return System2Engine(device="cpu")


def test_engine_adjudicate_science(engine):
    question = "Why do plants appear green under sunlight during the process of photosynthesis?"
    choices = [
        "Chlorophyll reflects green light while absorbing red and blue light wavelengths.",
        "Green light carries higher energy than ultraviolet light photons.",
        "Mitochondria convert carbon dioxide into liquid oxygen gas.",
        "Cellulose reflects all visible spectrum frequencies uniformly.",
    ]
    res = engine.adjudicate(question, choices)
    assert isinstance(res, DeliberationResult)
    assert 0 <= res.selected_index < 4
    assert res.selected_index == 0  # Choice A is correct
    assert 0.0 <= res.confidence <= 1.0
    assert res.domain == "science"
    assert "/5" in res.head_consensus
    assert res.total_latency_ms > 0


def test_engine_hot_swap_to_code(engine):
    # Tests hot-swapping from science to code domain
    question = "What is the exact stdout output of this Python code:\n```python\nprint(list(range(3)))\n```"
    choices = [
        "[0, 1, 2]",
        "[1, 2, 3]",
        "(0, 1, 2)",
        "Error: invalid syntax",
    ]
    res = engine.adjudicate(question, choices)
    assert isinstance(res, DeliberationResult)
    assert res.domain == "code"
    assert res.selected_index == 0
    assert 0.0 <= res.confidence <= 1.0
