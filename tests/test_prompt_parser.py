"""Unit tests for code-safe benchmark prompt parsing."""

import pytest
from sys2.pipeline import DualProcessPipeline


def test_parse_standard_annotated_prompt():
    pipe = DualProcessPipeline(device="cpu")
    raw = """
    QUESTION: What causes ocean tides?
      ----------------------------------------------------------------------
         [A] [15 tok] Earth's rotation
         [B] [15 tok] Moon's gravity
         [C] [14 tok] Wind currents
         [D] [16 tok] Sun's magnetism
    """
    q, choices, meta = pipe.parse_annotated_prompt(raw)
    assert q == "What causes ocean tides?"
    assert len(choices) == 4
    assert choices[0] == "Earth's rotation"
    assert choices[1] == "Moon's gravity"
    assert meta[1]["letter"] == "B"
    assert meta[1]["tokens"] == 15


def test_parse_prompt_with_code_blocks_and_brackets():
    pipe = DualProcessPipeline(device="cpu")
    raw = """
    QUESTION: What does the following Python snippet return?
    ```python
    def compute(A):
        # Line containing variable assignment and return
        x = 10
        return [A, x]
    ```
    ----------------------------------------------------------------------
    [A] [10 tok] [5, 10]
    [B] [10 tok] [10, 5]
    [C] [10 tok] None
    [D] [10 tok] Error
    """
    q, choices, meta = pipe.parse_annotated_prompt(raw)
    assert "def compute(A):" in q
    assert "x = 10" in q
    assert "```python" in q
    assert len(choices) == 4
    assert choices[0] == "[5, 10]"
    assert choices[1] == "[10, 5]"
    assert choices[2] == "None"
    assert choices[3] == "Error"


def test_parse_prompt_with_code_choices():
    pipe = DualProcessPipeline(device="cpu")
    raw = """
    QUESTION: Which Python line correctly slices the first 5 elements?
    ----------------------------------------------------------------------
    [A] [12 tok] arr[0:5]
    [B] [12 tok] [x for x in items if x > 0]
    [C] [12 tok] arr.slice(0, 5)
    [D] [12 tok] arr[[0, 5]]
    """
    q, choices, meta = pipe.parse_annotated_prompt(raw)
    assert q == "Which Python line correctly slices the first 5 elements?"
    assert choices[0] == "arr[0:5]"
    assert choices[1] == "[x for x in items if x > 0]"
    assert choices[2] == "arr.slice(0, 5)"
    assert choices[3] == "arr[[0, 5]]"
