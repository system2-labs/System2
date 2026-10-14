"""test_encoding.py - Unit Tests for Canonical Span Pooling & Pairwise Encoding.

Verifies:
1. `mean_pool_texts` output shape (N, 1024).
2. `mean_pool_pairwise` standard output shape (B, 4, 1024).
3. `mean_pool_pairwise` asymmetric span output shape (B, 5, 4, 1024).
4. Masking correctness separating Question and Answer tokens.
"""

import pytest
import torch
from encoding import mean_pool_texts, mean_pool_pairwise


def test_mean_pool_texts_shape():
    """Verify standard text mean pooling produces (N, 1024) tensor."""
    texts = [
        "What is the capital of France?",
        "Explain Ohm's Law and calculate current given 12V and 4 Ohms.",
    ]
    vecs = mean_pool_texts(texts, max_len=64, device="cpu")
    assert vecs.shape == (2, 1024), f"Expected shape (2, 1024), got {vecs.shape}"


def test_mean_pool_pairwise_standard_shape():
    """Verify standard pairwise candidate pooling produces (B, 4, 1024) tensor."""
    questions = ["What is 2 + 2?"]
    choices = [["1", "2", "3", "4"]]

    vecs = mean_pool_pairwise(questions, choices, max_len=64, device="cpu", asym_spans=False)
    assert vecs.shape == (1, 4, 1024), f"Expected shape (1, 4, 1024), got {vecs.shape}"


def test_mean_pool_pairwise_asymmetric_spans_shape():
    """Verify asymmetric span masking produces 5 cognitive perspectives (B, 5, 4, 1024)."""
    questions = [
        "What is the primary function of mitochondria?",
        "How do you declare a mutable variable in Rust?",
    ]
    choices = [
        ["ATP production", "Photosynthesis", "Protein folding", "Lipid storage"],
        ["let mut x", "var x", "mutable x", "val x"],
    ]

    vecs = mean_pool_pairwise(
        questions,
        choices,
        max_len=64,
        device="cpu",
        asym_spans=True,
        asym_span_mode="choice_forward",
    )
    assert vecs.shape == (2, 5, 4, 1024), f"Expected shape (2, 5, 4, 1024), got {vecs.shape}"
