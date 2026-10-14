"""encoding.py - Canonical Span Pooling & Text Embedding for System 2.

Deduplicates text and pairwise span encoding logic across train.py, eval.py,
interact.py, and system2_runtime.py into a single, verified, tested module.
"""

from typing import Any, List, Optional, Tuple, Union
import sys
import torch
import torch.nn.functional as F

from bert import get_bert, model as default_model, tokenizer as default_tokenizer


def mean_pool_texts(
    texts: List[str],
    max_len: int = 192,
    device: Optional[Union[str, torch.device]] = None,
    with_grad: bool = False,
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> torch.Tensor:
    """Encode a list of N texts into (N, D) representation via attention-weighted mean pooling.

    Args:
        texts: List of strings to encode.
        max_len: Maximum token sequence length (clamped to model max_position_embeddings).
        device: Device to place tensors on.
        with_grad: Whether to track gradients through the encoder.
        model: ModernBERT model instance (defaults to default_model from bert.py).
        tokenizer: Tokenizer instance (defaults to default_tokenizer from bert.py).

    Returns:
        Tensor of shape (N, 1024).
    """
    m = model if model is not None else default_model
    tok = tokenizer if tokenizer is not None else default_tokenizer
    dev = device if device is not None else next(m.parameters()).device

    ceiling = getattr(m.config, "max_position_embeddings", 8192)
    if max_len > ceiling:
        print(
            f"[encoding warning] Requested max_len={max_len} exceeds architectural ceiling ({ceiling}); clamped to {ceiling}.",
            file=sys.stderr,
        )
        max_len = ceiling

    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)

    if with_grad:
        last_hidden = m(**inputs).last_hidden_state
    else:
        with torch.no_grad():
            last_hidden = m(**inputs).last_hidden_state

    mask = inputs["attention_mask"].unsqueeze(-1)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def mean_pool_pairwise(
    questions: List[str],
    choices_list: List[List[str]],
    max_len: Union[int, str, torch.device] = 128,
    device: Optional[Union[str, torch.device, int]] = None,
    with_grad: bool = False,
    sources: Optional[List[str]] = None,
    task_prefixes: bool = False,
    span_pooling: bool = False,
    span_weight: float = 0.5,
    span_mode: str = "mean",
    asym_spans: bool = False,
    asym_span_mode: str = "default",
    model: Optional[torch.nn.Module] = None,
    tokenizer: Optional[Any] = None,
) -> torch.Tensor:
    """Pairwise encoding: (B questions, 4 choices) -> (B, 4, 1024) or (B, 5, 4, 1024).

    Constructs candidate representations with asymmetric span masking separating Question
    and Answer tokens, projecting into 5 distinct cognitive perspectives.

    Returns:
        Tensor of shape (B, 5, 4, 1024) if asym_spans=True, else (B, 4, 1024).
    """
    # Handle legacy argument order: mean_pool_pairwise(q, c, device, max_len=...)
    if isinstance(max_len, (str, torch.device)):
        real_dev = max_len
        real_len = device if isinstance(device, int) else 128
        device = real_dev
        max_len = real_len
    elif isinstance(device, int):
        max_len = device
        device = None

    m = model if model is not None else default_model
    tok = tokenizer if tokenizer is not None else default_tokenizer
    dev = device if device is not None else next(m.parameters()).device

    b = len(questions)
    ceiling = getattr(m.config, "max_position_embeddings", 8192)
    if max_len > ceiling:
        print(
            f"[encoding warning] Requested max_len={max_len} exceeds architectural ceiling ({ceiling}); clamped to {ceiling}.",
            file=sys.stderr,
        )
        max_len = ceiling

    if task_prefixes:
        flat_texts = [
            f"{'[Task: Scientific Facts] ' if (sources and sources[idx] == 'openbookqa') else '[Task: Causal Reasoning] '}Question: {questions[idx]} Answer: {choices_list[idx][c]}"
            for idx in range(b)
            for c in range(4)
        ]
    else:
        flat_texts = [
            f"Question: {questions[idx]} Answer: {choices_list[idx][c]}"
            for idx in range(b)
            for c in range(4)
        ]

    inputs = tok(flat_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(dev)

    if with_grad:
        last_hidden = m(**inputs).last_hidden_state
    else:
        with torch.no_grad():
            last_hidden = m(**inputs).last_hidden_state

    if asym_spans:
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        B, T = input_ids.shape

        # ModernBERT token ID 37741 = ' Answer', token ID 27 = ':'
        match = (input_ids[:, :-1] == 37741) & (input_ids[:, 1:] == 27)
        has_match = match.any(dim=1)
        ans_pos = match.int().argmax(dim=1)
        cols = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        valid_lens = attn_mask.sum(dim=1) - 1

        q_mask = (cols < ans_pos.unsqueeze(1)) & (attn_mask.bool())
        a_mask = (cols >= (ans_pos + 2).unsqueeze(1)) & (cols < valid_lens.unsqueeze(1)) & (attn_mask.bool())

        no_match = ~has_match.unsqueeze(1)
        q_mask = torch.where(no_match, attn_mask.bool(), q_mask)
        a_mask = torch.where(no_match, attn_mask.bool(), a_mask)

        q_mask_f = q_mask.unsqueeze(-1).float()
        a_mask_f = a_mask.unsqueeze(-1).float()
        q_vec = (last_hidden * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)

        mask = inputs["attention_mask"].unsqueeze(-1)
        holistic = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        if asym_span_mode == "causal_attn":
            H_norm = F.normalize(last_hidden, p=2, dim=-1)
            q_norm = F.normalize(q_vec, p=2, dim=-1)
            cos_sim = (H_norm * q_norm.unsqueeze(1)).sum(dim=-1)
            scores = (cos_sim / 0.1).masked_fill(~a_mask, -1e9)
            attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
            a_attn = (last_hidden * attn_weights).sum(dim=1)

            v0 = holistic
            v1 = 0.35 * q_vec + 0.65 * a_vec
            v2 = 0.15 * q_vec + 0.85 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = 0.35 * q_vec + 0.65 * a_attn
        elif asym_span_mode in ("choice_forward", "default"):
            v0 = holistic
            v1 = 0.40 * q_vec + 0.60 * a_vec
            v2 = 0.20 * q_vec + 0.80 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = 0.05 * q_vec + 0.95 * a_vec
        elif asym_span_mode == "choice_forward_v2":
            v0 = holistic
            v1 = 0.40 * q_vec + 0.60 * a_vec
            v2 = 0.20 * q_vec + 0.80 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = 0.15 * q_vec + 0.85 * a_vec
        elif asym_span_mode == "balanced_spectrum":
            v0 = holistic
            v1 = 0.40 * q_vec + 0.60 * a_vec
            v2 = 0.20 * q_vec + 0.80 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = 0.35 * q_vec + 0.65 * a_vec
        elif asym_span_mode == "constraint_spectrum":
            v0 = holistic
            v1 = 0.70 * q_vec + 0.30 * a_vec
            v2 = 0.20 * q_vec + 0.80 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = 0.35 * q_vec + 0.65 * a_vec
        else:
            # 5 asymmetric ThoughtHead perspectives (fallback):
            v0 = holistic
            v1 = 0.75 * q_vec + 0.25 * a_vec
            v2 = 0.25 * q_vec + 0.75 * a_vec
            v3 = 0.50 * q_vec + 0.50 * a_vec
            v4 = holistic

        multi_head_vecs = torch.stack([v0, v1, v2, v3, v4], dim=1)  # (B*4, 5, 1024)
        multi_head_vecs = multi_head_vecs.view(b, 4, 5, -1).permute(0, 2, 1, 3)  # (B, 5, 4, 1024)
        return multi_head_vecs

    if span_pooling:
        input_ids = inputs["input_ids"]
        attn_mask = inputs["attention_mask"]
        B, T = input_ids.shape
        match = (input_ids[:, :-1] == 37741) & (input_ids[:, 1:] == 27)
        has_match = match.any(dim=1)
        ans_pos = match.int().argmax(dim=1)
        cols = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)
        valid_lens = attn_mask.sum(dim=1) - 1

        q_mask = (cols < ans_pos.unsqueeze(1)) & (attn_mask.bool())
        a_mask = (cols >= (ans_pos + 2).unsqueeze(1)) & (cols < valid_lens.unsqueeze(1)) & (attn_mask.bool())

        no_match = ~has_match.unsqueeze(1)
        q_mask = torch.where(no_match, attn_mask.bool(), q_mask)
        a_mask = torch.where(no_match, attn_mask.bool(), a_mask)

        q_mask_f = q_mask.unsqueeze(-1).float()
        a_mask_f = a_mask.unsqueeze(-1).float()
        q_vec = (last_hidden * q_mask_f).sum(dim=1) / q_mask_f.sum(dim=1).clamp(min=1.0)
        a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)

        pooled = (1.0 - span_weight) * q_vec + span_weight * a_vec
        return pooled.view(b, 4, -1)

    # Standard whole-sequence mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.view(b, 4, -1)
