"""encoding.py - Canonical Span Pooling & Text Embedding for System 2.

Deduplicates text and pairwise span encoding logic across train.py, eval.py,
interact.py, and system2_runtime.py into a single, verified, tested module.
"""

from typing import Any, List, Optional, Tuple, Union
import sys
import torch
import torch.nn.functional as F

from bert import get_bert

# Kept as module attributes for anything that inspects them. They stay None: the
# default backbone is resolved lazily through get_bert() so importing this module
# never triggers a ModernBERT load.
default_model = None
default_tokenizer = None


def _resolve_backbone(model, tokenizer):
    """Return (model, tokenizer), filling gaps from the shared bert.get_bert() cache."""
    model = model if model is not None else default_model
    tokenizer = tokenizer if tokenizer is not None else default_tokenizer
    if model is None or tokenizer is None:
        default_tok, default_mod = get_bert()
        # get_bert returns (tokenizer, model); mix-and-match so a caller passing
        # only one of the two keeps theirs.
        model = model if model is not None else default_mod
        tokenizer = tokenizer if tokenizer is not None else default_tok
    return model, tokenizer


# ModernBERT subword token IDs for modal negation, exceptions, and boundary operators
MODAL_NEGATION_IDS = {
    417, 2550, 1620, 3707, 6747, 4543, 3221, 5803, 20753, 10311, 48231, 26194, 7479,
    13978, 14856, 17290, 8931, 38166
}


def _compute_qa_masks(
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split each sequence into Question and Answer spans at the LAST ' Answer:' delimiter."""
    attn_bool = attn_mask.bool()
    batch_size, seq_len = input_ids.shape
    if seq_len < 2:
        # Guard against T < 2 where input_ids[:, :-1] has width 0 and argmax would fail.
        return attn_bool, attn_bool

    # ModernBERT token ID 37741 = ' Answer', token ID 27 = ':'
    match = (input_ids[:, :-1] == 37741) & (input_ids[:, 1:] == 27)
    has_match = match.any(dim=1)
    # Flip along sequence dim so argmax selects the FINAL ' Answer:' delimiter rather than
    # splitting prematurely if the question text itself contains ' Answer:'.
    rev_pos = match.flip(dims=[1]).int().argmax(dim=1)
    ans_pos = (match.size(1) - 1) - rev_pos

    cols = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
    valid_lens = attn_mask.sum(dim=1) - 1

    q_mask = (cols < ans_pos.unsqueeze(1)) & attn_bool
    a_mask = (cols >= (ans_pos + 2).unsqueeze(1)) & (cols < valid_lens.unsqueeze(1)) & attn_bool

    no_match = ~has_match.unsqueeze(1)
    q_mask = torch.where(no_match, attn_bool, q_mask)
    a_mask = torch.where(no_match, attn_bool, a_mask)
    return q_mask, a_mask


def _pool_qa_spans(
    last_hidden: torch.Tensor,
    input_ids: torch.Tensor,
    attn_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pool Question tokens (with modal salience boost) and Answer tokens using _compute_qa_masks."""
    q_mask, a_mask = _compute_qa_masks(input_ids, attn_mask)
    # Modal salience weighting: boost negation and modal boundary tokens so polarity is not washed out
    neg_tensor = torch.tensor(list(MODAL_NEGATION_IDS), device=input_ids.device)
    is_modal = torch.isin(input_ids, neg_tensor) & q_mask
    q_weights = torch.where(is_modal, 3.0, 1.0) * q_mask.float()
    q_weights_f = q_weights.unsqueeze(-1)
    q_vec = (last_hidden * q_weights_f).sum(dim=1) / q_weights_f.sum(dim=1).clamp(min=1.0)

    a_mask_f = a_mask.unsqueeze(-1).float()
    a_vec = (last_hidden * a_mask_f).sum(dim=1) / a_mask_f.sum(dim=1).clamp(min=1.0)
    return q_vec, a_vec, a_mask


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
    m, tok = _resolve_backbone(model, tokenizer)
    # Run inputs on the backbone's device (it may have been moved to MPS/CUDA by
    # System2Engine) and move the pooled output to the caller's requested device.
    model_dev = next(m.parameters()).device
    out_dev = torch.device(device) if device is not None else model_dev

    ceiling = getattr(m.config, "max_position_embeddings", 8192)
    if max_len > ceiling:
        print(
            f"[encoding warning] Requested max_len={max_len} exceeds architectural ceiling ({ceiling}); clamped to {ceiling}.",
            file=sys.stderr,
        )
        max_len = ceiling

    inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(model_dev)

    if with_grad:
        last_hidden = m(**inputs).last_hidden_state
    else:
        with torch.no_grad():
            last_hidden = m(**inputs).last_hidden_state

    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled.to(out_dev) if pooled.device != out_dev else pooled


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
    """Pairwise encoding: (B questions, N choices) -> (B, N, 1024) or (B, 5, N, 1024).

    Constructs candidate representations with asymmetric span masking separating Question
    and Answer tokens, projecting into 5 distinct cognitive perspectives.

    Returns:
        Tensor of shape (B, 5, N, 1024) if asym_spans=True, else (B, N, 1024).
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

    if not questions or not choices_list:
        raise ValueError("questions and choices_list must be non-empty.")
    if len(questions) != len(choices_list):
        raise ValueError(
            f"questions ({len(questions)}) and choices_list ({len(choices_list)}) must have the same length."
        )
    num_choices = len(choices_list[0])
    if num_choices == 0:
        raise ValueError("Each row in choices_list must contain at least 1 choice.")
    for row_idx, row in enumerate(choices_list):
        if len(row) != num_choices:
            raise ValueError(
                f"All rows in choices_list must have the same number of choices "
                f"(row 0 has {num_choices}, row {row_idx} has {len(row)})."
            )

    m, tok = _resolve_backbone(model, tokenizer)
    # Run inputs on the backbone's device and move the pooled output to the
    # caller's requested device so CPU callers don't fail on an MPS-resident backbone.
    dev = next(m.parameters()).device
    out_dev = torch.device(device) if device is not None else dev

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
            for c in range(num_choices)
        ]
    else:
        flat_texts = [
            f"Question: {questions[idx]} Answer: {choices_list[idx][c]}"
            for idx in range(b)
            for c in range(num_choices)
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
        q_vec, a_vec, a_mask = _pool_qa_spans(last_hidden, input_ids, attn_mask)

        mask = attn_mask.unsqueeze(-1)
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

        multi_head_vecs = torch.stack([v0, v1, v2, v3, v4], dim=1)  # (B*N, 5, 1024)
        multi_head_vecs = multi_head_vecs.view(b, num_choices, 5, -1).permute(0, 2, 1, 3)  # (B, 5, N, 1024)
        return multi_head_vecs.to(out_dev) if multi_head_vecs.device != out_dev else multi_head_vecs

    if span_pooling:
        q_vec, a_vec, _ = _pool_qa_spans(last_hidden, inputs["input_ids"], inputs["attention_mask"])
        pooled = (1.0 - span_weight) * q_vec + span_weight * a_vec
        out = pooled.view(b, num_choices, -1)
        return out.to(out_dev) if out.device != out_dev else out

    # Standard whole-sequence mean pooling
    mask = inputs["attention_mask"].unsqueeze(-1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    out = pooled.view(b, num_choices, -1)
    return out.to(out_dev) if out.device != out_dev else out
