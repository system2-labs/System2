from contextlib import contextmanager
import threading

import torch
import torch.nn as nn

# Thread-local disable depth per LoRALinear (keyed by id(module)). Kept at module
# level rather than on each LoRALinear instance so modules stay picklable and
# deepcopy-able, and concurrent threads never clobber shared `m.enabled`.
_disable_state = threading.local()


def _get_disabled_depths():
    depths = getattr(_disable_state, "depths", None)
    if depths is None:
        depths = {}
        _disable_state.depths = depths
    return depths


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Freezes original weights and adds low-rank bottleneck:
    W' = W_base + (alpha / r) * (x @ A.T @ B.T)
    """

    def __init__(self, original_linear, r=16, alpha=16, dropout=0.0):
        super().__init__()
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = alpha / r
        dev = original_linear.weight.device
        dtype = original_linear.weight.dtype
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        # A initialized with small Gaussian, B initialized to 0
        self.lora_A = nn.Parameter(torch.randn(r, in_dim, device=dev, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, device=dev, dtype=dtype))
        # Plain attribute (not a buffer/param) so state_dict keys stay identical to
        # existing checkpoints.
        self.enabled = True

    def forward(self, x):
        base = self.original(x)
        depths = getattr(_disable_state, "depths", None)
        if not self.enabled or (depths and depths.get(id(self), 0) > 0):
            # Bypass the delta entirely: exact frozen-base output, no extra matmuls.
            return base
        lora = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


@contextmanager
def disable_lora(model):
    """Temporarily bypass every LoRALinear in `model` on the calling thread.

    Lets a LoRA-adapted shared backbone emit pure frozen-base embeddings (e.g. for
    the domain router's centroids, which were computed without any adapter) without
    allocating a second 395M model. Tracks per-thread nesting depth rather than
    mutating shared `m.enabled`, so overlapping calls across threads never bleed
    disabled state into concurrent forwards or leave layers stuck disabled.
    """
    layers = [m for m in model.modules() if isinstance(m, LoRALinear)]
    depths = _get_disabled_depths()
    layer_ids = [id(m) for m in layers]
    for lid in layer_ids:
        depths[lid] = depths.get(lid, 0) + 1
    try:
        yield model
    finally:
        for lid in layer_ids:
            new_depth = depths.get(lid, 1) - 1
            if new_depth <= 0:
                depths.pop(lid, None)
            else:
                depths[lid] = new_depth


def apply_lora(model, r=16, alpha=16, num_layers=4, train_norms=False, dropout=0.0):
    """Inject LoRA into Wqkv and Wo projections of the top num_layers of ModernBERT.

    All base weights remain strictly frozen. Only lora_A and lora_B are trainable.
    If train_norms=True, also unfreezes layer normalizations (mlp_norm, final_norm)
    for scale calibration without representation drift.
    """
    for p in model.parameters():
        p.requires_grad = False

    lora_params = []
    norm_params = []
    target_layers = model.layers[-num_layers:] if num_layers > 0 else model.layers
    for layer in target_layers:
        if not isinstance(layer.attn.Wqkv, LoRALinear):
            layer.attn.Wqkv = LoRALinear(layer.attn.Wqkv, r=r, alpha=alpha, dropout=dropout)
        else:
            layer.attn.Wqkv.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        if not isinstance(layer.attn.Wo, LoRALinear):
            layer.attn.Wo = LoRALinear(layer.attn.Wo, r=r, alpha=alpha, dropout=dropout)
        else:
            layer.attn.Wo.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        lora_params.extend([
            layer.attn.Wqkv.lora_A, layer.attn.Wqkv.lora_B,
            layer.attn.Wo.lora_A, layer.attn.Wo.lora_B,
        ])
        if train_norms:
            for norm in (layer.attn_norm, layer.mlp_norm):
                for p in norm.parameters():
                    p.requires_grad = True
                    norm_params.append(p)

    if train_norms:
        for p in model.final_norm.parameters():
            p.requires_grad = True
            norm_params.append(p)
        return lora_params, norm_params

    return lora_params
