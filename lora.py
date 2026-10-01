import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear.

    Freezes original weights and adds low-rank bottleneck:
    W' = W_base + (alpha / r) * (x @ A.T @ B.T)
    """

    def __init__(self, original_linear, r=16, alpha=16):
        super().__init__()
        self.original = original_linear
        for p in self.original.parameters():
            p.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.scaling = alpha / r
        dev = original_linear.weight.device
        dtype = original_linear.weight.dtype
        # A initialized with small Gaussian, B initialized to 0
        self.lora_A = nn.Parameter(torch.randn(r, in_dim, device=dev, dtype=dtype) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, device=dev, dtype=dtype))

    def forward(self, x):
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def apply_lora(model, r=16, alpha=16, num_layers=4):
    """Inject LoRA into Wqkv and Wo projections of the top num_layers of ModernBERT.

    All base weights remain strictly frozen. Only lora_A and lora_B are trainable.
    """
    for p in model.parameters():
        p.requires_grad = False

    lora_params = []
    target_layers = model.layers[-num_layers:] if num_layers > 0 else model.layers
    for layer in target_layers:
        if not isinstance(layer.attn.Wqkv, LoRALinear):
            layer.attn.Wqkv = LoRALinear(layer.attn.Wqkv, r=r, alpha=alpha)
        if not isinstance(layer.attn.Wo, LoRALinear):
            layer.attn.Wo = LoRALinear(layer.attn.Wo, r=r, alpha=alpha)
        lora_params.extend([
            layer.attn.Wqkv.lora_A, layer.attn.Wqkv.lora_B,
            layer.attn.Wo.lora_A, layer.attn.Wo.lora_B,
        ])
    return lora_params
