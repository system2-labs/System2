import torch
import torch.nn as nn
from bert import tokenizer, model


class ThoughtHead(nn.Module):
    """One hypothesis + calibrated confidence.

    Input:  (batch_size, 1024) - ModernBERT mean pooled output
    Output: (hypo_token, confidence)  - hypothesis token index + calibrated confidence
    Params: ~40M  (4-bit: ~160MB)
    """

    def __init__(self, hidden_size=1024, head_dim=512, vocab_size=30522, num_classes=None, diff_features=False):
        super().__init__()
        self.diff_features = diff_features

        # 1024 -> 512 projection
        self.input_proj = nn.Linear(hidden_size, head_dim)

        # 2 transformer encoder layers (512 dim, 8 heads)
        layer = nn.TransformerEncoderLayer(
            d_model=head_dim,
            nhead=8,
            dim_feedforward=2048,
            activation="gelu",
            batch_first=True,
            dropout=0.0,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=2)

        # Confidence machine: 512 -> 64 -> 1 (logit)
        self.confidence_head = nn.Sequential(
            nn.Linear(head_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Hypothesis machine:
        if diff_features:
            # Contrastive scoring head: takes [x, x - mean(x)] (512 + 512 = 1024)
            self.hypothesis_head = nn.Sequential(
                nn.Linear(head_dim * 2, 256),
                nn.GELU(),
                nn.Linear(256, 1),
            )
        else:
            self.hypothesis_head = nn.Linear(head_dim, num_classes or vocab_size)

    def forward(self, encoder_output):
        """encoder_output: (batch_size, 1024)
        Returns: (hypo_token: tensor, confidence: tensor)
        """
        x = self.input_proj(encoder_output)  # (B, 512)
        x = x.unsqueeze(1)                   # (B, 1, 512)
        x = self.layers(x).squeeze(1)       # (B, 512)

        # Confidence (0-1 via sigmoid)
        logit = self.confidence_head(x).squeeze(-1)  # (B,)
        confidence = torch.sigmoid(logit)             # (B,)

        # Hypothesis: top token
        logits = self.hypothesis_head(x)              # (B, vocab_size)
        hypo_token=logits.argmax(dim=-1)

        return hypo_token, confidence


# --- demo ---
if __name__ == "__main__":
    inputs = tokenizer("Is this ticket about billing?", return_tensors="pt")
    last_hidden = model(**inputs).last_hidden_state
    mask = inputs["attention_mask"].unsqueeze(-1)
    mean_pool = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)

    head = ThoughtHead(vocab_size=30522)
    token, conf = head(mean_pool)

    print(f"Token: {token.item()}")
    print(f"Confidence: {conf.item():.3f}")
