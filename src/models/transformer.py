"""
Shared TransformerEncoder with CLS-token pooling.

The same encoder weights are used for both teams' sequences, enforcing
a symmetric representation space and halving parameter count.

Architecture:
  - Prepend a learned [CLS] token to each 20-token sequence -> (B, 21, d_model)
  - Pass through TransformerEncoder (Pre-LN, 3 layers, 4 heads, ff=256)
  - Extract CLS output at position 0 -> (B, d_model) team representation
"""

import torch
import torch.nn as nn


class TeamEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        # Learned CLS token (1, 1, d_model) — broadcast over batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(
        self,
        x: torch.Tensor,          # (B, seq_len, d_model)
        pad_mask: torch.Tensor,   # (B, seq_len) bool, True = padded
    ) -> torch.Tensor:            # (B, d_model)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)            # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                    # (B, seq_len+1, d_model)

        # Prepend False (not padded) for the CLS position
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        full_mask = torch.cat([cls_mask, pad_mask], dim=1) # (B, seq_len+1)

        out = self.encoder(x, src_key_padding_mask=full_mask)  # (B, seq_len+1, d_model)
        return out[:, 0, :]                                # (B, d_model) — CLS token
