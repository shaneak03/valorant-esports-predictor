"""
FFNN binary classification head.

Input:  concat([team_a_repr, team_b_repr]) -> (B, 2 * d_model)
Output: (B, 1) raw logit (apply sigmoid at inference, BCEWithLogitsLoss at train)
"""

import torch
import torch.nn as nn


class ClassifierHead(nn.Module):
    def __init__(
        self,
        in_dim: int = 128,       # 2 * d_model
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, in_dim) -> (B, 1)
        return self.net(x)
