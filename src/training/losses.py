"""
Loss functions for binary match prediction.

Default: BCEWithLogitsLoss with label smoothing.
Label smoothing epsilon=0.1 discourages overconfident predictions and
improves calibration — critical for probabilistic sports forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothedBCELoss(nn.Module):
    """BCEWithLogitsLoss + label smoothing."""

    def __init__(self, epsilon: float = 0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, 1) or (B,), targets: (B,)
        logits = logits.squeeze(-1)
        smooth_targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)


class FocalLoss(nn.Module):
    """Focal loss — down-weights easy predictions, focuses on hard examples."""

    def __init__(self, gamma: float = 2.0, epsilon: float = 0.0):
        super().__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.squeeze(-1)
        if self.epsilon > 0:
            targets = targets * (1 - self.epsilon) + 0.5 * self.epsilon
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.exp(-bce)
        return ((1 - p_t) ** self.gamma * bce).mean()


def build_loss(loss_name: str = "bce_with_logits", label_smoothing: float = 0.1) -> nn.Module:
    if loss_name == "bce_with_logits":
        return SmoothedBCELoss(epsilon=label_smoothing)
    if loss_name == "focal":
        return FocalLoss(epsilon=label_smoothing)
    raise ValueError(f"Unknown loss: {loss_name}")
