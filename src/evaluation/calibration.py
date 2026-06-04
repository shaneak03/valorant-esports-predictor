"""
Post-hoc calibration via temperature scaling.

Temperature scaling fits a single scalar T on the validation set such that
the calibrated probability is sigmoid(logit / T). This is the most effective
simple post-hoc calibration method and adds zero parameters to inference.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def calibrate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 500,
    ) -> float:
        """Fit T on (logits, labels) arrays using NLL minimization."""
        logits_t = torch.tensor(logits, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.float32)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        loss_fn = nn.BCEWithLogitsLoss()

        def closure():
            optimizer.zero_grad()
            loss = loss_fn(self.forward(logits_t).squeeze(), labels_t)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(self.temperature.item())
        log.info("Temperature scaling: T=%.4f", T)
        return T

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"temperature": float(self.temperature.item())}, f)

    @classmethod
    def load(cls, path: str) -> "TemperatureScaler":
        with open(path) as f:
            data = json.load(f)
        scaler = cls()
        scaler.temperature = nn.Parameter(torch.tensor([data["temperature"]]))
        return scaler


def reliability_diagram(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: str | None = None,
) -> None:
    bins = np.linspace(0, 1, n_bins + 1)
    bin_mids = (bins[:-1] + bins[1:]) / 2
    bin_acc, bin_conf, bin_count = [], [], []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_acc.append(labels[mask].mean())
        bin_conf.append(probs[mask].mean())
        bin_count.append(mask.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_conf, bin_acc, width=0.08, alpha=0.6, label="Model")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        log.info("Saved reliability diagram to %s", save_path)
    else:
        plt.show()
    plt.close(fig)
