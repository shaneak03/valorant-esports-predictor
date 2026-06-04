"""
Evaluation metrics for probabilistic match prediction.

Accuracy alone is a poor metric for sports forecasting — use all of:
  - Accuracy        (threshold at 0.5)
  - ROC-AUC         (ranking quality, threshold-free)
  - Brier Score     (mean squared probability error, lower=better)
  - Log-loss        (penalizes confident wrong predictions)
  - ECE             (calibration: does 70% confidence -> 70% wins?)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, log_loss


def accuracy(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    return float((preds == labels).mean())


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def roc_auc(probs: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def logloss(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(log_loss(labels, probs))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() * abs(bin_conf - bin_acc)
    return float(ece / len(probs))


def compute_all(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy(probs, labels),
        "roc_auc": roc_auc(probs, labels),
        "brier_score": brier_score(probs, labels),
        "log_loss": logloss(probs, labels),
        "ece": expected_calibration_error(probs, labels),
    }
