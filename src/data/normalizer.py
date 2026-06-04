"""
Z-score normalizer for the 11 scalar features.

Fit on the training set only. Saves/loads params from a JSON file so
inference can apply the same normalization without re-fitting.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Normalizer:
    def __init__(self):
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "Normalizer":
        """X shape: (N, num_scalar_features)"""
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # avoid division by zero for constant features
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean is not None, "Call fit() first or load params."
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "mean": self.mean.tolist(),
                "std": self.std.tolist(),
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Normalizer":
        with open(path) as f:
            params = json.load(f)
        norm = cls()
        norm.mean = np.array(params["mean"], dtype=np.float32)
        norm.std = np.array(params["std"], dtype=np.float32)
        return norm
