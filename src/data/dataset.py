"""
PyTorch Dataset for Valorant match prediction.

Each sample represents one series (target match) and contains:
  - scalars_a / scalars_b  : (20, 11) float32 — normalized scalar features
  - map_idx_a / map_idx_b  : (20,)    int64   — map indices for embedding lookup
  - meta_idx_a / meta_idx_b: (20,)    int64   — meta period IDs (36 = unknown/padded)
  - pad_mask_a / pad_mask_b: (20,)    bool    — True where position is padded
  - label                  : float scalar — 1.0 if team_a wins, 0.0 otherwise

History windows are built from `team_histories` which is a dict:
  team_name -> list of match dicts sorted chronologically (oldest first).

Each history list contains match dicts with pre-extracted features:
  {"scalars": np.ndarray (11,), "map_idx": int, "meta_id": int, "date": str}
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

SEQ_LEN = 20
UNKNOWN_META_ID = 36

# Import from feature_extractor so this stays in sync automatically
from src.data.feature_extractor import NUM_SCALAR_FEATURES as NUM_SCALARS


class MatchDataset(Dataset):
    """
    samples: list of dicts with keys:
        match_id, date, team_a, team_b, winner (0 or 1),
        history_a (list of feature dicts), history_b (list of feature dicts)

    Each feature dict in history_*:
        {"scalars": np.ndarray (11,), "map_idx": int, "meta_id": int}
    """

    def __init__(self, samples: list[dict]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        scalars_a, map_idx_a, meta_idx_a, mask_a = self._build_sequence(s["history_a"])
        scalars_b, map_idx_b, meta_idx_b, mask_b = self._build_sequence(s["history_b"])
        label = 1.0 if s["winner"] == 0 else 0.0  # 1 = team_a wins

        return {
            "scalars_a": torch.tensor(scalars_a, dtype=torch.float32),
            "map_idx_a": torch.tensor(map_idx_a, dtype=torch.long),
            "meta_idx_a": torch.tensor(meta_idx_a, dtype=torch.long),
            "pad_mask_a": torch.tensor(mask_a, dtype=torch.bool),
            "scalars_b": torch.tensor(scalars_b, dtype=torch.float32),
            "map_idx_b": torch.tensor(map_idx_b, dtype=torch.long),
            "meta_idx_b": torch.tensor(meta_idx_b, dtype=torch.long),
            "pad_mask_b": torch.tensor(mask_b, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.float32),
        }

    @staticmethod
    def _build_sequence(history: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Take the last SEQ_LEN matches, left-pad with zeros if fewer.

        Returns:
          scalars  (SEQ_LEN, NUM_SCALARS) float32
          map_idx  (SEQ_LEN,)             int64
          meta_idx (SEQ_LEN,)             int64  — UNKNOWN_META_ID for padded positions
          pad_mask (SEQ_LEN,)             bool — True where padded
        """
        recent = history[-SEQ_LEN:]  # keep most recent up to SEQ_LEN
        n = len(recent)
        pad_len = SEQ_LEN - n

        scalars = np.zeros((SEQ_LEN, NUM_SCALARS), dtype=np.float32)
        map_idx = np.zeros(SEQ_LEN, dtype=np.int64)
        meta_idx = np.full(SEQ_LEN, UNKNOWN_META_ID, dtype=np.int64)
        pad_mask = np.ones(SEQ_LEN, dtype=bool)  # True = padded

        for i, entry in enumerate(recent):
            scalars[pad_len + i] = entry["scalars"]
            map_idx[pad_len + i] = entry["map_idx"]
            meta_idx[pad_len + i] = entry.get("meta_id", UNKNOWN_META_ID)
            pad_mask[pad_len + i] = False  # real token

        return scalars, map_idx, meta_idx, pad_mask
