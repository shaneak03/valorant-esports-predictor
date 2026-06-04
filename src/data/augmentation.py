"""
Team-swap augmentation.

For every (seq_a, seq_b, label) sample, also emit (seq_b, seq_a, 1-label).
This doubles the training data and perfectly balances classes regardless of
the underlying win-rate distribution.
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class TeamSwapDataset(Dataset):
    """Wraps another MatchDataset and doubles it with team-swapped samples."""

    def __init__(self, base_dataset: Dataset):
        self._base = base_dataset

    def __len__(self) -> int:
        return len(self._base) * 2

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx // 2
        swapped = (idx % 2 == 1)
        sample = self._base[base_idx]

        if not swapped:
            return sample

        # Swap teams and flip label
        return {
            "scalars_a": sample["scalars_b"],
            "map_idx_a": sample["map_idx_b"],
            "meta_idx_a": sample["meta_idx_b"],
            "pad_mask_a": sample["pad_mask_b"],
            "scalars_b": sample["scalars_a"],
            "map_idx_b": sample["map_idx_a"],
            "meta_idx_b": sample["meta_idx_a"],
            "pad_mask_b": sample["pad_mask_a"],
            "label": 1.0 - sample["label"],
        }
