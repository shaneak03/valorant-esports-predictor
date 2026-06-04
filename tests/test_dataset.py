"""Tests for MatchDataset and TeamSwapDataset."""

import numpy as np
import torch
import pytest

from src.data.dataset import MatchDataset, SEQ_LEN, NUM_SCALARS
from src.data.augmentation import TeamSwapDataset


def _make_history(n: int) -> list[dict]:
    return [
        {"scalars": np.random.randn(NUM_SCALARS).astype(np.float32), "map_idx": i % 12}
        for i in range(n)
    ]


def _make_samples(n: int = 5) -> list[dict]:
    samples = []
    for i in range(n):
        samples.append({
            "match_id": i,
            "date": f"2024-0{i+1}-01",
            "team_a": f"TeamA{i}",
            "team_b": f"TeamB{i}",
            "winner": i % 2,
            "history_a": _make_history(15),  # less than SEQ_LEN -> should pad
            "history_b": _make_history(20),
        })
    return samples


def test_dataset_shapes():
    ds = MatchDataset(_make_samples())
    sample = ds[0]
    assert sample["scalars_a"].shape == (SEQ_LEN, NUM_SCALARS)
    assert sample["map_idx_a"].shape == (SEQ_LEN,)
    assert sample["pad_mask_a"].shape == (SEQ_LEN,)
    assert sample["label"].dtype == torch.float32


def test_padding_applied_when_short():
    samples = _make_samples(1)
    samples[0]["history_a"] = _make_history(5)
    ds = MatchDataset(samples)
    sample = ds[0]
    # First 15 positions should be padded (SEQ_LEN - 5 = 15)
    assert sample["pad_mask_a"][:15].all()
    assert not sample["pad_mask_a"][15:].any()


def test_no_padding_when_full():
    samples = _make_samples(1)
    samples[0]["history_a"] = _make_history(SEQ_LEN)
    ds = MatchDataset(samples)
    sample = ds[0]
    assert not sample["pad_mask_a"].any()


def test_team_swap_doubles_length():
    base = MatchDataset(_make_samples(4))
    aug = TeamSwapDataset(base)
    assert len(aug) == len(base) * 2


def test_team_swap_flips_label():
    samples = _make_samples(1)
    samples[0]["winner"] = 0
    base = MatchDataset(samples)
    aug = TeamSwapDataset(base)
    original = aug[0]
    swapped = aug[1]
    assert abs(original["label"].item() + swapped["label"].item() - 1.0) < 1e-6
