"""Shape assertions for all tensor transformations in the model."""

import torch
import pytest

from src.models.match_encoder import MatchEncoder
from src.models.transformer import TeamEncoder
from src.models.classifier import ClassifierHead
from src.models.predictor import ValorantPredictor

B, S, NUM_SCALARS, D = 4, 20, 11, 64


def make_batch():
    scalars = torch.randn(B, S, NUM_SCALARS)
    map_idx = torch.randint(0, 12, (B, S))
    meta_idx = torch.randint(0, 37, (B, S))
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    pad_mask[:, :5] = True  # first 5 positions padded
    return scalars, map_idx, meta_idx, pad_mask


def test_match_encoder_shape():
    enc = MatchEncoder(num_scalars=NUM_SCALARS, d_model=D, seq_len=S)
    scalars, map_idx, meta_idx, _ = make_batch()
    out = enc(scalars, map_idx, meta_idx)
    assert out.shape == (B, S, D)


def test_team_encoder_shape():
    enc = TeamEncoder(d_model=D)
    x = torch.randn(B, S, D)
    pad_mask = torch.zeros(B, S, dtype=torch.bool)
    out = enc(x, pad_mask)
    assert out.shape == (B, D)


def test_classifier_head_shape():
    head = ClassifierHead(in_dim=D * 2)
    x = torch.randn(B, D * 2)
    out = head(x)
    assert out.shape == (B, 1)


def test_full_predictor_shape():
    model = ValorantPredictor()
    scalars, map_idx, meta_idx, pad_mask = make_batch()
    logit = model(scalars, map_idx, pad_mask, scalars, map_idx, pad_mask, meta_idx, meta_idx)
    assert logit.shape == (B, 1)


def test_predict_proba_range():
    model = ValorantPredictor()
    scalars, map_idx, meta_idx, pad_mask = make_batch()
    proba = model.predict_proba(scalars, map_idx, pad_mask, scalars, map_idx, pad_mask, meta_idx, meta_idx)
    assert proba.shape == (B, 1)
    assert (proba >= 0).all() and (proba <= 1).all()


def test_parameter_count():
    model = ValorantPredictor()
    # Model should be reasonable size (~100k-200k params)
    assert 50_000 < model.num_parameters < 500_000, f"Unexpected param count: {model.num_parameters}"
