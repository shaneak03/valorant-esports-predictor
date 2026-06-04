"""
CLI: Predict the winner of an upcoming Valorant series.

Usage:
    python scripts/predict.py --team-a "Team Vitality" --team-b "Sentinels"
    python scripts/predict.py --team-a "Fnatic" --team-b "NRG" --date 2025-06-01

Looks up each team's recent matches from the processed samples,
builds 20-match history sequences, and runs inference with calibrated temperature.

Output: P(team_a wins) as a probability.
"""

import argparse
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MatchDataset
from src.data.normalizer import Normalizer
from src.models.predictor import ValorantPredictor
from src.evaluation.calibration import TemperatureScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--team-a", required=True)
    parser.add_argument("--team-b", required=True)
    parser.add_argument("--date", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--temperature", default="checkpoints/temperature.json")
    parser.add_argument("--scaler", default="data/processed/scaler_params.json")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    device = torch.device(args.device)

    # --- Load all processed samples to build team histories ---
    data_dir = Path(args.data_dir)
    all_samples = []
    for split in ("train", "val", "test"):
        pkl = data_dir / f"{split}_samples.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                all_samples.extend(pickle.load(f))

    # Build team -> list of past matches with features
    team_history: dict[str, list] = defaultdict(list)
    for s in sorted(all_samples, key=lambda x: x["date"]):
        if s["date"] >= args.date:
            continue
        if s["history_a"]:
            team_history[s["team_a"]].extend(s["history_a"])
        if s["history_b"]:
            team_history[s["team_b"]].extend(s["history_b"])

    hist_a = team_history.get(args.team_a, [])
    hist_b = team_history.get(args.team_b, [])

    if not hist_a:
        log.warning("No history found for '%s'. Using empty history (heavy padding).", args.team_a)
    if not hist_b:
        log.warning("No history found for '%s'. Using empty history (heavy padding).", args.team_b)

    # Build a single sample
    sample = {
        "match_id": -1,
        "date": args.date,
        "team_a": args.team_a,
        "team_b": args.team_b,
        "winner": 0,  # dummy
        "history_a": hist_a[-20:],
        "history_b": hist_b[-20:],
    }

    ds = MatchDataset([sample])
    batch = ds[0]
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].unsqueeze(0).to(device)

    # --- Load model ---
    model = ValorantPredictor(
        num_scalars=model_cfg.get("num_scalar_features", 11),
        num_maps=model_cfg.get("num_maps", 12),
        map_embed_dim=model_cfg.get("map_embedding_dim", 16),
        d_model=model_cfg.get("d_model", 64),
        seq_len=model_cfg.get("seq_len", 20),
        num_heads=model_cfg.get("num_heads", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        num_layers=model_cfg.get("num_layers", 3),
        num_metas=model_cfg.get("num_metas", 37),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    model.eval()

    # --- Temperature scaler ---
    temp_scaler = TemperatureScaler.load(args.temperature)
    T = float(temp_scaler.temperature.item())

    # --- Inference ---
    # Run both orderings (A vs B) and (B vs A) and average for a symmetric result.
    # This guarantees P(A beats B) == 1 - P(B beats A) regardless of model bias.
    with torch.no_grad():
        logit_ab = model(
            batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
            batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
            batch["meta_idx_a"], batch["meta_idx_b"],
        ).squeeze()
        logit_ba = model(
            batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
            batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
            batch["meta_idx_b"], batch["meta_idx_a"],
        ).squeeze()
        prob_a_fwd = torch.sigmoid(logit_ab / T)
        prob_a_rev = 1.0 - torch.sigmoid(logit_ba / T)  # P(A wins) = 1 - P(B wins)
        prob_a = ((prob_a_fwd + prob_a_rev) / 2).item()

    print(f"\n  {args.team_a} vs {args.team_b}  [{args.date}]")
    print(f"  P({args.team_a} wins) = {prob_a:.3f}")
    print(f"  P({args.team_b} wins) = {1 - prob_a:.3f}")
    print(f"  Prediction: {'→ ' + args.team_a if prob_a > 0.5 else '→ ' + args.team_b}\n")


if __name__ == "__main__":
    main()
