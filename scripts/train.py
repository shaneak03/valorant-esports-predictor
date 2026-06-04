"""
CLI: Train the ValorantPredictor model.

Usage:
    python scripts/train.py
    python scripts/train.py --device cuda --epochs 200

Loads train/val splits from data/processed/*_samples.pkl.
Saves best checkpoint to checkpoints/best_model.pt.
Saves calibrated temperature to checkpoints/temperature.json.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MatchDataset
from src.data.augmentation import TeamSwapDataset
from src.models.predictor import ValorantPredictor
from src.training.trainer import Trainer
from src.training.metrics import compute_all
from src.evaluation.calibration import TemperatureScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_samples(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--train-config", default="configs/training_config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--epochs", type=int, default=None, help="Override max_epochs in config")
    args = parser.parse_args()

    model_cfg = load_config(args.model_config)
    train_cfg = load_config(args.train_config)
    if args.epochs:
        train_cfg["max_epochs"] = args.epochs

    log.info("Loading data...")
    data_dir = Path(args.data_dir)
    train_samples = load_samples(data_dir / "train_samples.pkl")
    val_samples = load_samples(data_dir / "val_samples.pkl")
    log.info("Train: %d samples, Val: %d samples", len(train_samples), len(val_samples))

    train_ds = MatchDataset(train_samples)
    if train_cfg.get("augment_team_swap", True):
        train_ds = TeamSwapDataset(train_ds)
        log.info("Team-swap augmentation applied: %d samples total", len(train_ds))

    val_ds = MatchDataset(val_samples)

    batch_size = train_cfg.get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    log.info("Building model...")
    model = ValorantPredictor(
        num_scalars=model_cfg.get("num_scalar_features", 11),
        num_maps=model_cfg.get("num_maps", 12),
        map_embed_dim=model_cfg.get("map_embedding_dim", 16),
        d_model=model_cfg.get("d_model", 64),
        seq_len=model_cfg.get("seq_len", 20),
        num_heads=model_cfg.get("num_heads", 4),
        dim_feedforward=model_cfg.get("dim_feedforward", 256),
        num_layers=model_cfg.get("num_layers", 3),
        encoder_dropout=model_cfg.get("dropout", 0.1),
        classifier_dropout=model_cfg.get("ffnn_dropout", 0.2),
        classifier_hidden=model_cfg.get("ffnn_dims", [64, 32]),
        num_metas=model_cfg.get("num_metas", 37),
    )
    log.info("Model parameters: %d", model.num_parameters)

    trainer = Trainer(model, train_cfg, device=args.device)
    trainer.fit(train_loader, val_loader)
    trainer.load_best()

    # --- Temperature calibration on val set ---
    log.info("Calibrating temperature scaling on validation set...")
    model.eval()
    device = torch.device(args.device)
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(
                batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
                batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
                batch["meta_idx_a"], batch["meta_idx_b"],
                batch["elo_a"], batch["elo_b"],
            ).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    scaler = TemperatureScaler()
    T = scaler.calibrate(all_logits, all_labels)
    scaler.save("checkpoints/temperature.json")

    cal_probs = torch.sigmoid(torch.tensor(all_logits) / T).numpy()
    metrics = compute_all(cal_probs, all_labels)
    log.info("Val metrics (calibrated): %s", {k: f"{v:.4f}" for k, v in metrics.items()})
    log.info("Training complete.")


if __name__ == "__main__":
    main()
