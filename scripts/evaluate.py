"""
CLI: Evaluate the trained model on test set.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --device cuda
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MatchDataset
from src.models.predictor import ValorantPredictor
from src.training.metrics import compute_all, brier_score
from src.evaluation.calibration import TemperatureScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_samples(path: str) -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    parser.add_argument("--temperature", default="checkpoints/temperature.json")
    parser.add_argument("--model-config", default="configs/model_config.yaml")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device(args.device)
    log.info("Using device: %s", device)

    # Load model config
    model_cfg = load_config(args.model_config)

    # Build model
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

    # Load checkpoint
    if Path(args.checkpoint).exists():
        log.info("Loading checkpoint from %s", args.checkpoint)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        log.warning("Checkpoint not found at %s — using untrained model", args.checkpoint)

    model.to(device)
    model.eval()

    # Load test set
    log.info("Loading test data...")
    data_dir = Path(args.data_dir)
    test_samples = load_samples(data_dir / "test_samples.pkl")
    test_ds = MatchDataset(test_samples)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    log.info("Test set: %d samples", len(test_ds))

    # Run inference
    log.info("Running inference on test set...")
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            logits = model(
                batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
                batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
                batch["meta_idx_a"], batch["meta_idx_b"],
            ).squeeze(-1)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["label"].cpu().numpy())

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)

    # Load temperature if available
    T = 1.0
    if Path(args.temperature).exists():
        try:
            with open(args.temperature) as f:
                data = json.load(f)
                T = data.get("temperature", 1.0)
            log.info("Loaded temperature scaling: T=%.4f", T)
        except Exception as e:
            log.warning("Failed to load temperature: %s", e)

    # Apply temperature scaling and compute metrics
    calibrated_logits = all_logits / T
    probs = 1 / (1 + np.exp(-calibrated_logits))  # sigmoid
    metrics = compute_all(probs, all_labels)

    # Print results
    log.info("=" * 60)
    log.info("TEST SET RESULTS (T=%.4f)", T)
    log.info("=" * 60)
    log.info("Accuracy:  %.4f", metrics["accuracy"])
    log.info("ROC-AUC:   %.4f", metrics["roc_auc"])
    log.info("Brier:     %.4f", metrics["brier_score"])
    log.info("Log-loss:  %.4f", metrics["log_loss"])
    log.info("ECE:       %.4f", metrics["ece"])
    log.info("=" * 60)

    log.info("\nTargets:")
    log.info("  Accuracy:  > 0.60 (random: 0.50)")
    log.info("  ROC-AUC:   > 0.65 (random: 0.50)")
    log.info("  Brier:     < 0.22 (random: 0.25)")
    log.info("  Log-loss:  < 0.65 (random: 0.69)")
    log.info("  ECE:       < 0.05")


if __name__ == "__main__":
    main()
