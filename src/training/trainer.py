"""
Training loop with early stopping on validation Brier score.

Usage:
    trainer = Trainer(model, config, device)
    trainer.fit(train_loader, val_loader)
    trainer.load_best()
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .losses import build_loss
from .metrics import brier_score, compute_all

log = logging.getLogger(__name__)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.triggered = False

    def step(self, metric: float) -> bool:
        if metric < self.best - self.min_delta:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class Trainer:
    def __init__(self, model: nn.Module, config: dict, device: torch.device | str = "cpu"):
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

        self.criterion = build_loss(
            config.get("loss", "bce_with_logits"),
            config.get("label_smoothing", 0.1),
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 1e-2),
        )

        self.max_epochs = config.get("max_epochs", 200)
        checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path = checkpoint_dir / "best_model.pt"

        self._early_stop = EarlyStopping(patience=config.get("early_stopping_patience", 20))
        self.history: list[dict] = []

    def _make_scheduler(self):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_epochs,
            eta_min=1e-5,
        )

    # ------------------------------------------------------------------
    # Single epoch helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = _batch_to_device(batch, self.device)
            self.optimizer.zero_grad()
            logits = self.model(
                batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
                batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
                batch["meta_idx_a"], batch["meta_idx_b"],
                batch["elo_a"], batch["elo_b"],
            )
            loss = self.criterion(logits, batch["label"])
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict]:
        self.model.eval()
        all_probs, all_labels = [], []
        total_loss = 0.0
        for batch in loader:
            batch = _batch_to_device(batch, self.device)
            logits = self.model(
                batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
                batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
                batch["meta_idx_a"], batch["meta_idx_b"],
                batch["elo_a"], batch["elo_b"],
            )
            loss = self.criterion(logits, batch["label"])
            total_loss += loss.item() * len(batch["label"])
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            labels = batch["label"].cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)

        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        avg_loss = total_loss / len(loader.dataset)
        metrics = compute_all(probs, labels)
        metrics["loss"] = avg_loss
        return brier_score(probs, labels), metrics

    # ------------------------------------------------------------------
    # Main fit loop
    # ------------------------------------------------------------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        scheduler = self._make_scheduler()

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self._train_epoch(train_loader)
            val_brier, val_metrics = self._eval_epoch(val_loader)

            scheduler.step()

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.history.append(row)

            log.info(
                "Epoch %3d | train_loss=%.4f | val_brier=%.4f | val_acc=%.3f | val_auc=%.3f",
                epoch, train_loss, val_brier,
                val_metrics["accuracy"], val_metrics["roc_auc"],
            )

            if val_brier <= self._early_stop.best:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                log.info("  -> Saved best checkpoint (brier=%.4f)", val_brier)

            if self._early_stop.step(val_brier):
                log.info("Early stopping at epoch %d (patience=%d)", epoch, self._early_stop.patience)
                break

        return self.history

    def load_best(self) -> None:
        self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        log.info("Loaded best checkpoint from %s", self.checkpoint_path)
