"""
Top-level Valorant match predictor model.

Data flow:
  team_a_scalars (B,20,11) ─┐
  team_a_map_idx (B,20)     ├─ MatchEncoder ──┐
  team_a_meta_idx (B,20)    │                  │
  team_a_pad_mask (B,20)    ┘                  ├─ TeamEncoder (shared) ──┐
                                               │                         ├─ concat ─ ClassifierHead ─ logit
  team_b_scalars (B,20,11) ─┐                 │                         │
  team_b_map_idx (B,20)     ├─ MatchEncoder ──┘                         │
  team_b_meta_idx (B,20)    │  (shared weights)  TeamEncoder (shared) ──┘
  team_b_pad_mask (B,20)    ┘

Output: (B, 1) raw logit — P(team_a wins) = sigmoid(logit)
"""

import torch
import torch.nn as nn

from .match_encoder import MatchEncoder
from .transformer import TeamEncoder
from .classifier import ClassifierHead


class ValorantPredictor(nn.Module):
    def __init__(
        self,
        num_scalars: int = 11,
        num_maps: int = 12,
        map_embed_dim: int = 16,
        d_model: int = 64,
        seq_len: int = 20,
        num_heads: int = 4,
        dim_feedforward: int = 256,
        num_layers: int = 3,
        encoder_dropout: float = 0.1,
        classifier_dropout: float = 0.2,
        classifier_hidden: list[int] | None = None,
        num_metas: int = 37,
    ):
        super().__init__()
        if classifier_hidden is None:
            classifier_hidden = [64, 32]

        # Shared encoder weights for both teams
        self.match_encoder = MatchEncoder(
            num_scalars=num_scalars,
            num_maps=num_maps,
            map_embed_dim=map_embed_dim,
            d_model=d_model,
            seq_len=seq_len,
            dropout=encoder_dropout,
            num_metas=num_metas,
        )
        self.team_encoder = TeamEncoder(
            d_model=d_model,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers,
            dropout=encoder_dropout,
        )
        # +2 for (elo_diff, elo_sum) appended after the transformer representations
        self.head = ClassifierHead(
            in_dim=d_model * 2 + 2,
            hidden_dims=classifier_hidden,
            dropout=classifier_dropout,
        )

    def encode_team(
        self,
        scalars: torch.Tensor,    # (B, seq_len, num_scalars)
        map_idx: torch.Tensor,    # (B, seq_len)
        pad_mask: torch.Tensor,   # (B, seq_len)
        meta_idx: torch.Tensor,   # (B, seq_len)
    ) -> torch.Tensor:            # (B, d_model)
        tokens = self.match_encoder(scalars, map_idx, meta_idx)
        return self.team_encoder(tokens, pad_mask)

    def forward(
        self,
        scalars_a: torch.Tensor,
        map_idx_a: torch.Tensor,
        pad_mask_a: torch.Tensor,
        scalars_b: torch.Tensor,
        map_idx_b: torch.Tensor,
        pad_mask_b: torch.Tensor,
        meta_idx_a: torch.Tensor,
        meta_idx_b: torch.Tensor,
        elo_a: torch.Tensor,      # (B,) normalised Elo for team_a
        elo_b: torch.Tensor,      # (B,) normalised Elo for team_b
    ) -> torch.Tensor:            # (B, 1)
        repr_a = self.encode_team(scalars_a, map_idx_a, pad_mask_a, meta_idx_a)
        repr_b = self.encode_team(scalars_b, map_idx_b, pad_mask_b, meta_idx_b)

        # Transformer representations: symmetric combination
        combined = torch.cat([repr_a - repr_b, repr_a + repr_b], dim=-1)  # (B, 2*d_model)

        # Append current Elo signals: difference (who is stronger) + sum (match quality)
        elo_diff = (elo_a - elo_b).unsqueeze(-1)   # (B, 1)
        elo_sum  = (elo_a + elo_b).unsqueeze(-1)   # (B, 1)
        combined = torch.cat([combined, elo_diff, elo_sum], dim=-1)  # (B, 2*d_model+2)

        return self.head(combined)                                     # (B, 1)

    def predict_proba(self, *args, **kwargs) -> torch.Tensor:
        """Return P(team_a wins) as a probability in [0, 1]."""
        with torch.no_grad():
            logit = self.forward(*args, **kwargs)
            return torch.sigmoid(logit)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
