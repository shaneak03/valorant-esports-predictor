"""
MatchEncoder: projects one team's 20-match history into a (batch, seq_len, d_model) tensor.

Per-token pipeline:
  1. Look up map embedding: nn.Embedding(num_maps, map_embed_dim) -> (*, map_embed_dim)
  2. Concat scalar features + map embedding -> (*, raw_feature_dim)
  3. Linear(raw_feature_dim, d_model) + LayerNorm
  4. Add learned positional embedding: nn.Embedding(seq_len, d_model)
  5. Add meta-period embedding: nn.Embedding(num_metas, d_model)

Inputs (batch dims omitted for clarity):
  scalars   (seq_len, num_scalars)  — normalized float32
  map_idx   (seq_len,)              — int64, index into MAP_LIST
  meta_idx  (seq_len,)              — int64, meta period ID (36 = unknown)
  pad_mask  (seq_len,)              — bool, True where padded
"""

import torch
import torch.nn as nn


class MatchEncoder(nn.Module):
    def __init__(
        self,
        num_scalars: int = 11,
        num_maps: int = 12,
        map_embed_dim: int = 16,
        d_model: int = 64,
        seq_len: int = 20,
        dropout: float = 0.1,
        num_metas: int = 37,
    ):
        super().__init__()
        self.seq_len = seq_len
        raw_dim = num_scalars + map_embed_dim

        self.map_embed = nn.Embedding(num_maps, map_embed_dim)
        self.proj = nn.Linear(raw_dim, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.meta_embed = nn.Embedding(num_metas, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        scalars: torch.Tensor,    # (B, seq_len, num_scalars)
        map_idx: torch.Tensor,    # (B, seq_len)
        meta_idx: torch.Tensor,   # (B, seq_len)
    ) -> torch.Tensor:            # (B, seq_len, d_model)
        map_emb = self.map_embed(map_idx)                    # (B, S, map_embed_dim)
        x = torch.cat([scalars, map_emb], dim=-1)            # (B, S, raw_dim)
        x = self.norm(self.proj(x))                          # (B, S, d_model)
        positions = torch.arange(self.seq_len, device=x.device)
        x = x + self.pos_embed(positions)                    # (B, S, d_model)
        x = x + self.meta_embed(meta_idx)                    # (B, S, d_model)
        return self.drop(x)
