"""
Feature extractor for series-level match data from the VLR scraper.

Feature vector (16 scalars) — all normalized by the Normalizer before training:
  0  win_binary          — 1.0 if this team won the series, else 0.0
  1  maps_won            — maps this team won (0–5)
  2  maps_lost           — maps opponent won  (0–5)
  3  maps_played         — total maps played  (1–5)
  4  map_win_rate        — maps_won / maps_played (0.0–1.0)
  5  map_score_diff      — maps_won - maps_lost  (-5 to +5)
  6  tournament_tier     — ordinal 0 / 1 / 2
  7  bracket_stage       — ordinal 0 (group) / 1 (playoff) / 2 (final)
  8  opponent_elo        — Elo of the opponent at match time (normalized later)
  9  days_since_match    — days before the target match (normalized later)
 10  max_round_streak    — max consecutive round win streak across maps in that match
 11  pistol_win_rate     — fraction of pistol rounds (rounds 1 + 13) won, averaged across maps
 12  overtime_rate       — fraction of maps in this series that went to overtime (>24 rounds)
 13  avg_round_win_rate  — average round win rate across all maps (how dominant round-by-round)
 14  avg_attack_win_rate — average T-side (attack) round win rate across maps
 15  avg_defense_win_rate— average CT-side (defense) round win rate across maps

Map embedding: map_idx is always set to UNKNOWN_MAP_IDX so the embedding is a constant.
"""

from __future__ import annotations

import bisect
from datetime import datetime

import numpy as np

# Keep map constants for compatibility with MatchEncoder / dataset code
MAP_LIST = [
    "Bind", "Haven", "Split", "Ascent", "Icebox",
    "Breeze", "Fracture", "Pearl", "Lotus", "Sunset", "Abyss",
]
MAP_TO_IDX = {m.lower(): i for i, m in enumerate(MAP_LIST)}
UNKNOWN_MAP_IDX = len(MAP_LIST)   # 11
NUM_MAPS = len(MAP_LIST) + 1      # 12 total slots

NUM_SCALAR_FEATURES = 16

# ---------------------------------------------------------------------------
# Meta periods
# ---------------------------------------------------------------------------

META_PERIODS = [
    # Episode 1
    ("2020-06-02", 0),   # E1A1
    ("2020-08-13", 1),   # E1A2
    ("2020-10-13", 2),   # E1A3
    # Episode 2
    ("2021-01-12", 3),   # E2A1
    ("2021-03-02", 4),   # E2A2
    ("2021-04-27", 5),   # E2A3
    # Episode 3
    ("2021-06-22", 6),   # E3A1
    ("2021-09-08", 7),   # E3A2
    ("2021-11-02", 8),   # E3A3
    # Episode 4
    ("2022-01-11", 9),   # E4A1
    ("2022-03-01", 10),  # E4A2
    ("2022-04-27", 11),  # E4A3
    # Episode 5
    ("2022-06-22", 12),  # E5A1
    ("2022-08-23", 13),  # E5A2
    ("2022-10-17", 14),  # E5A3
    # Episode 6
    ("2023-01-10", 15),  # E6A1
    ("2023-03-06", 16),  # E6A2
    ("2023-04-25", 17),  # E6A3
    # Episode 7
    ("2023-06-27", 18),  # E7A1
    ("2023-08-29", 19),  # E7A2
    ("2023-10-31", 20),  # E7A3
    # Episode 8
    ("2024-01-09", 21),  # E8A1
    ("2024-03-05", 22),  # E8A2
    ("2024-04-30", 23),  # E8A3
    # Episode 9
    ("2024-06-25", 24),  # E9A1
    ("2024-08-28", 25),  # E9A2
    ("2024-10-23", 26),  # E9A3
    # Season 2025
    ("2025-01-08", 27),  # S25A1
    ("2025-03-05", 28),  # S25A2
    ("2025-04-30", 29),  # S25A3
    ("2025-06-25", 30),  # S25A4
    ("2025-08-20", 31),  # S25A5
    ("2025-10-15", 32),  # S25A6
    # Season 2026
    ("2026-01-07", 33),  # S26A1
    ("2026-03-18", 34),  # S26A2
    ("2026-04-29", 35),  # S26A3
]
NUM_META_PERIODS = 37  # 36 defined + 1 unknown (ID 36)

_META_DATES = [p[0] for p in META_PERIODS]
_META_IDS = [p[1] for p in META_PERIODS]
_FIRST_META_DATE = _META_DATES[0]  # "2020-06-02"


def get_meta_period(date_str: str | None) -> int:
    """
    Return the meta period ID for a given date string (YYYY-MM-DD).
    Returns 36 (unknown) if date is None or before 2020-06-02.
    """
    if date_str is None:
        return 36
    if date_str < _FIRST_META_DATE:
        return 36
    # bisect_right gives the insertion point after all dates <= date_str,
    # so subtract 1 to get the last period that started on or before date_str.
    idx = bisect.bisect_right(_META_DATES, date_str) - 1
    if idx < 0:
        return 36
    return _META_IDS[idx]


def extract_match_features(
    match: dict,
    team_side: str,               # "a" or "b"
    opponent_elo: float,
    days_before_target: float,
) -> tuple[np.ndarray, int, int]:
    """
    Build the (11,) scalar feature vector for one team's perspective on
    a single historical match.

    Returns (scalars: np.ndarray shape (11,), map_idx: int, meta_id: int)
    """
    score_a = float(match.get("score_a", 0))
    score_b = float(match.get("score_b", 0))
    maps_played = float(match.get("maps_played", score_a + score_b))

    if team_side == "a":
        my_maps = score_a
        opp_maps = score_b
    else:
        my_maps = score_b
        opp_maps = score_a

    winner_side = "a" if match["winner"] == 0 else "b"
    win_binary = 1.0 if team_side == winner_side else 0.0

    map_win_rate = my_maps / max(maps_played, 1.0)
    map_score_diff = my_maps - opp_maps

    maps = match.get("maps", [])

    # max_round_streak: longest consecutive round win streak across maps
    if team_side == "a":
        max_streak = max((m.get("max_round_streak_a", 0) for m in maps), default=0)
    else:
        max_streak = max((m.get("max_round_streak_b", 0) for m in maps), default=0)

    # pistol_win_rate: average fraction of pistol rounds won across maps
    pistol_key = "pistol_win_rate_a" if team_side == "a" else "pistol_win_rate_b"
    pistol_vals = [m[pistol_key] for m in maps if pistol_key in m]
    pistol_win_rate = float(np.mean(pistol_vals)) if pistol_vals else 0.5

    # overtime_rate: fraction of maps that went to overtime (total rounds > 24)
    ot_maps = sum(
        1 for m in maps
        if m.get("rounds_a", 0) + m.get("rounds_b", 0) > 24
    )
    overtime_rate = ot_maps / max(len(maps), 1) if maps else 0.0

    # Round-level performance — how dominant were they within maps?
    # These capture whether a 2-1 series win was dominant or barely scraped by.
    if team_side == "a":
        rwr_key = "round_win_rate_a"
        atk_key = "atk_win_rate_a"
        def_key = "def_win_rate_a"
    else:
        rwr_key = "round_win_rate_b"
        atk_key = "atk_win_rate_b"
        def_key = "def_win_rate_b"

    rwr_vals = [m[rwr_key] for m in maps if rwr_key in m]
    atk_vals = [m[atk_key] for m in maps if atk_key in m]
    def_vals = [m[def_key] for m in maps if def_key in m]

    avg_round_win_rate  = float(np.mean(rwr_vals)) if rwr_vals else 0.5
    avg_attack_win_rate = float(np.mean(atk_vals)) if atk_vals else 0.25
    avg_defense_win_rate= float(np.mean(def_vals)) if def_vals else 0.25

    scalars = np.array([
        win_binary,
        my_maps,
        opp_maps,
        maps_played,
        map_win_rate,
        map_score_diff,
        float(match.get("tournament_tier", 0)),
        float(match.get("bracket_stage", 0)),
        float(opponent_elo),
        float(days_before_target),
        float(max_streak),
        float(pistol_win_rate),
        float(overtime_rate),
        float(avg_round_win_rate),
        float(avg_attack_win_rate),
        float(avg_defense_win_rate),
    ], dtype=np.float32)

    meta_id = get_meta_period(match.get("date"))

    return scalars, UNKNOWN_MAP_IDX, meta_id
