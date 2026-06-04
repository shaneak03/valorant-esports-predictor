"""
CLI: Process raw_matches.json into training-ready parquet + split files.

Steps:
  1. Load raw match dicts from JSON
  2. Build per-team Elo ratings (simple iterative Elo from chronological matches)
  3. For each match, build 20-match history windows for both teams
  4. Extract feature vectors
  5. Fit normalizer on train split, apply to all splits
  6. Save team_histories.parquet, matches.parquet, split ID files, scaler_params.json

Usage:
    python scripts/build_dataset.py --input data/processed/raw_matches.json
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.feature_extractor import extract_match_features, NUM_SCALAR_FEATURES
from src.data.normalizer import Normalizer
from src.data.team_filter import is_franchised_match, is_masters_london_match, involves_masters_london_team, resolve_team

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TRAIN_CUTOFF = "2024-09-01"
ELO_K = 32
ELO_INIT = 1500


def update_elo(elo_a: float, elo_b: float, won_a: bool) -> tuple[float, float]:
    expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    delta = ELO_K * ((1.0 if won_a else 0.0) - expected_a)
    return elo_a + delta, elo_b - delta


def build_elo_map(matches: list[dict]) -> tuple[dict, dict]:
    """
    Returns:
      elo_history : team -> [(date, opponent_elo_before, match_id)]
      self_elo    : match_id -> (elo_a_before, elo_b_before)
                    — each team's OWN Elo right before that match.
    """
    elo: dict[str, float] = defaultdict(lambda: ELO_INIT)
    history: dict[str, list] = defaultdict(list)
    self_elo: dict[int, tuple[float, float]] = {}

    for m in sorted(matches, key=lambda x: x.get("date") or ""):
        ta, tb = m["team_a"], m["team_b"]
        elo_a_before = elo[ta]
        elo_b_before = elo[tb]
        history[ta].append((m["date"], elo_b_before, m["match_id"]))
        history[tb].append((m["date"], elo_a_before, m["match_id"]))
        self_elo[m["match_id"]] = (elo_a_before, elo_b_before)
        won_a = m["winner"] == 0
        elo[ta], elo[tb] = update_elo(elo_a_before, elo_b_before, won_a)

    return history, self_elo


def build_samples(matches: list[dict], elo_history: dict, self_elo: dict) -> list[dict]:
    """
    For each match, build (history_a, history_b) feature windows and return samples.
    History is strictly from before the target match date.
    """
    # Index all matches by team for fast lookup
    team_matches: dict[str, list[dict]] = defaultdict(list)
    for m in sorted(matches, key=lambda x: x.get("date") or ""):
        team_matches[m["team_a"]].append(m)
        team_matches[m["team_b"]].append(m)

    samples = []
    for m in matches:
        if not m.get("date"):
            continue
        target_date = m["date"]
        ta, tb = m["team_a"], m["team_b"]

        history_a = _build_team_history(ta, m["match_id"], target_date, team_matches[ta], elo_history)
        history_b = _build_team_history(tb, m["match_id"], target_date, team_matches[tb], elo_history)

        # Each team's own Elo right before this match, normalised to ~[-2, +2]
        raw_elo_a, raw_elo_b = self_elo.get(m["match_id"], (ELO_INIT, ELO_INIT))
        elo_a_norm = (raw_elo_a - ELO_INIT) / 400.0
        elo_b_norm = (raw_elo_b - ELO_INIT) / 400.0

        samples.append({
            "match_id": m["match_id"],
            "date": target_date,
            "team_a": ta,
            "team_b": tb,
            "winner": m["winner"],
            "tournament_tier": m.get("tournament_tier", 0),
            "history_a": history_a,
            "history_b": history_b,
            "elo_a": float(elo_a_norm),
            "elo_b": float(elo_b_norm),
        })

    return samples


def _build_team_history(
    team: str,
    target_match_id: int,
    target_date: str,
    team_match_list: list[dict],
    elo_history: dict,
) -> list[dict]:
    """Build feature dicts for the last 20 matches before target_date for team."""
    elo_lookup = {mid: elo for (_, elo, mid) in elo_history.get(team, [])}

    past = [
        m for m in team_match_list
        if m.get("date") and m["date"] < target_date and m["match_id"] != target_match_id
    ]
    past = sorted(past, key=lambda x: x["date"])[-20:]

    features = []
    for m in past:
        side = "a" if m["team_a"] == team else "b"
        opponent = m["team_b"] if side == "a" else m["team_a"]
        opponent_elo = elo_lookup.get(m["match_id"], ELO_INIT)

        try:
            t_delta = (
                datetime.strptime(target_date, "%Y-%m-%d") -
                datetime.strptime(m["date"], "%Y-%m-%d")
            ).days
        except ValueError:
            t_delta = 0

        scalars, map_idx, meta_id = extract_match_features(m, side, opponent_elo, float(t_delta))
        features.append({"scalars": scalars, "map_idx": map_idx, "meta_id": meta_id})

    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/raw_matches.json")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--train-cutoff", default=TRAIN_CUTOFF)
    parser.add_argument("--min-tier", type=int, default=0, choices=[0, 1, 2],
                        help="Minimum tournament tier to include (default 0 = all).")
    parser.add_argument("--franchised-only", action="store_true",
                        help="Keep only matches where both teams are franchised VCT teams.")
    parser.add_argument("--masters-london-only", action="store_true",
                        help="Keep only matches where both teams qualified for Masters London.")
    parser.add_argument("--masters-london-any", action="store_true",
                        help="Keep matches where at least one team qualified for Masters London.")
    args = parser.parse_args()

    with open(args.input) as f:
        matches = json.load(f)
    log.info("Loaded %d matches.", len(matches))

    before = len(matches)
    if args.min_tier > 0:
        matches = [m for m in matches if m.get("tournament_tier", 0) >= args.min_tier]
        log.info("Tier filter: kept %d matches (removed %d).", len(matches), before - len(matches))

    if args.masters_london_only:
        before = len(matches)
        matches = [m for m in matches if is_masters_london_match(m["team_a"], m["team_b"])]
        log.info("Masters London (both) filter: kept %d matches (removed %d).", len(matches), before - len(matches))

    if args.masters_london_any:
        before = len(matches)
        matches = [m for m in matches if involves_masters_london_team(m["team_a"], m["team_b"])]
        log.info("Masters London (any) filter: kept %d matches (removed %d).", len(matches), before - len(matches))

    if args.franchised_only:
        before = len(matches)
        kept = []
        for m in matches:
            canonical_a = resolve_team(m["team_a"])
            canonical_b = resolve_team(m["team_b"])
            if canonical_a and canonical_b:
                # Normalise to canonical names so aliases merge into one identity
                m["team_a"] = canonical_a
                m["team_b"] = canonical_b
                kept.append(m)
        matches = kept
        log.info("Franchise filter: kept %d matches (removed %d).", len(matches), before - len(matches))

    log.info("Computing Elo ratings...")
    elo_history, self_elo = build_elo_map(matches)

    log.info("Building samples with history windows...")
    samples = build_samples(matches, elo_history, self_elo)
    log.info("Built %d samples.", len(samples))

    # --- Split ---
    train = [s for s in samples if s["date"] < args.train_cutoff]
    val = [s for s in samples if args.train_cutoff <= s["date"] < "2025-01-01"]
    test = [s for s in samples if s["date"] >= "2025-01-01"]
    log.info("Split: train=%d val=%d test=%d", len(train), len(val), len(test))

    # --- Normalizer: fit on train scalars ---
    def collect_scalars(sample_list: list[dict]) -> np.ndarray:
        rows = []
        for s in sample_list:
            for h in s["history_a"] + s["history_b"]:
                rows.append(h["scalars"])
        return np.stack(rows) if rows else np.zeros((1, NUM_SCALAR_FEATURES))

    normalizer = Normalizer()
    train_scalars = collect_scalars(train)
    normalizer.fit(train_scalars)

    scaler_path = Path(args.out_dir) / "scaler_params.json"
    normalizer.save(str(scaler_path))
    log.info("Saved normalizer to %s", scaler_path)

    # Apply normalization to all samples in-place
    for split in (train, val, test):
        for s in split:
            for h in s["history_a"] + s["history_b"]:
                h["scalars"] = normalizer.transform(h["scalars"].reshape(1, -1)).squeeze(0)

    # --- Save parquet ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def samples_to_df(sample_list: list[dict]) -> pd.DataFrame:
        rows = []
        for s in sample_list:
            rows.append({
                "match_id": s["match_id"],
                "date": s["date"],
                "team_a": s["team_a"],
                "team_b": s["team_b"],
                "winner": s["winner"],
                "tournament_tier": s["tournament_tier"],
            })
        return pd.DataFrame(rows)

    for name, split in [("train", train), ("val", val), ("test", test)]:
        df = samples_to_df(split)
        df.to_parquet(out_dir / f"{name}_matches.parquet", index=False)

    # Save full samples list (with history) as JSON for the Dataset
    import pickle
    for name, split in [("train", train), ("val", val), ("test", test)]:
        with open(out_dir / f"{name}_samples.pkl", "wb") as f:
            pickle.dump(split, f)
    log.info("Saved processed splits to %s", out_dir)

    # Save split ID files
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        ids = [str(s["match_id"]) for s in split]
        (splits_dir / f"{name}_ids.txt").write_text("\n".join(ids))
    log.info("Saved split ID files to %s", splits_dir)


if __name__ == "__main__":
    main()
