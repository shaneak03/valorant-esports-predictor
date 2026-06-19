"""
Flask API for the Valorant match predictor.
Run from the project root: python api/app.py
"""

import json as json_module
import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
from flask import Flask, jsonify, request
from flask_cors import CORS

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import MatchDataset
from src.data.team_filter import resolve_team
from src.evaluation.calibration import TemperatureScaler
from src.models.predictor import ValorantPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# ── Globals loaded at startup ────────────────────────────────────────────────
_model = None
_temperature = None
_all_samples: list = []
_raw_matches: list = []
_model_cfg: dict = {}

ELO_INIT = 1500.0
MIN_DATE_EDA = "2025-09-01"
CURRENT_POOL = ["Haven", "Lotus", "Split", "Pearl", "Ascent", "Breeze", "Fracture"]

ML_TEAMS_ORDERED = [
    "G2 Esports", "LEVIATÁN", "NRG",
    "Team Heretics", "Team Vitality", "FUT Esports",
    "Paper Rex", "FULL SENSE", "Global Esports",
    "EDward Gaming", "Xi Lai Gaming", "Dragon Ranger Gaming",
]

TEAMS_BY_REGION: dict[str, list[str]] = {
    "Americas": ["G2 Esports", "LEVIATÁN", "NRG"],
    "EMEA":     ["Team Heretics", "Team Vitality", "FUT Esports"],
    "Pacific":  ["Paper Rex", "FULL SENSE", "Global Esports"],
    "China":    ["EDward Gaming", "Xi Lai Gaming", "Dragon Ranger Gaming"],
}

ALL_MASTERS_TEAMS: set[str] = {t for teams in TEAMS_BY_REGION.values() for t in teams}

TEAM_REGION: dict[str, str] = {
    t: r for r, teams in TEAMS_BY_REGION.items() for t in teams
}


# ── Shared helper ─────────────────────────────────────────────────────────────

def _replay_matches(until_date: str) -> tuple[dict, dict, dict]:
    """Replay all samples up to (not including) until_date.
    Returns (elo_ratings, wins, losses) dicts keyed by team name.
    """
    elo: dict[str, float] = {}
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)

    for s in sorted(_all_samples, key=lambda x: x["date"]):
        if s["date"] >= until_date:
            break
        ta, tb = s["team_a"], s["team_b"]
        ea = elo.get(ta, ELO_INIT)
        eb = elo.get(tb, ELO_INIT)
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        delta = 32 * ((1.0 if s["winner"] == 0 else 0.0) - exp_a)
        elo[ta] = ea + delta
        elo[tb] = eb - delta
        if s["winner"] == 0:
            wins[ta] += 1
            losses[tb] += 1
        else:
            wins[tb] += 1
            losses[ta] += 1

    return elo, wins, losses


# ── Startup ──────────────────────────────────────────────────────────────────

def load_resources():
    global _model, _temperature, _all_samples, _raw_matches, _model_cfg

    root = Path(__file__).parent.parent

    with open(root / "configs/model_config.yaml") as f:
        _model_cfg = yaml.safe_load(f)

    _all_samples = []
    for split in ("train", "val", "test"):
        pkl = root / f"data/processed/{split}_samples.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                _all_samples.extend(pickle.load(f))
    log.info("Loaded %d samples", len(_all_samples))

    raw_path = root / "data/processed/raw_matches.json"
    if raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            _raw_matches = json_module.load(f)
        log.info("Loaded %d raw matches", len(_raw_matches))

    _model = ValorantPredictor(
        num_scalars=_model_cfg.get("num_scalar_features", 11),
        num_maps=_model_cfg.get("num_maps", 12),
        map_embed_dim=_model_cfg.get("map_embedding_dim", 16),
        d_model=_model_cfg.get("d_model", 64),
        seq_len=_model_cfg.get("seq_len", 20),
        num_heads=_model_cfg.get("num_heads", 4),
        dim_feedforward=_model_cfg.get("dim_feedforward", 256),
        num_layers=_model_cfg.get("num_layers", 3),
        num_metas=_model_cfg.get("num_metas", 37),
    )
    ckpt = root / "checkpoints/best_model.pt"
    _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    _model.eval()
    log.info("Model loaded from %s", ckpt)

    _temperature = TemperatureScaler.load(str(root / "checkpoints/temperature.json"))
    log.info("All resources ready")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/teams", methods=["GET"])
def get_teams():
    return jsonify(TEAMS_BY_REGION)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "team_a" not in data or "team_b" not in data:
        return jsonify({"error": "team_a and team_b are required"}), 400
    if data["team_a"] == data["team_b"]:
        return jsonify({"error": "Teams must be different"}), 400
    try:
        return jsonify(_run_prediction(data["team_a"], data["team_b"]))
    except Exception as e:
        log.exception("Prediction failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/rankings", methods=["GET"])
def rankings():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    thirty_ago = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")

    elo_now, wins, losses = _replay_matches(today)
    elo_30d, _, _ = _replay_matches(thirty_ago)

    result = []
    for region, teams in TEAMS_BY_REGION.items():
        for team in teams:
            elo = elo_now.get(team, ELO_INIT)
            w, l = wins[team], losses[team]
            result.append({
                "team":       team,
                "region":     region,
                "elo":        round(elo),
                "elo_change": round(elo - elo_30d.get(team, ELO_INIT)),
                "wins":       w,
                "losses":     l,
                "win_rate":   round(w / (w + l), 3) if (w + l) > 0 else 0.0,
            })

    result.sort(key=lambda x: x["elo"], reverse=True)
    for i, item in enumerate(result):
        item["rank"] = i + 1

    return jsonify(result)


@app.route("/api/team/<name>", methods=["GET"])
def team_stats(name):
    name = resolve_team(name) or name
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    elo: dict[str, float] = {}
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    elo_history: list[dict] = []
    recent_matches: list[dict] = []
    seen_elo_dates: set[str] = set()

    for s in sorted(_all_samples, key=lambda x: x["date"]):
        if s["date"] >= today:
            break
        ta, tb = s["team_a"], s["team_b"]

        if (ta == name or tb == name) and s["date"] not in seen_elo_dates:
            elo_history.append({"date": s["date"], "elo": round(elo.get(name, ELO_INIT))})
            seen_elo_dates.add(s["date"])

        ea = elo.get(ta, ELO_INIT)
        eb = elo.get(tb, ELO_INIT)
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        delta = 32 * ((1.0 if s["winner"] == 0 else 0.0) - exp_a)
        elo[ta] = ea + delta
        elo[tb] = eb - delta

        if s["winner"] == 0:
            wins[ta] += 1; losses[tb] += 1
        else:
            wins[tb] += 1; losses[ta] += 1

        if ta == name or tb == name:
            won = (ta == name and s["winner"] == 0) or (tb == name and s["winner"] == 1)
            recent_matches.append({
                "date":     s["date"],
                "opponent": tb if ta == name else ta,
                "result":   "W" if won else "L",
            })

    # Append final Elo snapshot
    final_elo = round(elo.get(name, ELO_INIT))
    if elo_history and elo_history[-1]["elo"] != final_elo:
        elo_history.append({"date": today, "elo": final_elo})

    w, l = wins[name], losses[name]

    return jsonify({
        "team":          name,
        "region":        TEAM_REGION.get(name, "Unknown"),
        "elo":           final_elo,
        "wins":          w,
        "losses":        l,
        "win_rate":      round(w / (w + l), 3) if (w + l) > 0 else 0.0,
        "recent_form":   [m["result"] for m in recent_matches[-10:]],
        "recent_matches": list(reversed(recent_matches[-20:])),
        "elo_history":   elo_history,
    })


@app.route("/api/stats", methods=["GET"])
def stats():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    elo: dict[str, float] = {}
    wins: dict[str, int] = defaultdict(int)
    losses: dict[str, int] = defaultdict(int)
    elo_timeline: dict[str, dict] = {}

    for s in sorted(_all_samples, key=lambda x: x["date"]):
        if s["date"] >= today:
            break
        ta, tb = s["team_a"], s["team_b"]

        month = s["date"][:7]
        if month not in elo_timeline:
            elo_timeline[month] = {t: round(elo.get(t, ELO_INIT)) for t in ALL_MASTERS_TEAMS}

        ea = elo.get(ta, ELO_INIT)
        eb = elo.get(tb, ELO_INIT)
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        delta = 32 * ((1.0 if s["winner"] == 0 else 0.0) - exp_a)
        elo[ta] = ea + delta
        elo[tb] = eb - delta
        if s["winner"] == 0:
            wins[ta] += 1; losses[tb] += 1
        else:
            wins[tb] += 1; losses[ta] += 1

    # Elo distribution
    elo_dist = []
    for region, teams in TEAMS_BY_REGION.items():
        for team in teams:
            w, l = wins[team], losses[team]
            elo_dist.append({
                "team":     team,
                "region":   region,
                "elo":      round(elo.get(team, ELO_INIT)),
                "wins":     w,
                "losses":   l,
            })
    elo_dist.sort(key=lambda x: x["elo"], reverse=True)

    # Win rate by region
    region_agg: dict[str, dict] = defaultdict(lambda: {"wins": 0, "total": 0})
    for region, teams in TEAMS_BY_REGION.items():
        for team in teams:
            w, l = wins[team], losses[team]
            region_agg[region]["wins"] += w
            region_agg[region]["total"] += w + l

    win_rates = [
        {
            "region":   r,
            "win_rate": round(v["wins"] / v["total"], 3) if v["total"] > 0 else 0.0,
            "wins":     v["wins"],
            "total":    v["total"],
        }
        for r, v in region_agg.items()
    ]

    # Elo timeline (monthly snapshots)
    timeline = [{"date": m, **elos} for m, elos in sorted(elo_timeline.items())]

    return jsonify({
        "elo_distribution":    elo_dist,
        "win_rates_by_region": win_rates,
        "elo_timeline":        timeline,
    })


@app.route("/api/eda/full", methods=["GET"])
def eda_full():
    min_date = request.args.get("min_date", MIN_DATE_EDA)
    matches = [m for m in _raw_matches if m.get("date", "") >= min_date]

    # ── 1. Map stats per team ────────────────────────────────────────────────
    map_raw: dict = {t: defaultdict(lambda: dict(played=0, wins=0, atk=[], def_=[]))
                     for t in ALL_MASTERS_TEAMS}

    for match in matches:
        ta, tb = match["team_a"], match["team_b"]
        winner_side_map = {"a": ta, "b": tb}
        for mp in match.get("maps", []):
            name = mp.get("map", "")
            if not name or name in ("unknown", "TBD"):
                continue
            ws = mp.get("winner_side")
            for team, side in [(ta, "a"), (tb, "b")]:
                if team not in ALL_MASTERS_TEAMS:
                    continue
                s = map_raw[team][name]
                s["played"] += 1
                if ws == side:
                    s["wins"] += 1
                atk = mp.get(f"atk_win_rate_{side}")
                def_ = mp.get(f"def_win_rate_{side}")
                if atk is not None:
                    s["atk"].append(atk)
                if def_ is not None:
                    s["def_"].append(def_)

    map_stats: dict = {}
    for team in ML_TEAMS_ORDERED:
        n_series = sum(1 for m in matches if m["team_a"] == team or m["team_b"] == team)
        map_stats[team] = {}
        for pool_map in CURRENT_POOL:
            s = map_raw[team].get(pool_map, {})
            played = s.get("played", 0)
            wins = s.get("wins", 0)
            atk_list = s.get("atk", [])
            def_list = s.get("def_", [])
            map_stats[team][pool_map] = {
                "played": played,
                "wins": wins,
                "losses": played - wins,
                "win_rate": round(wins / played, 3) if played >= 3 else None,
                "avg_atk": round(sum(atk_list) / len(atk_list), 3) if atk_list else None,
                "avg_def": round(sum(def_list) / len(def_list), 3) if def_list else None,
                "play_freq": round(played / n_series, 3) if n_series > 0 else 0.0,
            }

    # ── 2. Overall stats ────────────────────────────────────────────────────
    overall_stats = []
    for team in ML_TEAMS_ORDERED:
        team_matches = sorted(
            [(m, "a" if m["team_a"] == team else "b")
             for m in matches if m["team_a"] == team or m["team_b"] == team],
            key=lambda x: x[0]["date"],
        )
        results = [
            1 if (m["winner"] == 0 and s == "a") or (m["winner"] == 1 and s == "b") else 0
            for m, s in team_matches
        ]
        n, w = len(results), sum(results)

        streak, streak_char = 0, "-"
        if results:
            last = results[-1]
            for r in reversed(results):
                if r == last:
                    streak += 1
                else:
                    break
            streak_char = f"W{streak}" if last == 1 else f"L{streak}"

        last5 = "".join("W" if r else "L" for r in results[-5:])

        ps = {pm: map_stats[team][pm] for pm in CURRENT_POOL
              if map_stats[team][pm]["win_rate"] is not None}
        best_map = max(ps, key=lambda pm: ps[pm]["win_rate"]) if ps else None
        worst_map = min(ps, key=lambda pm: ps[pm]["win_rate"]) if ps else None

        play_freqs = {pm: map_stats[team][pm]["play_freq"] for pm in CURRENT_POOL}
        likely_ban = min(play_freqs, key=play_freqs.get) if play_freqs else None

        overall_stats.append({
            "team":         team,
            "region":       TEAM_REGION.get(team, "Unknown"),
            "series":       n,
            "win_pct":      round(w / n, 3) if n > 0 else 0.0,
            "last5":        last5,
            "streak":       streak_char,
            "best_map":     best_map,
            "best_map_wr":  round(ps[best_map]["win_rate"], 3) if best_map else None,
            "worst_map":    worst_map,
            "worst_map_wr": round(ps[worst_map]["win_rate"], 3) if worst_map else None,
            "likely_ban":   likely_ban,
        })

    # ── 3. H2H matrix ───────────────────────────────────────────────────────
    h2h: dict = {t: {} for t in ML_TEAMS_ORDERED}
    for match in matches:
        ta, tb = match["team_a"], match["team_b"]
        if ta not in ALL_MASTERS_TEAMS or tb not in ALL_MASTERS_TEAMS:
            continue
        ta_won = match["winner"] == 0
        for a, b, a_won in [(ta, tb, ta_won), (tb, ta, not ta_won)]:
            if b not in h2h[a]:
                h2h[a][b] = {"wins": 0, "total": 0}
            h2h[a][b]["total"] += 1
            if a_won:
                h2h[a][b]["wins"] += 1

    for a in h2h:
        for b in h2h[a]:
            e = h2h[a][b]
            e["win_rate"] = round(e["wins"] / e["total"], 3) if e["total"] > 0 else None

    # ── 4. Recent form (rolling 8-series) ───────────────────────────────────
    recent_form: dict = {}
    for team in ML_TEAMS_ORDERED:
        team_matches = sorted(
            [(m, "a" if m["team_a"] == team else "b")
             for m in matches if m["team_a"] == team or m["team_b"] == team],
            key=lambda x: x[0]["date"],
        )
        results = [
            1 if (m["winner"] == 0 and s == "a") or (m["winner"] == 1 and s == "b") else 0
            for m, s in team_matches
        ]
        window = min(8, len(results))
        rolling = []
        for i, res in enumerate(results):
            start = max(0, i - window + 1)
            span = i - start + 1
            if span >= 3:
                rolling.append({
                    "index": i,
                    "rolling_wr": round(sum(results[start: i + 1]) / span, 3),
                    "result": res,
                })
        recent_form[team] = rolling

    return jsonify({
        "teams":        ML_TEAMS_ORDERED,
        "map_pool":     CURRENT_POOL,
        "map_stats":    map_stats,
        "overall_stats": overall_stats,
        "h2h":          h2h,
        "recent_form":  recent_form,
        "min_date":     min_date,
    })


# ── Prediction logic ──────────────────────────────────────────────────────────

def _run_prediction(team_a: str, team_b: str, date: str | None = None) -> dict:
    date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    team_a = resolve_team(team_a) or team_a
    team_b = resolve_team(team_b) or team_b

    team_history: dict[str, list] = defaultdict(list)
    elo: dict[str, float] = {}

    for s in sorted(_all_samples, key=lambda x: x["date"]):
        if s["date"] >= date:
            break
        ta, tb = s["team_a"], s["team_b"]
        ea = elo.get(ta, ELO_INIT)
        eb = elo.get(tb, ELO_INIT)
        exp_a = 1 / (1 + 10 ** ((eb - ea) / 400))
        delta = 32 * ((1.0 if s["winner"] == 0 else 0.0) - exp_a)
        elo[ta] = ea + delta
        elo[tb] = eb - delta
        if s["history_a"]:
            team_history[ta].extend(s["history_a"])
        if s["history_b"]:
            team_history[tb].extend(s["history_b"])

    raw_elo_a = elo.get(team_a, ELO_INIT)
    raw_elo_b = elo.get(team_b, ELO_INIT)

    sample = {
        "match_id": -1,
        "date":     date,
        "team_a":   team_a,
        "team_b":   team_b,
        "winner":   0,
        "history_a": team_history.get(team_a, [])[-20:],
        "history_b": team_history.get(team_b, [])[-20:],
        "elo_a":    (raw_elo_a - ELO_INIT) / 400.0,
        "elo_b":    (raw_elo_b - ELO_INIT) / 400.0,
    }

    ds = MatchDataset([sample])
    batch = ds[0]
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].unsqueeze(0)

    T = float(_temperature.temperature.item())

    with torch.no_grad():
        logit_ab = _model(
            batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
            batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
            batch["meta_idx_a"], batch["meta_idx_b"],
            batch["elo_a"], batch["elo_b"],
        ).squeeze()
        logit_ba = _model(
            batch["scalars_b"], batch["map_idx_b"], batch["pad_mask_b"],
            batch["scalars_a"], batch["map_idx_a"], batch["pad_mask_a"],
            batch["meta_idx_b"], batch["meta_idx_a"],
            batch["elo_b"], batch["elo_a"],
        ).squeeze()
        prob_a = ((torch.sigmoid(logit_ab / T) + 1.0 - torch.sigmoid(logit_ba / T)) / 2).item()

    return {
        "team_a": team_a,
        "team_b": team_b,
        "prob_a": round(prob_a, 4),
        "prob_b": round(1 - prob_a, 4),
        "winner": team_a if prob_a > 0.5 else team_b,
        "elo_a":  round(raw_elo_a),
        "elo_b":  round(raw_elo_b),
    }


if __name__ == "__main__":
    load_resources()
    app.run(host="0.0.0.0", debug=False, port=5000)
