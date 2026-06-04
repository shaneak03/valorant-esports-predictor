# Valorant Esports Match Predictor

A transformer-based model that predicts which team wins a professional Valorant series. Given each team's last 20 match results, a shared Transformer encoder builds a representation for each team, and a FFNN classification head outputs the win probability.

Built to predict Masters London 2026 matches.

---

## How It Works

Each match in a team's recent history is encoded as a 16-feature vector (round win rates, attack/defense rates, pistol win rates, opponent Elo, etc.). These 20 vectors form a sequence fed into a shared Transformer encoder — the same weights process both teams, producing two team representations. These are combined symmetrically as `[repr_a − repr_b, repr_a + repr_b]` and passed through a FFNN to output P(team_a wins).

```
team_a history (20 matches) ──┐
                               ├── Shared Transformer → repr_a ──┐
team_b history (20 matches) ──┘                                   ├── [a-b, a+b] → FFNN → P(A wins)
                                                       repr_b ───┘
```

Inference is symmetric: both orderings (A vs B) and (B vs A) are run and averaged, so the output is identical regardless of which team you call team_a.

---

## Quickstart

### 1. Set up environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Scrape match data from VLR.gg

```bash
# Scrape all matches (all tiers, all teams)
python scripts/scrape_data.py

# Scrape only matches involving at least one Masters London qualifier
python scripts/scrape_data.py --masters-london-any

# Pick up new matches added since last scrape (re-fetches first 5 listing pages)
python scripts/scrape_data.py --force-listing-pages 5
```

If the scrape is interrupted, re-run the same command — it resumes from a checkpoint automatically.

### 3. Build the training dataset

```bash
python scripts/build_dataset.py
```

This builds Elo ratings, 20-match history windows, extracts features, normalizes, and saves train/val/test splits.

### 4. Train the model

```bash
python scripts/train.py
```

Checkpoints are saved to `checkpoints/`.

### 5. Evaluate on test set

```bash
python scripts/evaluate.py
```

### 6. Predict a match

```bash
python scripts/predict.py --team-a "Team Vitality" --team-b "Paper Rex" --date 2026-06-01
# Output: P(Team Vitality wins) = 0.61
```

### 7. Run tests

```bash
pytest tests/
```

---

## Project Structure

```
val_esports_predictor/
├── src/
│   ├── scraper/
│   │   ├── vlr_scraper.py          # Main VLR.gg HTML scraper
│   │   ├── rate_limiter.py         # Token bucket — 0.5 req/s default
│   │   └── cache.py                # SHA256-keyed HTML disk cache
│   ├── data/
│   │   ├── feature_extractor.py    # Match dict → (16 scalars, map_idx, meta_id)
│   │   ├── dataset.py              # PyTorch Dataset: sequences + padding masks + labels
│   │   ├── normalizer.py           # Z-score fit/transform, save/load JSON
│   │   ├── augmentation.py         # Team-swap augmentation (doubles data, balances classes)
│   │   └── team_filter.py          # Franchise allowlist + Masters London team set
│   ├── models/
│   │   ├── match_encoder.py        # Scalar+map → 64-dim token (proj + LayerNorm + meta embed)
│   │   ├── transformer.py          # Shared TransformerEncoder + mean pooling
│   │   ├── classifier.py           # FFNN binary head: 128 → 64 → 32 → 1
│   │   └── predictor.py            # Top-level model wiring all components
│   ├── training/
│   │   ├── trainer.py              # Train/val loops, early stopping on Brier score
│   │   ├── losses.py               # BCEWithLogitsLoss + label smoothing (ε=0.1)
│   │   └── metrics.py              # Accuracy, ROC-AUC, Brier score, log-loss, ECE
│   └── evaluation/
│       └── calibration.py          # Temperature scaling + reliability diagrams
├── scripts/
│   ├── scrape_data.py              # CLI: scrape VLR.gg → raw_matches.json
│   ├── build_dataset.py            # CLI: raw JSON → processed splits + normalizer
│   ├── train.py                    # CLI: train the model
│   ├── evaluate.py                 # CLI: evaluate on test set
│   └── predict.py                  # CLI: predict a single upcoming match
├── notebooks/
│   ├── 02_simple_model.ipynb       # Logistic regression / MLP / TinyTransformer experiments
│   └── 03_masters_london_eda.ipynb # Per-team EDA: map win rates, permabans, recent form
├── configs/
│   ├── model_config.yaml           # Architecture hyperparameters
│   └── training_config.yaml        # Training hyperparameters
├── tests/
│   ├── test_model_shapes.py        # Shape/range assertions for all model components
│   ├── test_dataset.py             # Dataset + augmentation unit tests
│   ├── scraper_preview.py          # Scrapes 3 live matches and prints parsed output
│   ├── verify_scraper.py           # Checks cached HTML parses cleanly
│   └── inspect_html.py             # Debug tool: check CSS selectors against cached HTML
└── requirements.txt
```

---

## File-by-File Reference

### Scripts (entry points)

| File | What it does |
|---|---|
| `scripts/scrape_data.py` | Paginates VLR.gg results, fetches each match page, saves raw match dicts to JSON. Supports checkpoint/resume and `--force-listing-pages` to pick up new matches. |
| `scripts/build_dataset.py` | Builds Elo ratings, 20-match history windows, extracts 16 features per match, fits normalizer on train split, saves pkl splits. |
| `scripts/train.py` | Loads processed splits, trains the model with early stopping, temperature-calibrates on val set, saves checkpoint. |
| `scripts/evaluate.py` | Loads checkpoint, runs inference on test set, prints accuracy / ROC-AUC / Brier / log-loss / ECE. |
| `scripts/predict.py` | Loads a checkpoint, builds sequences from match history, outputs calibrated P(team_a wins) using symmetric inference. |

### Scraper (`src/scraper/`)

| File | What it does |
|---|---|
| `vlr_scraper.py` | Core scraper. Fetches match listing pages and individual match pages from VLR.gg. Parses teams, scores, dates, tournament info, and per-map round/attack/defense/pistol/streak stats. |
| `rate_limiter.py` | Token bucket rate limiter. Default: 0.5 req/s (1 request every 2 seconds) to avoid getting IP-banned. |
| `cache.py` | Saves raw HTML to `data/raw/` keyed by SHA256 of the URL. Re-running the scraper never re-fetches already-cached pages. |

### Data (`src/data/`)

| File | What it does |
|---|---|
| `feature_extractor.py` | Converts a raw match dict into **16 scalar features** + a map index + a meta period ID. See feature table below. |
| `dataset.py` | PyTorch `Dataset`. Pads sequences to length 20 (left-pad with zeros), returns scalars, map indices, meta indices, padding masks, and labels. |
| `normalizer.py` | Fits z-score (mean/std per feature) on training data only. Saves params to JSON so inference uses the same scale without data leakage. |
| `augmentation.py` | Wraps the base dataset with team-swap: each sample is doubled with A↔B swapped and label flipped. Perfectly balances classes and doubles training data. |
| `team_filter.py` | `TEAM_ALIASES` maps VLR.gg display name variants to canonical names. `MASTERS_LONDON_TEAMS` is the set of 12 qualified teams. |

### Models (`src/models/`)

| File | What it does |
|---|---|
| `match_encoder.py` | Projects each match token to 64 dims: looks up 16-dim map embedding, concatenates with 16 scalar features → `Linear(32→64)` + `LayerNorm`, then adds a learned meta period embedding (which Valorant act/patch the match was in). |
| `transformer.py` | `TeamEncoder`: runs a 3-layer `TransformerEncoder` (d=64, heads=4, ff=256, Pre-LN, GELU, dropout=0.1), mean-pools over non-padded positions to get a single team representation. Same weights used for both teams. |
| `classifier.py` | `ClassifierHead`: takes `[repr_a − repr_b, repr_a + repr_b]` (128-dim symmetric combination), applies `Linear(128→64)→GELU→Dropout→Linear(64→32)→GELU→Dropout→Linear(32→1)`. |
| `predictor.py` | `ValorantPredictor`: wires together `MatchEncoder`, `TeamEncoder` (shared), and `ClassifierHead`. |

### Training (`src/training/`)

| File | What it does |
|---|---|
| `trainer.py` | Runs train and validation loops. Uses AdamW + CosineAnnealingLR. Early stopping monitors validation Brier score (patience=20). Saves best checkpoint. |
| `losses.py` | `SmoothedBCELoss` — `BCEWithLogitsLoss` with manual label smoothing (ε=0.1). Prevents overconfident predictions. |
| `metrics.py` | Computes accuracy, ROC-AUC, Brier score, log-loss, and ECE (Expected Calibration Error). |

### Evaluation (`src/evaluation/`)

| File | What it does |
|---|---|
| `calibration.py` | `TemperatureScaler`: fits a single scalar `T` on the validation set (LBFGS). Divides logits by `T` before sigmoid to produce calibrated probabilities. |

---

## Feature Vector (16 scalars per historical match)

| # | Feature | Description |
|---|---|---|
| 0 | `win_binary` | Did this team win the series? (1 or 0) |
| 1 | `maps_won` | Maps this team won |
| 2 | `maps_lost` | Maps opponent won |
| 3 | `maps_played` | Total maps in the series |
| 4 | `map_win_rate` | maps_won / maps_played |
| 5 | `map_score_diff` | maps_won − maps_lost |
| 6 | `tournament_tier` | 0=Challengers, 1=VCT League, 2=Masters/Champions |
| 7 | `bracket_stage` | 0=groups, 1=playoffs, 2=grand final |
| 8 | `opponent_elo` | Elo rating of the opponent at match time |
| 9 | `days_since_match` | Days before the target match (recency signal) |
| 10 | `max_round_streak` | Longest consecutive round win streak across maps |
| 11 | `pistol_win_rate` | Fraction of pistol rounds (rounds 1+13) won, averaged across maps |
| 12 | `overtime_rate` | Fraction of maps that went to overtime (>24 rounds) |
| 13 | `avg_round_win_rate` | Average round win rate across all maps (dominance signal) |
| 14 | `avg_attack_win_rate` | Average T-side round win rate (attack strength) |
| 15 | `avg_defense_win_rate` | Average CT-side round win rate (defense strength) |

Each token also carries two embeddings added on top:
- **Map embedding** (16-dim): which map was played
- **Meta period embedding** (64-dim): which Valorant act/patch the match was played in (36 defined periods + unknown)

---

## Scrape Flags Reference

```
python scripts/scrape_data.py [OPTIONS]

--max-pages N              Max listing pages to paginate (default 500, ~20 matches/page)
--force-listing-pages N    Re-fetch first N listing pages even if cached (default 5)
                           Use this to pick up new matches since last scrape
--min-tier {0,1,2}         0=all (default)  1=VCT leagues+intl  2=Masters/Champions only
--franchised-only          Keep only matches where both teams are franchised VCT teams
--masters-london-any       Keep only matches involving at least one Masters London qualifier
--rate FLOAT               Requests per second (default 0.5)
--save-every N             Save checkpoint every N matches processed (default 50)
--out PATH                 Output JSON path (default data/processed/raw_matches.json)
```

```
python scripts/build_dataset.py [OPTIONS]

--input PATH               Input JSON (default data/processed/raw_matches.json)
--min-tier {0,1,2}         Filter by tournament tier
--franchised-only          Keep only franchised VCT team matches
--masters-london-only      Keep only matches where BOTH teams are in Masters London
--masters-london-any       Keep matches where at least one team is in Masters London
--train-cutoff DATE        Train/val split date (default 2024-09-01)
```

---

## Team Filters

| Flag | Effect |
|---|---|
| `--franchised-only` | Both teams must be in the franchise partner list (VCT Americas, EMEA, Pacific, China). Filters out Challengers teams entirely. |
| `--masters-london-any` | At least one team must have qualified for Masters London 2026. Good for a focused dataset. |
| `--masters-london-only` | Both teams must have qualified. Very few matches — mainly useful for final evaluation. |

To add a new team alias (e.g. after a rebrand), edit `src/data/team_filter.py`:
```python
TEAM_ALIASES: dict[str, str] = {
    ...
    "New Sponsor Name": "Canonical Team Name",
    ...
}
```

---

## Train / Val / Test Split

| Split | Date range | Purpose |
|---|---|---|
| Train | before 2024-09-01 | Model training |
| Val | 2024-09-01 to 2024-12-31 | Early stopping, temperature calibration |
| Test | 2025-01-01 onwards | Final evaluation — touch only once |

The normalizer is fit on the train split only. Val and test use the same train statistics to prevent data leakage.

---

## Achieved Performance (test set, 579 samples)

| Metric | Random baseline | Target | Achieved |
|---|---|---|---|
| Accuracy | 50% | >60% | **62.0%** ✅ |
| ROC-AUC | 0.50 | >0.65 | **0.652** ✅ |
| Brier Score | 0.250 | <0.220 | 0.233 |
| Log-loss | 0.693 | <0.650 | 0.658 |
| ECE | — | <0.05 | **0.040** ✅ |
