# Valorant Esports Match Predictor

A transformer-based model that predicts which team wins a professional Valorant series. Given each team's last 20 match results, a shared Transformer encoder builds a representation for each team, and a FFNN classification head outputs the win probability.

Built to predict Masters London 2026 matches.

---

## How It Works

Each match in a team's recent history is encoded as a feature vector (round win rates, attack/defense rates, opponent strength, etc.). These 20 vectors form a sequence fed into a Transformer encoder with a CLS token — the same encoder is shared for both teams. The two CLS outputs are concatenated and passed through a small FFNN to produce P(team_a wins).

```
team_a history (20 matches) ──┐
                               ├── Shared Transformer → repr_a ──┐
team_b history (20 matches) ──┘                  repr_b ──┤── concat → FFNN → P(A wins)
```

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
# Scrape all matches involving at least one Masters London qualifier
python scripts/scrape_data.py --masters-london-any

# Or scrape everything (all tiers, all teams) — takes longer
python scripts/scrape_data.py --max-pages 500

# Scrape only franchised VCT teams
python scripts/scrape_data.py --franchised-only
```

If the scrape is interrupted, just re-run the same command — it resumes from a checkpoint automatically.

### 3. Build the training dataset

```bash
python scripts/build_dataset.py --input data/processed/raw_matches.json
```

This builds Elo ratings, 20-match history windows, normalizes features, and saves train/val/test splits.

### 4. Train the model

```bash
python scripts/train.py
```

Checkpoints and logs are saved to `checkpoints/`.

### 5. Predict a match

```bash
python scripts/predict.py --team-a "Team Vitality" --team-b "Paper Rex" --date 2026-06-01
# Output: P(Team Vitality wins) = 0.61
```

### 6. Run tests

```bash
pytest tests/
```

---

## Project Structure

```
val_esports_predictor/
├── data/
│   ├── raw/                        # Cached HTML from VLR.gg (keyed by URL hash)
│   ├── processed/
│   │   ├── raw_matches.json        # Raw match dicts from the scraper
│   │   ├── train/val/test_samples.pkl  # Processed samples with history windows
│   │   ├── train/val/test_matches.parquet  # Lightweight match metadata
│   │   └── scaler_params.json      # Z-score normalization stats (fit on train only)
│   └── splits/
│       ├── train_ids.txt
│       ├── val_ids.txt
│       └── test_ids.txt
├── src/
│   ├── scraper/
│   │   ├── vlr_scraper.py          # Main VLR.gg HTML scraper
│   │   ├── rate_limiter.py         # Token bucket — 0.5 req/s default
│   │   └── cache.py                # SHA256-keyed HTML disk cache
│   ├── data/
│   │   ├── feature_extractor.py    # Match dict → (scalar_features, map_idx)
│   │   ├── dataset.py              # PyTorch Dataset: sequences + padding masks + labels
│   │   ├── normalizer.py           # Z-score fit/transform, save/load JSON
│   │   ├── augmentation.py         # Team-swap augmentation (doubles data, balances classes)
│   │   └── team_filter.py          # Franchise allowlist + Masters London team set
│   ├── models/
│   │   ├── match_encoder.py        # Scalar+map → 64-dim token (proj + LayerNorm + pos embed)
│   │   ├── transformer.py          # Shared TransformerEncoder + CLS-token pooling
│   │   ├── classifier.py           # FFNN binary head: 128 → 64 → 32 → 1
│   │   └── predictor.py            # Top-level model wiring all components (~164k params)
│   ├── training/
│   │   ├── trainer.py              # Train/val loops, early stopping on Brier score
│   │   ├── losses.py               # BCEWithLogitsLoss + label smoothing (ε=0.1)
│   │   └── metrics.py              # Accuracy, ROC-AUC, Brier score, log-loss, ECE
│   └── evaluation/
│       ├── calibration.py          # Temperature scaling + reliability diagrams
│       └── baselines.py            # Random / win-rate / Elo-logistic baselines
├── scripts/
│   ├── scrape_data.py              # CLI: scrape VLR.gg → raw_matches.json
│   ├── build_dataset.py            # CLI: raw JSON → processed splits + normalizer
│   ├── train.py                    # CLI: train the model
│   └── predict.py                  # CLI: predict a single upcoming match
├── configs/
│   ├── model_config.yaml           # Architecture hyperparameters
│   └── training_config.yaml        # Training hyperparameters
├── tests/
│   ├── test_model_shapes.py        # Shape/range/param-count assertions for the model
│   ├── test_dataset.py             # Dataset + augmentation unit tests
│   ├── smoke_test.py               # End-to-end: fake data → model forward pass
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
| `scripts/scrape_data.py` | Paginates VLR.gg results, fetches each match page, saves raw match dicts to JSON. Supports checkpoint/resume. |
| `scripts/build_dataset.py` | Builds Elo ratings, 20-match history windows, extracts features, fits normalizer on train split, saves pkl/parquet splits. |
| `scripts/train.py` | Loads processed splits, trains the model with early stopping, temperature-calibrates on val set, saves checkpoint. |
| `scripts/predict.py` | Loads a checkpoint, builds sequences from match history, outputs calibrated P(team_a wins). |

### Scraper (`src/scraper/`)

| File | What it does |
|---|---|
| `vlr_scraper.py` | Core scraper. Fetches match listing pages and individual match pages from VLR.gg. Parses teams, scores, dates, tournament info, and per-map round/attack/defense/pistol stats. |
| `rate_limiter.py` | Token bucket rate limiter. Default: 0.5 req/s (1 request every 2 seconds) to stay respectful to VLR.gg. |
| `cache.py` | Saves raw HTML to `data/raw/` keyed by SHA256 of the URL. Re-running the scraper never re-fetches already-cached pages. |

### Data (`src/data/`)

| File | What it does |
|---|---|
| `feature_extractor.py` | Converts a raw match dict into 11 scalar features + a map index integer. Features include: win/loss, map scores, attack/defense win rates, pistol win rate, opponent Elo, days since match, tournament tier. |
| `dataset.py` | PyTorch `Dataset`. Loads pkl splits, pads sequences to length 20 (left-pad), returns `(scalars_a, map_idx_a, pad_mask_a, scalars_b, map_idx_b, pad_mask_b, label)`. |
| `normalizer.py` | Fits z-score (mean/std per feature) on training data. Saves params to JSON so inference uses the same scale. |
| `augmentation.py` | Wraps the base dataset to apply team-swap: each sample is doubled with A↔B swapped and the label flipped. Perfectly balances classes and doubles training data. |
| `team_filter.py` | `TEAM_ALIASES`: maps every VLR.gg display name variant to a canonical team name (handles rebrands like "Talon Esports" → "FULL SENSE"). `MASTERS_LONDON_TEAMS`: set of teams that qualified for Masters London 2026. Helper functions: `resolve_team()`, `is_franchised_match()`, `involves_masters_london_team()`. |

### Models (`src/models/`)

| File | What it does |
|---|---|
| `match_encoder.py` | Projects each match token to 64 dimensions. Looks up a 16-dim map embedding, concatenates with 11 scalar features, applies `Linear(27→64)` + `LayerNorm`, then adds learned positional embeddings. |
| `transformer.py` | `TeamEncoder`: prepends a learned `[CLS]` token, runs a 3-layer `TransformerEncoder` (d=64, heads=4, ff=256, Pre-LN, GELU, dropout=0.1), returns the CLS output as the team representation. Same weights used for both teams. |
| `classifier.py` | `ClassifierHead`: takes the concatenated team representations (128-dim), applies `Linear(128→64)→GELU→Dropout→Linear(64→32)→GELU→Dropout→Linear(32→1)`. Outputs a raw logit. |
| `predictor.py` | `ValorantPredictor`: wires together `MatchEncoder`, `TeamEncoder` (shared), and `ClassifierHead`. ~164k parameters total. |

### Training (`src/training/`)

| File | What it does |
|---|---|
| `trainer.py` | Runs train and validation loops. Uses AdamW + CosineAnnealingLR. Early stopping monitors validation Brier score (patience=20). Saves best checkpoint. |
| `losses.py` | `SmoothedBCELoss` (label smoothing ε=0.1) and `FocalLoss`. Label smoothing prevents overconfident predictions. |
| `metrics.py` | Computes accuracy, ROC-AUC, Brier score, log-loss, and ECE (Expected Calibration Error) from logits and labels. |

### Evaluation (`src/evaluation/`)

| File | What it does |
|---|---|
| `calibration.py` | `TemperatureScaler`: fits a single temperature parameter `T` on the validation set using LBFGS. Divides logits by `T` before sigmoid. Also plots reliability diagrams. |
| `baselines.py` | Three baselines to compare against: random (50%), recent win-rate predictor, and Elo-logistic regression. The real bar to clear is the Elo baseline. |

### Configs

| File | What it does |
|---|---|
| `configs/model_config.yaml` | Architecture hyperparameters: embedding dims, d_model, num_heads, num_layers, dropout, etc. |
| `configs/training_config.yaml` | Training hyperparameters: learning rate, batch size, max epochs, early stopping metric, augmentation flag. |

---

## Scrape Flags Reference

```bash
python scripts/scrape_data.py [OPTIONS]

--max-pages N          Max listing pages to paginate (default 500, ~20 matches/page)
--min-tier {0,1,2}     0=all (default)  1=VCT leagues+intl  2=Masters/Champions only
--franchised-only      Keep only matches where both teams are franchised VCT teams
--masters-london-any   Keep only matches involving at least one Masters London qualifier
--rate FLOAT           Requests per second (default 0.5)
--save-every N         Save checkpoint every N matches (default 50)
--out PATH             Output JSON path (default data/processed/raw_matches.json)
```

```bash
python scripts/build_dataset.py [OPTIONS]

--input PATH               Input JSON (default data/processed/raw_matches.json)
--min-tier {0,1,2}         Same tier filter as scraper
--franchised-only          Keep only franchised VCT team matches
--masters-london-only      Keep only matches where BOTH teams are in Masters London
--masters-london-any       Keep matches where at least one team is in Masters London
--train-cutoff DATE        Train/val split date (default 2024-09-01)
```

---

## Team Filters Explained

| Flag | Effect |
|---|---|
| `--franchised-only` | Both teams must be in the franchise partner list (VCT Americas, EMEA, Pacific, China). Filters out Challengers teams entirely. |
| `--masters-london-any` | At least one team must have qualified for Masters London 2026. Good for building a focused dataset around the teams you care about. |
| `--masters-london-only` | Both teams must have qualified for Masters London. Very few matches — mainly useful for final evaluation. |

To add a new team alias (e.g. after a rebrand), edit `src/data/team_filter.py`:
```python
TEAM_ALIASES: dict[str, str] = {
    ...
    "New Sponsor Name":  "Canonical Team Name",
    ...
}
```

---

## Train / Val / Test Split

| Split | Date range | Purpose |
|---|---|---|
| Train | before 2024-09-01 | Model training |
| Val | 2024-09-01 to 2024-12-31 | Hyperparameter tuning, early stopping, temperature calibration |
| Test | 2025-01-01 onwards | Final evaluation — touch only once |

The normalizer (z-score) is fit on the train split only. Val and test are transformed using the same train stats to prevent data leakage.

---

## Target Performance

| Metric | Random baseline | Target |
|---|---|---|
| Accuracy | 50% | >60% |
| ROC-AUC | 0.50 | >0.65 |
| Brier Score | 0.250 | <0.220 |
| Log-loss | 0.693 | <0.650 |
| ECE | — | <0.05 |
