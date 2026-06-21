# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Full-stack ML application predicting Valorant professional esports match outcomes. The stack is:
- **Frontend:** Next.js 16 + React 19 + TypeScript + Tailwind CSS 4 (`frontend/`)
- **Backend API:** Flask 3.0 on port 5000 (`api/`)
- **ML Pipeline:** PyTorch Transformer model trained on match history (`src/`, `scripts/`)

## Development Commands

### Frontend
```bash
cd frontend
npm run dev       # dev server on port 3000
npm run build     # production build
npm run lint      # ESLint
```

### Backend API
```bash
pip install -r api/requirements-api.txt
python api/app.py    # Flask on port 5000
```

### ML Pipeline (run from repo root)
```bash
pip install -r requirements.txt
python scripts/scrape_data.py     # fetch matches → data/processed/raw_matches.json
python scripts/build_dataset.py   # build train/val/test splits + normalizer
python scripts/train.py           # train → checkpoints/best_model.pt
python scripts/evaluate.py        # evaluate on test set
python scripts/predict.py --team-a "Team A" --team-b "Team B"
```

### Tests
```bash
pytest tests/
```

### Docker (full stack)
```bash
docker compose up --build
```

## Architecture

### Data Flow
1. Scraper fetches match history from VLR.gg → JSON
2. `build_dataset.py` builds per-team Elo, assembles 20-match sequences, extracts 16 scalar features → train/val/test splits
3. `train.py` trains the Transformer model with early stopping on Brier score
4. Flask API loads `checkpoints/best_model.pt` + `checkpoints/temperature.json` at startup, replays all matches to rebuild Elo, then serves prediction/stats endpoints

### API Endpoints (`api/app.py`)
| Method | Path | Returns |
|--------|------|---------|
| POST | `/api/predict` | `{prob_a, prob_b, elo_a, elo_b}` |
| GET | `/api/teams` | `{Region: [team, ...]}` |
| GET | `/api/rankings` | `[{rank, team, elo, ...}]` |
| GET | `/api/team/<name>` | `{elo_history, recent_matches}` |
| GET | `/api/stats` | `{elo_dist, elo_timeline}` |
| GET | `/api/eda/full` | `{map_stats, h2h, recent_form}` |

### Frontend (`frontend/`)
- **`app/page.tsx`** — root page, renders 4 tabs: Predictor, Team Analysis, Elo Rankings, EDA
- **`components/tabs/`** — one component per tab
- **`lib/api.ts`** — typed API client; defaults `NEXT_PUBLIC_API_URL` to `http://localhost:5000`

### ML Model (`src/`)
- `src/models/` — TransformerEncoder (3 layers, 64-dim, 4 heads) encodes each team's 20-match history independently; a classifier FFNN head produces P(team_a wins)
- `src/data/` — feature extraction (16 scalars per match), dataset with team-swap augmentation, z-score normalizer
- `src/scraper/` — rate-limited VLR.gg scraper (0.5 req/s) with SHA256-keyed HTML disk cache
- `configs/model_config.yaml` and `configs/training_config.yaml` — all hyperparameters

## Important Notes

### Next.js 16 Breaking Changes
This project uses Next.js 16 (not 13/14/15). APIs, conventions, and file structure **may differ significantly** from your training data. Before writing any frontend code, read the relevant guide in `frontend/node_modules/next/dist/docs/`.

### Tailwind CSS 4
Uses `@tailwindcss/postcss` (not the old `tailwindcss` PostCSS plugin). No `tailwind.config.js` — configuration is done via CSS.

### Model Inference is Symmetric
`src/models/predictor.py` runs inference twice (A vs B and B vs A) and averages the results to ensure the model is order-invariant.

### API Startup Cost
`api/app.py` replays all historical matches at startup to rebuild Elo ratings — this takes several seconds. The API is not immediately ready on process start.

### Checkpoints Required
The Flask API requires `checkpoints/best_model.pt` and `checkpoints/temperature.json` to exist before starting. Run the full training pipeline if these are missing.
