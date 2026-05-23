# Recovery notes

This project lost its `src/mlb_model/data/` subpackage, its DuckDB
warehouse, and its trained models when the original development laptop
became unavailable. The source code has been reconstructed; the
warehouse and models need to be rebuilt from public data.

## What went wrong

The `.gitignore` had this rule:

```
data/
```

Without a leading slash, that pattern matches **any** directory named
`data` — including `src/mlb_model/data/` (real source code), not just
the intended top-level `data/` directory (raw datasets, never committed
on purpose). The subpackage was never pushed to GitHub.

Fixed by anchoring the rule to `/data/` (commit on `main`).

## Path back to a working app

Run these in order. Each step is idempotent — safe to interrupt and
resume.

### 1. Install the macOS LightGBM dependency

LightGBM links against `libomp.dylib`, which doesn't ship with
Homebrew's Python or with the LightGBM wheel. Without it,
`mlb-model predict` and `mlb-model train` will crash on first model
load. The app's UI catches the import error and degrades gracefully
to "no predictions yet", but you need this fixed before training.

If you see a Homebrew permission warning:

```bash
sudo chown -R $(whoami) /usr/local/share/man/man8
```

Then:

```bash
brew install libomp
```

### 2. Verify the test suite

Should pass all 64 tests. If you see `ModuleNotFoundError: No module
named 'mlb_model'`, macOS has marked the editable-install `.pth`
file as hidden (iCloud Desktop sync does this for files in
`~/Desktop`). Clear it with:

```bash
chflags nohidden .venv/lib/python3.13/site-packages/*.pth
uv run pytest
```

If you want to avoid the periodic re-hiding, move the project out of
`~/Desktop` to e.g. `~/Projects/`.

### 3. Rebuild the warehouse

The schema is created automatically on any query, but you'll want
seeded venue metadata too:

```bash
uv run mlb-model data init
```

Then pull every season you want to train on. Each year takes ~15-30
minutes (rate-limited HTTP to MLB Stats API and Baseball Savant);
runs in the background fine.

```bash
# 2018-2025 = ~12,000 games + Statcast pitch-level
uv run mlb-model data pull-range 2018 2025
```

If you want to skip the slow Statcast pull on the first pass (so you
can see the rest finish quickly and verify), add `--no-statcast`.
You can fill it in later season-by-season.

### 4. Bring odds data back

Two complementary sources cover historical and live:

#### Historical (2014–present) via Kaggle CSV

The project no longer depends on SportsBookReviewsOnline. Instead, the
ingester accepts any reasonable CSV/XLSX with one row per game (or
two rows per game in V/H pair format). Auto-detects common column
layouts.

Drop a CSV into `data/raw/odds/`, then:

```bash
uv run python scripts/backfill_odds.py
```

Where to get the CSV (any of these work — the ingester is forgiving):

* Kaggle, search **"MLB betting odds"** — multiple users have
  uploaded multi-year archives (e.g. *"MLB Baseball Game Odds &
  Results 2010–2024"*).
* Sports Book Review's legacy XLSX format is also supported if you
  have your own archive of it.
* OpenSports or similar community archives.

The filename becomes the `book` label, so `mlb_odds_2014.xlsx` lands
as `book = 'csv_mlb_odds_2014'`. If you want all your CSVs treated
as the same source, pass `--book consensus_close`:

```bash
uv run python scripts/backfill_odds.py data/raw/odds/ --book consensus_close
```

When the auto-detector picks the wrong column, you can override:

```python
from mlb_model.data.sources.odds_csv import ingest_csv
ingest_csv(
    "data/raw/odds/my_weird_format.csv",
    column_overrides={"date": "GameDay", "home_team": "HomeName"},
)
```

#### Live + going forward via The Odds API

For current-day slates and ongoing line tracking, the project ships
a client for [the-odds-api.com](https://the-odds-api.com/) (free
tier: 500 requests/month — plenty for one daily call).

1. Sign up, copy your key.
2. Set `MLB_ODDS_API_KEY` in your environment (or a `.env` file at
   the project root).
3. The morning-sync job will pick up live odds automatically once
   the key is set; or pull on demand:
   ```bash
   uv run python -c "from mlb_model.data.sources.odds_api import ingest_live_slate; print(ingest_live_slate())"
   ```

Live rows land with `book = 'odds_api'`. When both a CSV (historical)
and an Odds API row exist for the same game, the `features/market.py`
priority picks `consensus_*` first, then `odds_api`, then everything
else alphabetically — so the labels you choose for your CSV imports
matter for tie-breaks.

Without any odds, the market features (`market_ml_*`,
`market_total_close`, `market_runline_*`) will be NaN and the model
trains around them. The moneyline can still be calibrated; run-line
and over/under will be weaker without the closing-line signal.

### 5. Backtest and train

```bash
uv run mlb-model backtest --start 2019 --end 2025 --output logs/backtest_v4.csv
uv run mlb-model model-card --csv-path logs/backtest_v4.csv
uv run mlb-model train --through-season 2024 --train-start 2018
```

The trained artifacts land in `models/*.joblib`.

### 6. Launch the app

```bash
uv run mlb-model app          # native window (pywebview)
uv run mlb-model serve        # browser tab
```

Or double-click the `Launch MLB Forecast.command` file in the project
root.

## Reconstruction caveats

The reconstructed source is faithful to every observed call site but
some details are best-effort approximations:

- **`venues_seed.py::cf_bearing_deg`** — I used reasonable public
  values (~0° = north, 45° = NE, etc.). The original may have been
  measured more precisely. If you find a park whose wind feature
  feels miscalibrated, that's the column to update.
- **`mlb_statsapi.normalize_*`** — wraps the live MLB-StatsAPI
  library. `plate_appearances` is approximated as `AB + BB` (HBP
  and sacrifice flies are missing from the lite boxscore endpoint).
  `batters_faced` for pitchers is left as NULL — the lite endpoint
  doesn't include it either.
- **`statcast.py`** — aggregates pitch-level rows from `pybaseball`
  into per-(game, pitcher) features. The exact set of derived
  features (in-zone %, chase %, barrels) matches what the rest of
  the codebase expects, but if the original had additional features
  they're not here.
- **`pipeline.py`** — orchestration is correct but the original may
  have had finer-grained progress reporting, per-source error
  recovery, or specific rate-limit handling that this version
  doesn't replicate.

If any of these matter for production accuracy, the place to look is
the SQL query in `features/*.py` that consumes the column —
correcting the source-side population is straightforward once you
know what the consumer expects.

## What did not survive

- The original `models/*.joblib` artifacts. You must retrain.
- The original DuckDB warehouse contents. The schema is reconstructed,
  but you must re-pull every game.
- The prediction journal (`data/journal/predictions.parquet`). The
  model starts a fresh journal on first prediction.
- End-of-season reports for prior years. The most recent backtest
  CSV in `logs/` may also be missing if it was gitignored.

## Don't let this happen again

The `/data/` anchored pattern in `.gitignore` is the actual fix. A
secondary recommendation: don't keep this project in `~/Desktop` if
you have iCloud Desktop & Documents sync turned on — iCloud's hidden
flag handling can break the editable install (`.pth` files become
"hidden" and Python 3.13's `site.py` skips them silently). Either
disable iCloud Desktop sync, or move the project to `~/Projects/`.
