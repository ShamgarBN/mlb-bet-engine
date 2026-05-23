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

This is the only piece you cannot automate from public APIs. The
model uses two odds vintages:

- **2014–2021** — SportsBookReviewsOnline XLSX archive. Drop files
  named `mlb_odds_2014.xlsx` … `mlb_odds_2021.xlsx` into
  `data/raw/odds/`, then:
  ```bash
  uv run python scripts/backfill_odds.py 2014 2015 2016 2017 2018 2019 2020 2021
  ```
- **2022+** — SBR consensus JSON dump. Place the JSON files under
  `data/raw/odds_scraped/`, then:
  ```bash
  uv run python -c "from mlb_model.data.sources.odds_sbr_json import ingest_dataset; ingest_dataset()"
  ```

Without odds, the market features will be NaN; the model will still
train but the run-line and over/under markets won't have closing-line
context (the moneyline can be calibrated without it).

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
