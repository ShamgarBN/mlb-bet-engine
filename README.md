# MLB Bet Engine

[![Python](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-64%20passing-brightgreen.svg)](#testing)
[![Data sources](https://img.shields.io/badge/data-100%25%20free-success.svg)](#data-sources-all-free)

A **free-data**, **honestly-calibrated** MLB game prediction system covering
moneyline, run line, and over/under markets. Walk-forward backtested
2014–present on 15,000+ games. Ships with a local desktop app (FastAPI +
HTMX + pywebview) for daily picks, season-long accuracy tracking, and an
automated end-of-season self-evaluation.

> **No real-money betting.** This is a research / hobby project. The
> code can recommend picks and grade itself; it does **not** place bets.
> See the [Disclaimer](#disclaimer).

## What makes this different

Most public MLB models either over-promise ("80% on moneyline!") or
under-deliver because they ignore the closing line as a feature.
This one is built to be:

- **Honest about ceilings.** The closing MLB line is one of the
  sharpest single-game forecasts in sports. Anything claiming
  &gt;70% on *all* games is selling you something. We publish the
  full walk-forward numbers below — including the misses.
- **Probabilistically calibrated.** Isotonic-calibrated outputs
  mean a 65%-confidence pick really wins ~65% of the time. The
  app shows a calibration curve so you can verify this yourself.
- **Coherent across markets.** ML / RL / O/U come from the *same*
  Monte Carlo simulation of the joint run distribution, so they
  agree with each other instead of contradicting.
- **Self-evaluating.** The model logs every prediction it ever makes
  to an append-only journal. At the end of each season it writes a
  full post-mortem (worst-performing slices, recommendations, model
  archive) to `reports/end_of_season_<YEAR>/`.

## Honest performance targets vs. measured results

The closing line on MLB games is one of the most efficient single-game forecasts
in all of sports. Realistic targets for an elite public-data model — and the
**actual measured results** from the 2019–2025 walk-forward backtest
(15,773 games):

| Slice | Target | **Measured (2019–25 mean)** | Status |
|---|---|---|---|
| Moneyline — all picks | 55–58% | **60.7%** | met |
| Moneyline — top 30% confidence | 60–63% | **69.2%** | met |
| Moneyline — top 10% confidence | 66–72% | **74.3%** | met |
| Moneyline — top 3% conviction | 70%+ | **75.9%** | met |
| Run line — all picks | 55%+ | **65.2%** | met |
| Run line — top 10% confidence | 70%+ | **80.7%** | met |
| Over/under — all picks | 52%+ | 52.0% | borderline |
| Closing-line value (cents) | positive | **+3.48** | met |

The model exposes confidence tiers so you can choose the trade-off between
**sample size** and **hit rate**. Anything claiming >70% on *all* games is
either lying or measuring against in-game data — physics doesn't allow it on
public information.

### Per-season detail (walk-forward, train on prior seasons only)

| Season | Games | ML Acc | ML Top-10% | ML Top-3% | RL Top-10% | OU | CLV |
|---|---|---|---|---|---|---|---|
| 2019 | 2,466 | 62.7% | 77.3% | **85.1%** | 81.8% | n/a | +1.47 |
| 2020 |   951 | 61.7% | 72.6% | 62.1%     | 76.8% | 52.3% | +4.42 |
| 2021 | 2,466 | **65.0%** | **83.0%** | **89.2%** | 84.2% | 52.5% | +3.18 |
| 2022 | 2,470 | 61.4% | 68.4% | 66.2% | **85.4%** | 51.7% | +4.28 |
| 2023 | 2,471 | 58.5% | 74.9% | **82.4%** | 77.7% | 52.8% | +4.76 |
| 2024 | 2,472 | 58.4% | 72.9% | 73.0% | 83.4% | 49.5% | +3.24 |
| 2025 | 2,477 | 57.1% | 71.0% | 73.0% | 75.8% | 53.0% | +3.00 |

`logs/backtest_v4.csv` is the source of truth; `uv run mlb-model model-card`
will render this table fresh.

## Data sources (all free)

- **pybaseball** → Statcast pitch-level (2015+), FanGraphs, Baseball-Reference
- **MLB Stats API** (official) → schedules, lineups, probable pitchers
- **Retrosheet** → historical play-by-play back to 1916
- **Baseball Savant** → expected stats (xwOBA, xBA, xSLG, EV, barrel%)
- **Open-Meteo / NOAA** → free weather (temp, wind, humidity, pressure)
- **Umpire Scorecards** → plate umpire tendencies
- **Historical closing lines** (backtest only):
  - SportsBookReviewsOnline / Kaggle dump → 2012–2021
  - [ArnavSaraogi/mlb-odds-scraper](https://github.com/ArnavSaraogi/mlb-odds-scraper)
    free JSON dataset → 2021–2025, six books (FanDuel, DraftKings, BetMGM,
    Caesars, Bet365, BetRivers), opening + closing lines
- **[The Odds API](https://the-odds-api.com)** → *live* per-game ML / RL / O/U
  for today's slate. Optional; free tier (500 credits/month, ~5 per refresh) is
  plenty for one pull a day. Add your key in-app on the **Settings** page. Without
  a key the app falls back to a season-aware league-average total line.

## What's modelled (feature set)

The model takes a wide per-game row containing:

- **Starting-pitcher form**: rolling 4-start FIP/K%/BB%/HR% (Statcast 2015+
  when available, boxscore fallback otherwise), recency-weighted, with
  handedness as a categorical feature.
- **Lineup quality & platoon**: projected lineup xwOBA vs opposing-hand
  pitchers; per-team batting form (wRC+ proxy) over 10 and 30 days.
- **Bullpen quality + availability**: rolling ERA/WHIP/K-BB% per team,
  plus IP **used in the last 1 / 3 / 7 days** so a gassed pen is
  surfaced explicitly.
- **Park factors**: 3-year rolling runs / HR / K factors, **plus a
  handedness split** (runs-vs-LHP, runs-vs-RHP, HR-vs-LHP, HR-vs-RHP)
  for parks with strong LHB/RHB asymmetry (Coors vs. Petco etc.).
- **Schedule context**: days of rest, **great-circle travel miles since
  last game**, get-away day flag, doubleheader leg 2 flag.
- **Umpire tendencies**: career K%, BB%, and runs-per-game for the
  plate ump (leakage-safe — computed strictly from prior games).
- **Weather**: temperature, humidity, pressure, wind speed, and
  **wind-out-to-CF** component (projected onto each park's compass
  bearing), pulled from the right Open-Meteo endpoint for the date
  (archive vs forecast).
- **Market features**: de-vigged moneyline implied probability, runline
  and O/U implied probabilities, line movement open→close.

Stage A models predict each team's expected run distribution
(LightGBM, mean+std). Stage B runs a Monte Carlo simulation with
correlated home/away draws (a shared lognormal "game environment"
multiplier captures shared run-scoring conditions that the marginal
models can't), an outcome-weighted extra-innings tiebreak, and reads
off the ML / RL / O/U probabilities. A separate direct LightGBM
classifier produces P(over) whenever a market line is available, and
isotonic calibrators trained on the **same** OOF distribution the
inference path produces ensure outputs are well-calibrated.

## Architecture

```
Raw data  →  Cleaned facts  →  Engineered features  →  Two-stage model  →  Calibrated probs  →  Picks
(DuckDB)     (DuckDB)          (Parquet feature       (Stage A: run        (isotonic)         (filtered
                                store)                 distributions;                          by edge)
                                                       Stage B: MC sim)
```

## Installation

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/) (a fast
Python package manager). On macOS:

```bash
# install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone + sync (installs Python 3.13 if needed)
git clone https://github.com/ShamgarBN/mlb-bet-engine.git
cd mlb-bet-engine
uv sync --all-extras

# verify everything imports + tests pass
uv run pytest
```

> **`data/` and `models/` are gitignored.** Cloning gets you the code,
> not 3.9 GB of warehouse parquet + trained model artifacts. Run the
> data-pull + train commands below to rebuild locally; everything is
> pulled fresh from public sources.

## Quick start

```bash
# 1) build the warehouse + seed venue metadata
uv run mlb-model data init

# 2) pull schedule + boxscores + weather for each season you want
uv run mlb-model data pull-range 2018 2025

# 3) drop the SBR odds dataset (76 MB JSON) into data/raw/odds_scraped/
#    and ingest it -- see scripts/backfill_odds.py for the older xlsx archive
uv run python -c "from mlb_model.data.sources.odds_sbr_json import ingest_dataset; ingest_dataset()"

# 4) walk-forward backtest the model
uv run mlb-model backtest --start 2019 --end 2025 --output logs/backtest_v4.csv
uv run mlb-model model-card --csv-path logs/backtest_v4.csv

# 5) train final production models + calibrators
uv run mlb-model train --through-season 2024 --train-start 2018

# 6) produce today's picks (probable pitchers + weather refreshed automatically)
uv run mlb-model predict --date today

# 7) or launch the local desktop app
uv run mlb-model serve
```

## Desktop app

> **Download:** grab the latest `MLB-Forecast-arm64.dmg` from the
> [Releases page](https://github.com/ShamgarBN/mlb-bet-engine/releases/latest),
> open it, and drag **MLB Forecast.app** to Applications. Apple Silicon only.

Three ways to open the app, in order of "least friction":

1. **Double-click `MLB Forecast.app`** in the repo root. macOS launches a
   native window via pywebview (WebKit). No browser tab, no terminal, no
   visible server. Drag the bundle to `/Applications` to pin it to the
   Dock.
2. **Double-click `Launch MLB Forecast.command`**. Same as above, but with
   a visible terminal so you can see logs (useful for debugging).
3. **`uv run mlb-model app`** in a terminal — identical to the .app
   bundle. Or `uv run mlb-model serve --port 8765` for a browser-tab
   experience.

Everything binds to `127.0.0.1` only — never exposed to the network.

Pages:

- `/` — **Today's slate**, ranked by model confidence, filterable by market
  (ML / RL / OU) and confidence tier. O/U picks now show for every game:
  when no closing market total is available we fall back to the
  season-aware league average and mark the line with "vs. lg avg".
  A search box filters by team abbr or pitcher name; every card has a
  one-click **Log** button that appends to your picks tracker without
  navigating away; a CSV export button on each page downloads the
  currently-visible table.
- `/game/<game_pk>` — **Game detail** with the predicted run distribution,
  pitching matchup, weather, leakage-safe feature drivers, and a "try a
  different line" input that recomputes p(over) against any total in
  real time.
- `/season` — **Ongoing model accuracy** for the current season. Every
  prediction the model has ever produced is logged to an append-only
  journal (`data/journal/predictions.parquet`), auto-graded against
  finalized scores, and rendered as per-market summaries (win rate, P/L,
  **CLV proxy** = mean edge at posting time, Brier, log-loss), a
  rolling-30-day win-rate chart, a calibration curve (does 65%-confidence
  actually win 65% of the time?), and a per-tier breakdown. The rolling
  and calibration charts only appear once enough graded games exist
  (early in a season they collapse to a one-line note instead of empty
  space). A **Retrain & analyze** button runs a fresh pull + refit +
  14-day validation in the background and reports whether the new model
  was promoted or rolled back. When an end-of-season report exists for
  the selected season, a banner links to the report's markdown. A "What
  do these numbers mean?" disclosure inlines a glossary of every metric.
- `/performance` — Walk-forward backtest table + 2026 season-to-date.
- `/log` — Your **logged picks** with stake editing, per-row delete, and
  a "clear all" button. Outcomes auto-grade once the box score finalizes;
  ROI scales by stake.
- `/settings` — One-time config for **live odds**. Paste your The Odds API
  key, click **Test key** to validate it against the API (shows remaining
  monthly credits) and **Save** to persist it. Stored locally in a `.env`
  beside the warehouse; never leaves the machine.

### Season-long tracking and end-of-season self-evaluation

The tool is designed to be useful season after season. Three pieces:

1. **Prediction journal** (`data/journal/predictions.parquet`)
   — every time the model produces a slate, a row is appended for each
   game × market. Survives morning-sync cache invalidations and model
   retraining. Read it via the `/season` page or:
   ```bash
   uv run mlb-model journal-grade               # current season
   uv run mlb-model journal-grade --season 2027 # any season in the journal
   ```

2. **Season state detection**:
   ```bash
   uv run mlb-model season-status
   ```
   Returns one of `pre_season` / `in_progress` / `ended` / `off_season`.
   The weekly retrain uses this to know when to trigger the end-of-season
   sweep.

3. **End-of-season sweep**:
   ```bash
   uv run mlb-model end-of-season --season 2026
   ```
   Once the regular season finishes, this writes a full self-evaluation
   to `reports/end_of_season_<YEAR>/`:
   - `report.md` — human-readable post-mortem with per-market performance,
     worst-performing slices (e.g. "day games went 42% / -3.7u"), and
     off-season recommendations
   - `summary.json` — diffable metrics for year-over-year comparison
   - `backtest.csv` — walk-forward results across every season trained
   - `slices.csv` — full slice breakdown
   - `model_of_record_<YEAR>/` — frozen copy of the model that produced
     this season's live predictions
   - `predictions_journal_snapshot.parquet` — frozen journal

   The weekly-train job runs this **automatically** on the first Sunday
   after the regular season ends — the user doesn't have to remember.
   It's idempotent: subsequent Sundays in the off-season see the
   existing `summary.json` and skip.

### Slumping-slugger tracker

Find the season's big power bats (15+ HR) who have gone cold, and get a
read on *why*. Cause is **verified against the official MLB transactions
feed** — a player is only labelled `INJURY (verified)` (with IL type, start
date, and the injury itself) or `EXTERNAL (verified)` (optioned / DFA /
suspended) when MLB actually logged the move. Everyone else stays honestly
`UNCLEAR`. Recent news (key-less Google News RSS, classified with
name-proximity + negation guards) is shown as **supplementary context only**
— it never drives the verdict.

Available both in the desktop app (the **Sluggers** tab — percentage cards,
a moving-percentage trend chart, and the verified-cause table with news
links) and on the CLI:

```bash
# Share of players with 15+ HR, across three denominators; appends a
# dated snapshot to data/processed/slugger_hr_pct_history.csv so the
# percentage can be tracked as a moving number through the season.
uv run mlb-model slugger percent --season 2026

# The recorded series for one denominator (all_pa | has_hr | qualified).
uv run mlb-model slugger history --season 2026 --denominator qualified

# Sluggers in a 5+ game HR drought (counting only games they appeared in),
# each with an IL-verified INJURY / EXTERNAL / UNCLEAR status + news context.
uv run mlb-model slugger slumps --season 2026 --min-drought 5
uv run mlb-model slugger slumps --season 2026 --csv-out reports/slumps.csv
uv run mlb-model slugger slumps --season 2026 --no-news   # skip the news lookup
```

The engine lives in `mlb_model.analysis` — `slugger_slump` (analysis +
plain dataclasses), `transactions` (IL/roster verification), and `news`
(supplementary headlines). The web page is served from a dated cache
(`data/cache/slugger/`); the Refresh button recomputes.

### Automated data refresh and self-improvement

Two background jobs keep the app honest:

- **Morning sync** (`mlb-model morning-sync`): pulls yesterday's finalized
  scores so the picks log can grade itself, refreshes today's schedule /
  probable pitchers / weather (including a real **forecast-endpoint**
  pull for future games — historically we only had archive weather), and
  invalidates the prediction cache, and records the day's 15+ HR share so
  the Sluggers trend chart accumulates a point per day. Footer state is
  honest: it shows the last *fully successful* run, not just "we attempted
  something", and turns amber/red when the data is stale or only partially
  synced. On macOS, if today's slate contains a premium-tier pick the sync
  emits a single Notification Center alert (deduped per day).
- **Weekly retrain** (`mlb-model weekly-train`): on Sundays, archives the
  current model files, refits on a fresh 6-year window, and validates
  the new model against the last 14 days. If the moneyline accuracy
  regresses by more than 2 pp the previous model is restored
  automatically. Up to 5 archives are kept under `models/archive/`.

Both run **lazily on app launch** (in a background thread), so you don't
need cron. To make them run unconditionally at a fixed time, install the
macOS LaunchAgents:

```bash
uv run mlb-model install-schedule      # 07:00 daily + 03:00 Sunday
uv run mlb-model install-schedule --uninstall
```

The app footer shows the last successful run of each job.

### Shipping a standalone .app / DMG to another Mac

The recommended path is a **self-contained PyInstaller bundle** — it embeds
Python, every dependency, the warehouse, and the trained models, so the target
Mac needs *nothing* installed (no uv, no Python, no clone). See
[`packaging/README.md`](packaging/README.md) for full detail. Short version:

```bash
# Build on an Apple Silicon Mac (arm64-only bundle)
uv sync --all-extras                                 # includes the `package` extra
uv run pyinstaller --noconfirm packaging/mlb_forecast.spec   # → dist/MLB Forecast.app
bash packaging/make_dmg.sh                           # → ./MLB-Forecast-arm64.dmg (project root)
```

The finished `MLB-Forecast-arm64.dmg` (~230 MB compressed) lands at the project
root. On the target Mac: open the DMG, drag **MLB Forecast** to Applications,
launch. First launch copies the bundled warehouse + models into
`~/Library/Application Support/MLB Forecast/` (a writable home that survives
reinstalls), then opens instantly thereafter. Enter your Odds API key once on
the Settings page and it persists across restarts.

> Note: the bundle is **arm64-only** by design. Build on Apple Silicon; if your
> dev Python is x86_64 (Intel Homebrew), run `uv python install 3.13 && rm -rf
> .venv && uv sync --all-extras` first to get arm64 wheels.

**Developer transfer** (source + warehouse for a second dev environment, not an
end-user install) still works via the tarball script:

```bash
./scripts/package_for_transfer.sh             # ~25 MB: source + models + warehouse
```

### Persisted artifacts

Everything stays on your machine:

```
data/cache/predictions/<YYYY-MM-DD>.parquet   # per-date predictions cache
data/cache/picks_log.parquet                  # logged picks + graded outcomes
data/cache/last_morning_sync.txt              # marker for the morning job
data/cache/last_weekly_train.txt              # marker for the weekly job
data/cache/weekly_train_status.txt            # present while a Retrain & analyze run is in flight
data/cache/weekly_train_result.json           # outcome of the last UI-triggered retrain
data/journal/predictions.parquet              # append-only model journal (every prediction ever made)
.env                                          # MLB_ODDS_API_KEY (written by the Settings page)
models/*.joblib                               # trained models + calibrators
models/archive/<timestamp>/                   # rollback copies (last 5)
reports/end_of_season_<YEAR>/                 # annual self-evaluation reports
logs/desktop_launcher.log                     # .app launcher diagnostics
```

> In a packaged .app these paths are rooted at
> `~/Library/Application Support/MLB Forecast/` instead of the repo, so your
> data and saved key persist across app reinstalls.

## Project layout

```
src/mlb_model/
    cli.py               # typer CLI entry point
    config.py            # pydantic settings, paths
    data/
        sources/         # one file per upstream source
        warehouse.py     # DuckDB-backed local store
    features/            # feature builders (SP, lineup, bullpen, park, ump…)
    model/
        runs.py          # Stage A: expected run distributions
        simulate.py      # Stage B: Monte Carlo → ML/RL/OU
        calibrate.py     # isotonic / Platt
        ensemble.py
    backtest/
        walkforward.py
        metrics.py       # Brier, log-loss, CLV, ROI by decile
    predict/             # daily prediction pipeline
    app/                 # FastAPI desktop app (templates + services + pywebview)
    automation/          # morning-sync, weekly-train, LaunchAgent installer
    journal/             # append-only prediction log + season grading
    season/              # season-state detection + end-of-season sweep
packaging/               # PyInstaller spec + entry point + make_dmg.sh (standalone .app/DMG)
notebooks/               # exploration + backtest reports
tests/
data/                    # gitignored — raw + cached pulls live here
reports/                 # end-of-season reports (gitignored; one folder per year)
```

## Testing

```bash
uv run pytest -q          # all 64 tests, ~3 seconds
uv run pytest -q -k journal  # subset by name
uv run pytest --cov       # with coverage
```

The test suite covers feature builders (rolling windows, leakage),
calibration math, the prediction journal, season state detection,
the end-of-season sweep, and an end-to-end app smoke test.

## Contributing

Issues and PRs welcome at
[github.com/ShamgarBN/mlb-bet-engine](https://github.com/ShamgarBN/mlb-bet-engine).
This is primarily a personal project — I'm a one-person shop — but
I'm happy to review fixes, new feature builders, additional data
sources, and improvements to calibration / backtest methodology.

Before opening a PR:

```bash
uv run pytest
uv run ruff check src tests
```

## Disclaimer

This is a **probabilistic model**. Even an elite MLB model loses ~45% of its
picks. Predictions ship with measured confidence intervals. **No code in this
project places real-money bets.** Use at your own risk; sports betting can be
addictive and financially harmful. If you or someone you know has a gambling
problem, call 1-800-GAMBLER (US) or visit
[ncpgambling.org](https://www.ncpgambling.org/).

## License

MIT — see [LICENSE](LICENSE).
