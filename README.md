# MLB Prediction & Betting Model

A free-data, honestly-calibrated MLB game prediction system covering moneyline,
run line, and over/under markets. Walk-forward backtested 2014–present.

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
- **SportsBookReviewsOnline** → historical closing lines (backtest only)

## Architecture

```
Raw data  →  Cleaned facts  →  Engineered features  →  Two-stage model  →  Calibrated probs  →  Picks
(DuckDB)     (DuckDB)          (Parquet feature       (Stage A: run        (isotonic)         (filtered
                                store)                 distributions;                          by edge)
                                                       Stage B: MC sim)
```

## Quick start

```bash
# 0) install deps (one-time)
uv sync --all-extras

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
```

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
notebooks/               # exploration + backtest reports
tests/
data/                    # gitignored — raw + cached pulls live here
```

## Disclaimer

This is a **probabilistic model**. Even an elite MLB model loses ~45% of its
picks. Predictions ship with measured confidence intervals. No code in this
project places real-money bets. Use at your own risk; sports betting can be
addictive and financially harmful.
