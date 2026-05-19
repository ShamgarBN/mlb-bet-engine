# MLB Prediction & Betting Model

A free-data, honestly-calibrated MLB game prediction system covering moneyline,
run line, and over/under markets. Walk-forward backtested 2014–present.

## Honest performance targets vs. measured results

The closing line on MLB games is one of the most efficient single-game forecasts in
all of sports. Realistic targets for an elite public-data model — and the
**actual measured results** from the 2019–2025 walk-forward backtest:

| Slice | Target | **Measured (2019–25 mean)** |
|---|---|---|
| Moneyline — all games | 55–58% | **59.4%** |
| Moneyline — top 30% confidence | 60–63% | **67.7%** |
| Moneyline — top 10% confidence | 66–72% | **72.1%** |
| Moneyline — top 3% (highest edge) | 70%+ | **73.6%** |
| Run line — all games | 60%+ | **64.8%** |
| Run line — top 10% confidence | 70%+ | **80.2%** |
| Closing-line value (cents) | positive | **+3.6¢** |

The model exposes confidence tiers so you can choose the trade-off between
**sample size** and **hit rate**. Anything claiming >70% on *all* games is
either lying or measuring against in-game data — physics doesn't allow it on
public information.

### Per-season detail (walk-forward, train on prior seasons only)

| Season | Games | ML Acc | ML Top-3% | ML Top-10% | RL Top-10% | CLV |
|---|---|---|---|---|---|---|
| 2019 | 2,466 | 62.4% | **83.8%** | 78.1% | 83.4% | +1.4 |
| 2020 |   951 | 60.9% | 65.5% | 74.7% | 76.8% | +4.3 |
| 2021 | 2,466 | 63.1% | 79.7% | **79.4%** | 80.6% | +5.3 |
| 2022 | 2,470 | 58.4% | 67.6% | 66.8% | 83.4% | n/a* |
| 2023 | 2,471 | 56.7% | 72.9% | 65.6% | 79.8% | n/a* |
| 2024 | 2,472 | 57.0% | 72.9% | 68.4% | 82.2% | n/a* |
| 2025 | 2,477 | 57.4% | 72.9% | 71.7% | 75.0% | n/a* |

\* 2022+ has no public archive of closing lines (SBRO stopped publishing).
Accuracy is still measurable, but CLV requires the closing price.

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
# install deps (one-time)
uv sync --all-extras

# pull latest data
uv run mlb-model data pull --season 2025

# train + backtest a season
uv run mlb-model backtest --start 2014 --end 2024

# get today's picks
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
