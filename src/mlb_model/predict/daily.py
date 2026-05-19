"""Daily prediction pipeline.

Workflow for producing today's picks:
  1) Pull schedule + probable pitchers for the target date (or window).
  2) Pull (or refresh) any missing per-team/SP context from recent games.
  3) Pull weather forecast for outdoor games.
  4) Assemble feature rows for the upcoming games.
  5) Apply the latest trained models to produce ML / RL / OU probabilities.
  6) Apply the calibrators.
  7) Output a ranked picks table by confidence tier.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from mlb_model.config import settings
from mlb_model.data.pipeline import pull_date_range
from mlb_model.features.assemble import build_features_table
from mlb_model.logging import get_logger
from mlb_model.model.calibrate import load_calibrator
from mlb_model.model.feature_matrix import (
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)
from mlb_model.model.runs import load_model
from mlb_model.model.simulate import simulate_games

log = get_logger("predict.daily")


def predict_for_date(target_date: date_cls, *, refresh_data: bool = True) -> pd.DataFrame:
    """Produce calibrated picks for all games on ``target_date``.

    Returns one row per game with model and market columns plus tier flags.
    """
    season = target_date.year
    log.info("predict.start", target_date=target_date)

    if refresh_data:
        # Pull schedule for ~next 3 days so probable pitchers are populated.
        pull_date_range(target_date, target_date + timedelta(days=2))

    # Build features. We pass the whole season as training so rolling
    # windows for SP / team form / etc. include all season-to-date games.
    features = build_features_table(season, season)
    if features.empty:
        log.warning("predict.no_games", target_date=target_date)
        return features

    target_mask = features["game_date"].dt.date == target_date
    target = features[target_mask].copy()
    if target.empty:
        log.warning("predict.no_target_games", target_date=target_date)
        return target

    # Prefer the persisted FeatureSpec from training so we use the exact
    # same imputation values; fall back to fitting on the historical
    # portion if no persisted spec exists.
    spec_path = settings.model_dir / "feature_spec.joblib"
    if spec_path.exists():
        spec = joblib.load(spec_path)
    else:
        historical = features[~target_mask]
        if historical.empty:
            log.warning("predict.no_historical_features")
            return pd.DataFrame()
        spec = fit_spec(historical)

    home_model = load_model("home_runs")
    away_model = load_model("away_runs")

    X_home, mh, _ = build_runs_matrix(target, spec)
    X_away, ma, _ = build_runs_matrix_away(target, spec)
    valid = mh & ma
    rows = target.loc[valid].reset_index(drop=True)

    home_mean, home_std = home_model.predict_distribution(X_home[valid])
    away_mean, away_std = away_model.predict_distribution(X_away[valid])

    total_lines = rows["market_total_close"].astype("float64").to_numpy()
    preds = simulate_games(
        game_pks=rows["game_pk"].to_numpy(),
        pred_home=(home_mean, home_std),
        pred_away=(away_mean, away_std),
        total_lines=total_lines,
        n_sims=settings.monte_carlo_iterations,
        seed=settings.random_seed,
    )

    # Direct OU classifier if persisted. Falls back to simulated P(over).
    totals_path = settings.model_dir / "totals.joblib"
    if totals_path.exists():
        state = joblib.load(totals_path)
        booster = lgb.Booster(model_str=state["model_str"])
        p_over_direct = np.clip(booster.predict(X_home[valid]), 1e-6, 1 - 1e-6)
        no_line = ~np.isfinite(total_lines)
        if no_line.any():
            sim_p = np.array([p.p_total_over for p in preds], dtype=np.float64)
            p_over_direct = np.where(no_line, sim_p, p_over_direct)
        for i, pred in enumerate(preds):
            pred.p_total_over = float(p_over_direct[i])

    pred_df = pd.DataFrame([p.__dict__ for p in preds])

    # Apply calibrators if available
    for market_col, market in [
        ("p_home_win", "moneyline"),
        ("p_home_runline_cover", "runline"),
        ("p_total_over", "total"),
    ]:
        try:
            cal = load_calibrator(market)
            pred_df[market_col + "_raw"] = pred_df[market_col]
            pred_df[market_col] = cal.transform(pred_df[market_col].to_numpy())
        except FileNotFoundError:
            log.debug("calibrator.missing", market=market)

    # Add context columns
    output = rows[
        [
            "game_pk", "game_date", "home_team_abbr", "away_team_abbr",
            "home_sp_id", "away_sp_id",
            "market_ml_home_close_prob", "market_total_close",
        ]
    ].merge(pred_df, on="game_pk", how="left")

    # Confidence tiers
    output["confidence_ml"] = (output["p_home_win"] - 0.5).abs() * 2.0
    output["confidence_rl"] = (output["p_home_runline_cover"] - 0.5).abs() * 2.0
    output["confidence_ou"] = (output["p_total_over"] - 0.5).abs() * 2.0
    output = output.sort_values("confidence_ml", ascending=False).reset_index(drop=True)
    return output


def write_picks_csv(picks: pd.DataFrame, target_date: date_cls) -> Path:
    """Persist daily picks to a CSV for review."""
    out_dir = settings.project_root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"picks_{target_date.isoformat()}.csv"
    picks.to_csv(path, index=False)
    log.info("predict.csv.written", path=str(path), rows=len(picks))
    return path
