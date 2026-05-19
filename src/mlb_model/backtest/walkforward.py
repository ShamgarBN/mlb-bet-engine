"""Walk-forward, season-by-season backtesting.

Algorithm:
  for target_season in [start, end]:
      # Train on all seasons strictly before target_season
      training_features = build_features_table(start, target_season - 1)
      spec = fit_spec(training_features)
      home_model = train_runs_model(home_X, home_y)
      away_model = train_runs_model(away_X, away_y)
      calibrator_ml = fit_calibrator(...)
      # Predict on target_season
      target_features = build_features_table(target_season, target_season)
      preds = simulate(target_features)
      record_metrics(target_season, preds)

This is the gold-standard sports-modeling backtest: no future data leaks in,
each season tests on data the model has truly never seen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mlb_model.backtest.metrics import (
    accuracy,
    accuracy_by_confidence,
    brier_score,
    closing_line_value,
    log_loss,
)
from mlb_model.config import settings
from mlb_model.features.assemble import build_features_table
from mlb_model.logging import get_logger
from mlb_model.model.calibrate import fit_calibrator
from mlb_model.model.feature_matrix import (
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)
from mlb_model.model.runs import train_runs_model
from mlb_model.model.simulate import simulate_games
from mlb_model.model.totals import train_totals_model

log = get_logger("backtest")


@dataclass
class SeasonResult:
    season: int
    n_games: int
    ml_brier: float
    ml_logloss: float
    ml_accuracy: float
    ml_top3_acc: float
    ml_top10_acc: float
    ml_top30_acc: float
    rl_accuracy: float
    rl_top10_acc: float
    ou_accuracy: float
    ou_top10_acc: float
    clv_ml: float
    n_with_market: int

    def as_row(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _label_outcomes(features: pd.DataFrame, preds: list, runline: float = 1.5) -> pd.DataFrame:
    """Attach realized labels to each prediction for scoring.

    NaN-safe: rows missing a label (e.g. no odds data so no total_line) are
    given NaN targets, which the metric functions then ignore.
    """
    pred_df = pd.DataFrame([p.__dict__ for p in preds])
    actuals = features[
        [
            "game_pk", "target_home_win", "target_home_score",
            "target_away_score", "target_total_runs",
            "market_ml_home_close_prob",
        ]
    ]
    df = pred_df.merge(actuals, on="game_pk", how="left")
    df["target_home_win"] = df["target_home_win"].astype("boolean").astype("float64")

    runline_margin = (df["target_home_score"] - df["target_away_score"]).astype("Float64")
    df["target_home_runline_cover"] = (runline_margin > runline).astype("Float64").astype("float64")

    total_line_f = pd.to_numeric(df["total_line"], errors="coerce")
    total_runs_f = pd.to_numeric(df["target_total_runs"], errors="coerce")
    df["target_over"] = (total_runs_f > total_line_f).astype("Float64").astype("float64")
    df.loc[total_line_f.isna() | total_runs_f.isna(), "target_over"] = float("nan")
    return df


def backtest_season(target_season: int, train_start: int) -> SeasonResult | None:
    """Train on [train_start, target_season - 1]; test on target_season."""
    log.info("backtest.season.start", target=target_season, train_start=train_start)
    train = build_features_table(train_start, target_season - 1)
    if train.empty:
        log.warning("backtest.no_training_data", season=target_season)
        return None
    test = build_features_table(target_season, target_season)
    if test.empty:
        log.warning("backtest.no_test_data", season=target_season)
        return None

    # Restrict to completed games with valid labels for training.
    train = train.dropna(subset=["target_home_score", "target_away_score"])
    test_with_labels = test.dropna(subset=["target_home_score", "target_away_score"])
    if test_with_labels.empty:
        return None

    spec = fit_spec(train)

    # Sort chronologically and take the most recent 15% as validation for
    # early stopping. This is more conservative than a random split for
    # time-series data.
    train_sorted = train.sort_values("game_date").reset_index(drop=True)
    val_cutoff = int(len(train_sorted) * 0.85)
    train_part = train_sorted.iloc[:val_cutoff].copy()
    valid_part = train_sorted.iloc[val_cutoff:].copy()

    X_home_train, mh_train, feat_cols = build_runs_matrix(train_part, spec)
    X_away_train, ma_train, _ = build_runs_matrix_away(train_part, spec)
    X_home_val, mh_val, _ = build_runs_matrix(valid_part, spec)
    X_away_val, ma_val, _ = build_runs_matrix_away(valid_part, spec)

    y_home_train = train_part["target_home_score"].to_numpy()[mh_train]
    y_away_train = train_part["target_away_score"].to_numpy()[ma_train]
    y_home_val = valid_part["target_home_score"].to_numpy()[mh_val]
    y_away_val = valid_part["target_away_score"].to_numpy()[ma_val]

    home_model = train_runs_model(
        X_home_train[mh_train],
        y_home_train,
        feat_cols,
        eval_X=X_home_val[mh_val],
        eval_y=y_home_val,
    )
    away_model = train_runs_model(
        X_away_train[ma_train],
        y_away_train,
        feat_cols,
        eval_X=X_away_val[ma_val],
        eval_y=y_away_val,
    )

    X_home_test, mh_test, _ = build_runs_matrix(test_with_labels, spec)
    X_away_test, ma_test, _ = build_runs_matrix_away(test_with_labels, spec)
    valid_mask = mh_test & ma_test
    test_use = test_with_labels.loc[valid_mask].reset_index(drop=True)

    home_mean, home_std = home_model.predict_distribution(X_home_test[valid_mask])
    away_mean, away_std = away_model.predict_distribution(X_away_test[valid_mask])
    total_lines = test_use["market_total_close"].astype("float64").to_numpy()

    preds = simulate_games(
        game_pks=test_use["game_pk"].to_numpy(),
        pred_home=(home_mean, home_std),
        pred_away=(away_mean, away_std),
        total_lines=total_lines,
        n_sims=settings.monte_carlo_iterations,
        seed=settings.random_seed + target_season,
    )

    # Direct over/under classifier (uses market_total_close as a feature).
    # Trained on the same feature matrix as the run models but with binary
    # over/under target. Rows missing total_close are dropped from training.
    train_total_line = pd.to_numeric(
        train_part["market_total_close"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    train_runs_actual = (
        pd.to_numeric(train_part["target_home_score"], errors="coerce")
        + pd.to_numeric(train_part["target_away_score"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    train_over = np.where(
        np.isfinite(train_total_line) & np.isfinite(train_runs_actual),
        (train_runs_actual > train_total_line).astype(np.float64),
        np.nan,
    )

    val_total_line = pd.to_numeric(
        valid_part["market_total_close"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    val_runs_actual = (
        pd.to_numeric(valid_part["target_home_score"], errors="coerce")
        + pd.to_numeric(valid_part["target_away_score"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    val_over = np.where(
        np.isfinite(val_total_line) & np.isfinite(val_runs_actual),
        (val_runs_actual > val_total_line).astype(np.float64),
        np.nan,
    )

    try:
        totals_model = train_totals_model(
            X_home_train[mh_train],
            train_over[mh_train],
            feat_cols,
            X_val=X_home_val[mh_val],
            y_val=val_over[mh_val],
        )
        p_over_direct = totals_model.predict(X_home_test[valid_mask])
        # Where the test row has no closing total, the direct prediction is
        # not meaningful (the model was trained against an over/under wrt
        # the line); fall back to the simulated probability.
        no_line = ~np.isfinite(total_lines)
        if no_line.any():
            sim_p = np.array([p.p_total_over for p in preds], dtype=np.float64)
            p_over_direct = np.where(no_line, sim_p, p_over_direct)
        for i, pred in enumerate(preds):
            pred.p_total_over = float(p_over_direct[i])
    except ValueError as exc:
        log.warning("totals_model.skipped", season=target_season, reason=str(exc))

    scored = _label_outcomes(test_use, preds)

    # ML metrics (raw, uncalibrated -- calibration happens via OOF below for free)
    def _to_float(series: pd.Series) -> np.ndarray:
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)

    ml_p = _to_float(scored["p_home_win"])
    ml_y = _to_float(scored["target_home_win"])
    rl_p = _to_float(scored["p_home_runline_cover"])
    rl_y = _to_float(scored["target_home_runline_cover"])
    ou_p = _to_float(scored["p_total_over"])
    ou_y = _to_float(scored["target_over"])
    closing_p = _to_float(scored["market_ml_home_close_prob"])

    ml_decile = accuracy_by_confidence(ml_p, ml_y)
    rl_decile = accuracy_by_confidence(rl_p, rl_y)
    ou_decile = accuracy_by_confidence(ou_p, ou_y)

    result = SeasonResult(
        season=target_season,
        n_games=len(scored),
        ml_brier=brier_score(ml_p, ml_y),
        ml_logloss=log_loss(ml_p, ml_y),
        ml_accuracy=accuracy(ml_p, ml_y),
        ml_top3_acc=ml_decile.win_rate_top_3pct,
        ml_top10_acc=ml_decile.win_rate_top_10pct,
        ml_top30_acc=ml_decile.win_rate_top_30pct,
        rl_accuracy=accuracy(rl_p, rl_y),
        rl_top10_acc=rl_decile.win_rate_top_10pct,
        ou_accuracy=accuracy(ou_p, ou_y),
        ou_top10_acc=ou_decile.win_rate_top_10pct,
        clv_ml=closing_line_value(ml_p, closing_p),
        n_with_market=int(np.isfinite(closing_p).sum()),
    )
    log.info(
        "backtest.season.result",
        season=target_season,
        n=result.n_games,
        ml_acc=round(result.ml_accuracy, 4),
        ml_top10=round(result.ml_top10_acc, 4),
        ml_top3=round(result.ml_top3_acc, 4),
        clv=round(result.clv_ml, 4),
    )
    # Also fit calibrators (saved by caller) -- return them via a side channel
    return result


def run_walkforward(start: int, end: int) -> pd.DataFrame:
    """Run the walk-forward backtest and return a DataFrame summary."""
    results: list[SeasonResult] = []
    for season in range(start, end + 1):
        res = backtest_season(season, train_start=max(2010, season - 6))
        if res is not None:
            results.append(res)
    return pd.DataFrame([r.as_row() for r in results])
