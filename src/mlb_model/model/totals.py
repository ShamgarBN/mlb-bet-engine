"""Direct over/under classifier.

The Monte Carlo approach to O/U probabilities under-performs in practice
because it depends on the absolute accuracy of our run predictions. A
direct binary classifier that takes ``market_total_close`` as an input
feature can learn to deviate from the line only when the rest of the
feature set says so -- which is much closer to what professional total
modelers actually do.

This module trains a LightGBM classifier on
``target = (home_score + away_score) > total_close`` and produces a
calibrated P(over). When odds are missing for the target game, we fall
back to the simulated P(over) from ``simulate_games``.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from mlb_model.logging import get_logger

log = get_logger("model.totals")


_DEFAULT_PARAMS: dict = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "verbosity": -1,
}


@dataclass
class TotalsModel:
    booster: lgb.Booster
    feature_cols: list[str]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.clip(self.booster.predict(X), 1e-6, 1 - 1e-6)


def train_totals_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    *,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    params: dict | None = None,
    num_rounds: int = 5000,
    early_stopping_rounds: int = 100,
) -> TotalsModel:
    """Train a direct O/U classifier.

    Rows where ``y`` is NaN (e.g. no closing total available for the
    game) are silently dropped from the training set.
    """
    valid = np.isfinite(y_train)
    X_train = X_train[valid]
    y_train = y_train[valid].astype(np.float64)
    if X_train.shape[0] == 0:
        raise ValueError("totals model received no rows with valid targets")

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
    valid_sets = [train_set]
    valid_names = ["train"]
    callbacks = [lgb.log_evaluation(period=0)]
    if X_val is not None and y_val is not None:
        v_mask = np.isfinite(y_val)
        if v_mask.sum() > 0:
            val_set = lgb.Dataset(
                X_val[v_mask],
                label=y_val[v_mask].astype(np.float64),
                feature_name=feature_cols,
                free_raw_data=False,
            )
            valid_sets.append(val_set)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))

    booster = lgb.train(
        params=params or _DEFAULT_PARAMS,
        train_set=train_set,
        num_boost_round=num_rounds,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    log.info("totals_model.trained", best_iter=booster.best_iteration, n=len(y_train))
    return TotalsModel(booster=booster, feature_cols=feature_cols)
