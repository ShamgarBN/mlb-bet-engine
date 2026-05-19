"""Stage A: model expected run distributions for each team-game.

We fit two LightGBM regressors:
  * ``HomeRunsModel`` -- predicts log(home runs scored)
  * ``AwayRunsModel`` -- the same model fit on the mirrored matrix

Why log-runs? Major-league run distributions are right-skewed (most games
land in 2-7 runs, with a long tail). Modeling log(runs+1) lets a Gaussian-ish
loss approximate a count distribution and improves calibration of the
high-scoring tail.

We *also* fit a Poisson regressor on the same target for comparison and use
the better-calibrated one per market.

At prediction time we don't just emit a point estimate -- we emit a
**distribution** parameterized by (mean, sigma) so the Monte Carlo simulator
can draw realistic scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("model.runs")


_DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 2.5,
    "lambda_l1": 0.1,
    "min_gain_to_split": 0.0,
    "verbose": -1,
    "force_col_wise": True,
    "seed": settings.random_seed,
}


@dataclass
class RunsModel:
    """LightGBM model + residual stddev for distribution prediction."""

    model: Any = None
    residual_std: float = 1.5
    feature_cols: list[str] = field(default_factory=list)

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predicted run expectation in *original units* (runs)."""
        if self.model is None:
            raise RuntimeError("Model has not been trained.")
        log_pred = self.model.predict(X)
        return np.expm1(log_pred).clip(min=0.0)

    def predict_distribution(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std) for each prediction, in run units."""
        means = self.predict_mean(X)
        # Heuristic: std scales sublinearly with mean (Poisson would imply
        # std = sqrt(mean); empirically MLB scores are slightly overdispersed
        # vs Poisson so we use sqrt(mean) * 1.3 as a baseline).
        stds = np.sqrt(np.clip(means, 0.5, None)) * 1.3
        # Floor at residual_std for very low predictions.
        stds = np.maximum(stds, self.residual_std)
        return means, stds


def train_runs_model(
    X: np.ndarray,
    y: np.ndarray,
    feature_cols: list[str],
    *,
    n_estimators: int = 1200,
    early_stopping_rounds: int = 50,
    eval_X: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> RunsModel:
    """Fit a LightGBM regressor for log(runs + 1)."""
    y_log = np.log1p(np.clip(y.astype(float), 0.0, None))
    dtrain = lgb.Dataset(X, label=y_log)
    valid_sets, valid_names = None, None
    callbacks = []
    if eval_X is not None and eval_y is not None:
        deval = lgb.Dataset(eval_X, label=np.log1p(np.clip(eval_y.astype(float), 0.0, None)), reference=dtrain)
        valid_sets = [dtrain, deval]
        valid_names = ["train", "valid"]
        callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
    callbacks.append(lgb.log_evaluation(0))

    booster = lgb.train(
        _DEFAULT_LGB_PARAMS,
        dtrain,
        num_boost_round=n_estimators,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    # Residual std on training set, in log space then converted back
    train_preds = booster.predict(X)
    resids = y_log - train_preds
    # Convert log-space std to runs space approximately at the mean
    residual_std = float(np.std(resids) * np.exp(float(np.mean(train_preds))))
    log.info("runs_model.trained", n=len(y), residual_std=residual_std, best_iter=booster.best_iteration)
    return RunsModel(model=booster, residual_std=max(residual_std, 1.0), feature_cols=feature_cols)


def save_model(model: RunsModel, name: str) -> Path:
    """Persist a runs model to ``models/<name>.joblib``."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"{name}.joblib"
    joblib.dump(
        {
            "feature_cols": model.feature_cols,
            "residual_std": model.residual_std,
            "model_str": model.model.model_to_string() if model.model is not None else None,
        },
        path,
        compress=("zlib", 3),
    )
    log.info("runs_model.saved", path=str(path))
    return path


def load_model(name: str) -> RunsModel:
    """Reload a previously saved runs model."""
    path = settings.model_dir / f"{name}.joblib"
    state = joblib.load(path)
    booster = lgb.Booster(model_str=state["model_str"]) if state.get("model_str") else None
    return RunsModel(
        model=booster,
        residual_std=float(state.get("residual_std", 1.5)),
        feature_cols=list(state.get("feature_cols", [])),
    )
