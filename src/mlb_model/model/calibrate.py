"""Probability calibration using isotonic regression.

Even a strong raw model can be poorly calibrated -- e.g., it might predict
60% when the true rate at that confidence level is only 55%. Calibration
fixes this without changing the *ranking* of predictions.

We use isotonic (monotonic step-function) regression because it works well
with limited data and handles both directions of miscalibration -- unlike
Platt scaling which assumes a logistic shape.

A separate calibrator is fit per market (ML, RL, OU) since their
miscalibration patterns differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("model.calibrate")


@dataclass
class Calibrator:
    """Wraps an IsotonicRegression with a market label."""

    market: str
    regressor: IsotonicRegression

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities -> calibrated probabilities."""
        return np.clip(self.regressor.predict(p), 1e-4, 1 - 1e-4)


def fit_calibrator(
    market: str,
    raw_probs: np.ndarray,
    outcomes: np.ndarray,
) -> Calibrator:
    """Fit isotonic regression on raw_probs -> outcomes.

    Args:
        market: 'moneyline' | 'runline' | 'total'.
        raw_probs: Model-predicted probability of the "yes" event.
        outcomes: 1.0 if event occurred, 0.0 otherwise.
    """
    mask = np.isfinite(raw_probs) & np.isfinite(outcomes)
    if mask.sum() < 50:
        log.warning("calibrate.too_few_points", market=market, n=int(mask.sum()))
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(raw_probs[mask], outcomes[mask])
    return Calibrator(market=market, regressor=ir)


def save_calibrator(cal: Calibrator) -> None:
    """Save a calibrator to ``models/calibrator_<market>.joblib``."""
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    path = settings.model_dir / f"calibrator_{cal.market}.joblib"
    joblib.dump(cal, path, compress=("zlib", 3))
    log.info("calibrator.saved", market=cal.market, path=str(path))


def load_calibrator(market: str) -> Calibrator:
    path = settings.model_dir / f"calibrator_{market}.joblib"
    return joblib.load(path)
