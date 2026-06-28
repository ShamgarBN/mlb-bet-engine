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

# No single MLB game is a near-certainty; clamp calibrated probabilities to a
# sane range so nothing ever reports a literal 0% / 100%.
_CLAMP_EPS = 0.02


@dataclass
class Calibrator:
    """Wraps an IsotonicRegression with a market label."""

    market: str
    regressor: IsotonicRegression

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Map raw probabilities -> calibrated probabilities.

        NaN-safe: any non-finite input passes through unchanged so the
        caller can detect "no prediction available" rather than receive a
        misleadingly confident 0/1.

        Anti-saturation guard: isotonic regression on sparse tails can snap a
        whole high-confidence region to a hard 1.0 (e.g. raw 0.82 -> 1.000),
        which surfaced as bogus "100% confidence" single-game picks. Since this
        model is overconfident throughout (the calibrator only ever shrinks raw
        probabilities toward 0.5 where it has data), we forbid calibration from
        making a prediction *more extreme* than its raw input — it may regress
        overconfidence toward a coin flip, never manufacture certainty. A small
        [eps, 1-eps] clamp is the final backstop.
        """
        p_arr = np.asarray(p, dtype=np.float64)
        finite = np.isfinite(p_arr)
        out = np.full_like(p_arr, np.nan, dtype=np.float64)
        if finite.any():
            raw = p_arr[finite]
            cal = self.regressor.predict(raw)
            # Cap the distance-from-0.5 to the raw input's (no amplification),
            # keeping the calibrator's direction.
            dev = np.minimum(np.abs(cal - 0.5), np.abs(raw - 0.5))
            side = np.where(cal >= 0.5, 1.0, -1.0)
            capped = 0.5 + side * dev
            out[finite] = np.clip(capped, _CLAMP_EPS, 1.0 - _CLAMP_EPS)
        return out


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
    # Bound the fit away from a hard 0/1 so a sparse top/bottom bin can't snap
    # the calibrator to certainty at the source. (The transform-time guard is
    # the primary defense; this is belt-and-suspenders for future retrains.)
    ir = IsotonicRegression(out_of_bounds="clip", y_min=_CLAMP_EPS, y_max=1.0 - _CLAMP_EPS)
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
