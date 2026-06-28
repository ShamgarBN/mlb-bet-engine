"""Tests for the probability calibrator's anti-saturation guard.

Regression coverage for the bug where isotonic regression on a sparse top bin
mapped raw ~0.82 win probabilities to a hard 1.000 ("100% confidence" picks).
"""

from __future__ import annotations

import numpy as np

from mlb_model.model.calibrate import _CLAMP_EPS, Calibrator, fit_calibrator


def _saturating_calibrator() -> Calibrator:
    """Fit on data engineered so isotonic snaps the top bin to 1.0."""
    # Lots of mid-range games (raw ~ outcome), plus a sparse top cluster at
    # raw 0.82 that all won -> isotonic would map 0.82 -> 1.0 without the guard.
    raw = np.r_[np.linspace(0.4, 0.78, 400), np.full(6, 0.82)]
    y = np.r_[(np.linspace(0.4, 0.78, 400) > 0.5).astype(float), np.ones(6)]
    return fit_calibrator("moneyline", raw, y)


def test_no_amplification_beyond_raw():
    cal = _saturating_calibrator()
    raw = np.array([0.82, 0.85, 0.90, 0.95])
    out = cal.transform(raw)
    # Calibration may shrink toward 0.5 but never push *past* the raw input.
    assert np.all(out <= raw + 1e-9)
    # And specifically: the old bug (-> 1.0) is gone.
    assert np.all(out < 0.99)


def test_never_returns_hard_certainty():
    cal = _saturating_calibrator()
    out = cal.transform(np.array([0.999, 0.5, 0.001]))
    assert out.max() <= 1.0 - _CLAMP_EPS + 1e-9
    assert out.min() >= _CLAMP_EPS - 1e-9


class _StubReg:
    """Minimal stand-in for IsotonicRegression with a fixed mapping."""

    def __init__(self, fn):
        self.fn = fn

    def predict(self, x):
        return self.fn(np.asarray(x, dtype=float))


def test_guard_caps_amplifying_calibrator():
    # A regressor that pushes everything to 1.0 (the saturation bug) is capped
    # back to the raw input — calibration never amplifies past raw.
    cal = Calibrator(market="moneyline", regressor=_StubReg(lambda x: np.ones_like(x)))
    out = cal.transform(np.array([0.82, 0.60]))
    assert abs(out[0] - 0.82) < 1e-9
    assert abs(out[1] - 0.60) < 1e-9


def test_guard_preserves_shrinkage():
    # A regressor that halves the distance from 0.5 is *less* extreme than raw,
    # so it passes through unchanged — legitimate calibration is preserved.
    cal = Calibrator(market="moneyline", regressor=_StubReg(lambda x: 0.5 + (x - 0.5) * 0.5))
    out = cal.transform(np.array([0.80]))
    assert abs(out[0] - 0.65) < 1e-9  # 0.5 + 0.30*0.5


def test_nan_passes_through():
    cal = _saturating_calibrator()
    out = cal.transform(np.array([np.nan, 0.6]))
    assert np.isnan(out[0]) and np.isfinite(out[1])
