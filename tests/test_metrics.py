"""Unit tests for backtest metrics.

Validates the math on Brier, log loss, CLV, and the confidence-decile
report. These metrics are load-bearing for the headline "realistic
targets" numbers in the README, so they need to be right.
"""

from __future__ import annotations

import math

import numpy as np

from mlb_model.backtest.metrics import (
    accuracy,
    accuracy_by_confidence,
    american_to_decimal,
    brier_score,
    closing_line_value,
    log_loss,
    settle_unit,
)


def test_brier_perfect() -> None:
    p = np.array([1.0, 0.0, 1.0, 0.0])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(p, y) == 0.0


def test_brier_worst() -> None:
    p = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    assert math.isclose(brier_score(p, y), 1.0)


def test_log_loss_zero_for_correct_certain() -> None:
    p = np.array([1 - 1e-9, 1e-9])
    y = np.array([1.0, 0.0])
    assert log_loss(p, y) < 1e-6


def test_accuracy_threshold() -> None:
    p = np.array([0.6, 0.4, 0.51, 0.49])
    y = np.array([1.0, 0.0, 1.0, 0.0])
    assert accuracy(p, y) == 1.0


def test_american_to_decimal_positive() -> None:
    assert math.isclose(american_to_decimal(150), 1 + 150 / 100)


def test_american_to_decimal_negative() -> None:
    assert math.isclose(american_to_decimal(-200), 1 + 100 / 200)


def test_settle_unit_winner() -> None:
    # Bet $1 at +200, win => profit $2.
    assert math.isclose(settle_unit(1.0, 200, True), 2.0)


def test_settle_unit_loser() -> None:
    # Bet $1 at -200, lose => -$1.
    assert math.isclose(settle_unit(1.0, -200, False), -1.0)


def test_clv_positive_when_model_beats_close() -> None:
    # We say home is 60%, market closes at 50% => +10pp CLV.
    model = np.array([0.60, 0.55])
    close = np.array([0.50, 0.50])
    clv = closing_line_value(model, close)
    assert clv > 0


def test_decile_report_sorts_by_confidence() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0.0, 1.0, size=500)
    y = (p > rng.uniform(0.0, 1.0, size=500)).astype(float)
    report = accuracy_by_confidence(p, y, n_bins=10)
    assert len(report.accuracies) == 10
    assert 0.0 <= report.win_rate_top_10pct <= 1.0
