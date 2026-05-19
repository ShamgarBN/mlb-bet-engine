"""Metric helpers for binary classification + betting ROI.

Includes Brier score, log loss, accuracy, accuracy-by-confidence-decile,
ROI (American odds), and closing-line value (CLV).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core classification metrics
# ---------------------------------------------------------------------------

def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted prob and binary outcome."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    return float(np.mean((p[mask] - y[mask]) ** 2))


def log_loss(p: np.ndarray, y: np.ndarray, *, eps: float = 1e-9) -> float:
    """Binary cross-entropy."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def accuracy(p: np.ndarray, y: np.ndarray, *, threshold: float = 0.5) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    preds = (p[mask] >= threshold).astype(float)
    return float((preds == y[mask]).mean())


@dataclass
class DecileReport:
    """Accuracy by confidence decile (most useful real-world metric)."""

    decile_labels: list[str]
    counts: list[int]
    accuracies: list[float]
    mean_confidence: list[float]
    win_rate_top_3pct: float
    win_rate_top_10pct: float
    win_rate_top_30pct: float


def accuracy_by_confidence(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> DecileReport:
    """Bin predictions by |p - 0.5| and report accuracy in each bin.

    Higher decile = higher confidence (more lopsided prediction).
    """
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if len(p) == 0:
        return DecileReport([], [], [], [], 0.0, 0.0, 0.0)

    confidence = np.abs(p - 0.5) * 2.0  # in [0, 1]
    quantiles = np.quantile(confidence, np.linspace(0, 1, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    # Pick the side the model would actually bet (>=0.5 = "yes")
    chosen = (p >= 0.5).astype(float)
    correct = (chosen == y).astype(float)

    counts, accs, confs, labels = [], [], [], []
    for i in range(n_bins):
        in_bin = (confidence > quantiles[i]) & (confidence <= quantiles[i + 1])
        n = int(in_bin.sum())
        counts.append(n)
        if n == 0:
            accs.append(float("nan"))
            confs.append(float("nan"))
        else:
            accs.append(float(correct[in_bin].mean()))
            confs.append(float(confidence[in_bin].mean()))
        labels.append(f"D{i+1}")

    sorted_conf = np.argsort(-confidence)  # descending
    def top_pct_acc(pct: float) -> float:
        k = max(1, int(round(len(p) * pct)))
        idx = sorted_conf[:k]
        return float(correct[idx].mean())

    return DecileReport(
        decile_labels=labels,
        counts=counts,
        accuracies=accs,
        mean_confidence=confs,
        win_rate_top_3pct=top_pct_acc(0.03),
        win_rate_top_10pct=top_pct_acc(0.10),
        win_rate_top_30pct=top_pct_acc(0.30),
    )


# ---------------------------------------------------------------------------
# Betting math
# ---------------------------------------------------------------------------

def american_to_decimal(price: float) -> float:
    if pd.isna(price):
        return float("nan")
    p = float(price)
    return 1 + (p / 100.0) if p > 0 else 1 + (100.0 / -p)


def settle_unit(stake: float, price: float, won: bool) -> float:
    """Profit (or loss) of a one-unit bet on ``price`` (American)."""
    if pd.isna(price):
        return 0.0
    if won:
        return stake * (american_to_decimal(price) - 1.0)
    return -stake


def roi(
    picks: pd.DataFrame,
    *,
    pick_col: str,            # 0/1 indicator of which side we bet
    actual_col: str,          # 0/1 outcome
    price_home_col: str,
    price_away_col: str,
    stake: float = 1.0,
) -> float:
    """Compute realized ROI on a slate of bets.

    ``pick_col`` = 1 means we backed home; ``pick_col`` = 0 means away.
    """
    df = picks.copy()
    df["price"] = np.where(df[pick_col] == 1, df[price_home_col], df[price_away_col])
    df["won"] = (df[pick_col] == df[actual_col])
    df["pnl"] = df.apply(lambda r: settle_unit(stake, r["price"], bool(r["won"])), axis=1)
    if len(df) == 0:
        return 0.0
    return float(df["pnl"].sum() / (stake * len(df)))


def closing_line_value(
    model_prob: np.ndarray,
    closing_prob: np.ndarray,
) -> float:
    """Mean CLV in percentage points -- positive means model is sharper than close.

    For each game we take the side with the larger model probability,
    compare model_prob vs closing_prob for that same side, and average
    the percentage-point delta. Long-run CLV is the single best leading
    indicator of profitability.
    """
    # Coerce both arrays to numeric float; None / missing become NaN.
    model_prob = pd.to_numeric(pd.Series(model_prob), errors="coerce").to_numpy(dtype=np.float64)
    closing_prob = pd.to_numeric(pd.Series(closing_prob), errors="coerce").to_numpy(dtype=np.float64)
    mask = np.isfinite(model_prob) & np.isfinite(closing_prob)
    if not mask.any():
        return float("nan")
    mp, cp = model_prob[mask], closing_prob[mask]
    side_home = mp >= 0.5
    model_side_prob = np.where(side_home, mp, 1 - mp)
    closing_side_prob = np.where(side_home, cp, 1 - cp)
    return float(np.mean(model_side_prob - closing_side_prob) * 100.0)
