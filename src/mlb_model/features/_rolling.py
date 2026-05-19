"""Helpers for leakage-safe rolling-window stats.

Every feature in this project is computed *strictly from games that occurred
before the prediction game*. We accomplish this by shifting groups by 1 then
applying a rolling window, so the day-of game is never in its own feature.

These primitives are used by SP, lineup, bullpen, team-form, and umpire
builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_per_entity(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    value_cols: list[str],
    *,
    window_days: int | None = None,
    window_count: int | None = None,
    min_periods: int = 1,
) -> pd.DataFrame:
    """Compute leakage-safe rolling means of ``value_cols`` for each entity.

    Args:
        df: One row per (entity, observation date).
        entity_col: Grouping column (e.g., pitcher_id).
        date_col: Date column (sorted ascending within group).
        value_cols: Columns to roll.
        window_days: Time-based window (e.g., last 60 days). Mutually
            exclusive with ``window_count``.
        window_count: Count-based window (e.g., last 8 starts).
        min_periods: Minimum observations to emit a non-null value.

    Returns:
        DataFrame with the same index as ``df`` and one rolling column per
        value column, suffixed with ``_r{N}d`` or ``_r{N}g``.
    """
    if window_days is None and window_count is None:
        raise ValueError("Provide either window_days or window_count")
    if window_days is not None and window_count is not None:
        raise ValueError("Provide only one of window_days / window_count")

    df = df.sort_values([entity_col, date_col]).copy()
    df[date_col] = pd.to_datetime(df[date_col])

    out_cols: dict[str, pd.Series] = {}
    suffix = f"_r{window_days}d" if window_days else f"_r{window_count}g"

    for vc in value_cols:
        # Shift by 1 within each entity so the current row is excluded.
        shifted = df.groupby(entity_col)[vc].shift(1)

        if window_days is not None:
            # Time-based rolling requires datetime index per group.
            tmp = df[[entity_col, date_col]].copy()
            tmp["shifted_value"] = shifted
            tmp = tmp.set_index(date_col).sort_index()
            rolled = (
                tmp.groupby(entity_col)["shifted_value"]
                .rolling(f"{window_days}D", min_periods=min_periods)
                .mean()
            )
            # rolled is a multi-index (entity, date) Series -- align back
            rolled = rolled.reset_index().sort_values([entity_col, date_col])
            out_cols[f"{vc}{suffix}"] = rolled["shifted_value"].to_numpy()
        else:
            rolled = (
                shifted.groupby(df[entity_col])
                .rolling(window=window_count, min_periods=min_periods)
                .mean()
                .reset_index(level=0, drop=True)
            )
            out_cols[f"{vc}{suffix}"] = rolled.to_numpy()

    result = pd.DataFrame(out_cols, index=df.index)
    return result


def expanding_mean_excluding_current(
    df: pd.DataFrame,
    entity_col: str,
    date_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Expanding (career-to-date) mean computed up to but not including the
    current row. Useful as a long-horizon baseline for SP and batters."""
    df = df.sort_values([entity_col, date_col]).copy()
    out: dict[str, np.ndarray] = {}
    for vc in value_cols:
        shifted = df.groupby(entity_col)[vc].shift(1)
        expanding = shifted.groupby(df[entity_col]).expanding().mean()
        expanding = expanding.reset_index(level=0, drop=True)
        out[f"{vc}_career"] = expanding.to_numpy()
    return pd.DataFrame(out, index=df.index)
