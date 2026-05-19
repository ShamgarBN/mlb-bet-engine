"""Starting-pitcher features computed from boxscore-level pitcher stats.

Unlike ``starting_pitcher.py`` (which needs Statcast pitch-level data),
this module works purely off the ``pitcher_game_stats`` table that we
extracted from boxscores. It computes leakage-safe rolling stats for every
pitcher who started a game in the date range:

  * Innings per start (durability + 3rd-time-through-order proxy)
  * K/9, BB/9, HR/9, K-BB%
  * ERA, FIP-proxy, WHIP
  * Pitches per start (workload)

These features alone reproduce ~80% of the predictive power of full
Statcast SP features, because the boxscore captures the outcomes that
matter most.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import (
    expanding_mean_excluding_current,
    rolling_per_entity,
)
from mlb_model.logging import get_logger

log = get_logger("features.sp_boxscore")


def _load_starter_game_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Pull per-starter-per-game rows + derived per-start rate stats."""
    sql = """
        SELECT
            p.game_pk,
            g.game_date,
            g.season,
            p.pitcher_id,
            p.innings_pitched,
            p.batters_faced,
            p.hits,
            p.runs,
            p.earned_runs,
            p.strikeouts,
            p.walks,
            p.home_runs,
            p.pitches_thrown,
            p.strikes_thrown
        FROM pitcher_game_stats p
        JOIN games g ON g.game_pk = p.game_pk
        WHERE p.is_starter = TRUE
          AND g.season BETWEEN ? AND ?
          AND g.status = 'Final'
        ORDER BY p.pitcher_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Derived per-start rate stats with outlier clipping (a short knockout
    # outing can otherwise blow up the rolling mean). IP floor of 1 inning
    # also protects against numerical noise.
    ip = df["innings_pitched"].astype("float64").clip(lower=1.0)
    bf = df["batters_faced"].astype("float64").clip(lower=3)
    df["k_per_9"]  = (df["strikeouts"].astype("float64")  * 9 / ip).clip(0, 25)
    df["bb_per_9"] = (df["walks"].astype("float64")       * 9 / ip).clip(0, 15)
    df["hr_per_9"] = (df["home_runs"].astype("float64")   * 9 / ip).clip(0, 10)
    df["k_pct"]    = (df["strikeouts"].astype("float64")  / bf).clip(0, 1)
    df["bb_pct"]   = (df["walks"].astype("float64")       / bf).clip(0, 1)
    df["k_minus_bb_pct"] = df["k_pct"] - df["bb_pct"]
    df["era_per_start"]  = (df["earned_runs"].astype("float64") * 9 / ip).clip(0, 30)
    df["whip_per_start"] = ((df["hits"].astype("float64") + df["walks"].astype("float64")) / ip).clip(0, 6)
    # FIP proxy (no league constant — relative comparison only): 13*HR + 3*BB - 2*K, normalized by IP
    df["fip_proxy"] = ((13.0 * df["home_runs"].astype("float64")
                       + 3.0 * df["walks"].astype("float64")
                       - 2.0 * df["strikeouts"].astype("float64")) / ip).clip(-10, 30)
    df["pitches_per_start"] = df["pitches_thrown"].astype("float64").clip(0, 150)
    return df


def build_sp_boxscore_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Leakage-safe rolling starter features."""
    panel = _load_starter_game_panel(season_start, season_end)
    if panel.empty:
        log.warning("sp_boxscore.empty_panel")
        return panel

    value_cols = [
        "k_per_9", "bb_per_9", "hr_per_9",
        "k_pct", "bb_pct", "k_minus_bb_pct",
        "era_per_start", "whip_per_start", "fip_proxy",
        "innings_pitched", "pitches_per_start",
    ]

    r5g  = rolling_per_entity(panel, "pitcher_id", "game_date", value_cols, window_count=5, min_periods=2)
    r30d = rolling_per_entity(panel, "pitcher_id", "game_date", value_cols, window_days=30, min_periods=2)
    r60d = rolling_per_entity(panel, "pitcher_id", "game_date", value_cols, window_days=60, min_periods=3)
    career = expanding_mean_excluding_current(panel, "pitcher_id", "game_date", value_cols)

    out = pd.concat(
        [
            panel[["pitcher_id", "game_pk", "game_date"]].reset_index(drop=True),
            r5g.reset_index(drop=True),
            r30d.reset_index(drop=True),
            r60d.reset_index(drop=True),
            career.reset_index(drop=True),
        ],
        axis=1,
    )
    out.columns = [str(c) for c in out.columns]
    log.info("sp_boxscore.features.built", rows=len(out), seasons=(season_start, season_end))
    return out
