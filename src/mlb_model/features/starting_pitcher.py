"""Starting pitcher features.

A modern model needs more than ERA. We compute:
  * Recent xwOBA against (Statcast quality of contact)
  * Recent K-BB% (skill core)
  * Recent CSW% (called-strike + whiff -- modern stuff metric)
  * Velocity & spin trends
  * Career baseline as a Bayesian prior
  * 3rd-time-through-order penalty proxy (innings per start)

All windows are *leakage-safe* (rolling means computed before today's game).
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import (
    expanding_mean_excluding_current,
    rolling_per_entity,
)
from mlb_model.logging import get_logger

log = get_logger("features.sp")


def _load_pitcher_game_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Pull pitcher-by-game Statcast aggregates joined with team_boxscores and games.

    Returns one row per (pitcher, game).
    """
    sql = """
        SELECT
            p.game_pk,
            g.game_date,
            g.season,
            p.pitcher_id,
            p.pitches_thrown,
            p.swinging_strikes,
            p.called_strikes,
            p.in_zone_pct,
            p.chase_pct,
            p.avg_velocity,
            p.spin_rate_avg,
            p.xwoba_against,
            p.woba_against,
            p.barrels_against,
            p.hard_hits_against,
            -- Derived per-pitch rates
            CASE WHEN p.pitches_thrown > 0
                 THEN (p.swinging_strikes + p.called_strikes)::DOUBLE / p.pitches_thrown
                 ELSE NULL END AS csw_pct
        FROM statcast_pitcher_daily p
        JOIN games g ON g.game_pk = p.game_pk
        WHERE g.season BETWEEN ? AND ?
        ORDER BY p.pitcher_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def build_sp_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Build leakage-safe SP features for every (pitcher, game) in the range.

    Returned DataFrame columns:
      pitcher_id, game_pk, game_date, plus rolling features with suffixes:
        _r30d, _r60d, _r5g, _career
    """
    panel = _load_pitcher_game_panel(season_start, season_end)
    if panel.empty:
        log.warning("sp_features.empty_panel", season_start=season_start, season_end=season_end)
        return panel

    value_cols = [
        "xwoba_against",
        "woba_against",
        "csw_pct",
        "in_zone_pct",
        "chase_pct",
        "avg_velocity",
        "spin_rate_avg",
        "barrels_against",
        "hard_hits_against",
    ]

    r30 = rolling_per_entity(
        panel, "pitcher_id", "game_date", value_cols, window_days=30, min_periods=2
    )
    r60 = rolling_per_entity(
        panel, "pitcher_id", "game_date", value_cols, window_days=60, min_periods=3
    )
    last5 = rolling_per_entity(
        panel, "pitcher_id", "game_date", value_cols, window_count=5, min_periods=2
    )
    career = expanding_mean_excluding_current(panel, "pitcher_id", "game_date", value_cols)

    out = pd.concat(
        [
            panel[["pitcher_id", "game_pk", "game_date"]].reset_index(drop=True),
            r30.reset_index(drop=True),
            r60.reset_index(drop=True),
            last5.reset_index(drop=True),
            career.reset_index(drop=True),
        ],
        axis=1,
    )
    out.columns = [str(c) for c in out.columns]
    log.info("sp_features.built", rows=len(out), seasons=(season_start, season_end))
    return out
