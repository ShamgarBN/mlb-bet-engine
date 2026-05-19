"""Lineup quality features.

Aggregates the *starting nine* per game into team-level offensive
attributes using leakage-safe rolling stats per batter. The pipeline:

  1. For each (batter, game) row in batter_game_stats, compute rolling
     30-day rates (K%, BB%, ISO, OPS-proxy).
  2. Join those rolling rates onto the *lineup* table by (game_pk,
     batter_id).
  3. Aggregate to one row per (team, game) by averaging across the 9
     starting batters.

The result is a team-level "today's lineup quality" feature, sensitive to
who's actually playing (vs. the team-form features which use the team's
overall recent results regardless of lineup).
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import (
    expanding_mean_excluding_current,
    rolling_per_entity,
)
from mlb_model.logging import get_logger

log = get_logger("features.lineup_quality")


def _load_batter_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-batter-per-game panel with derived rate stats."""
    sql = """
        SELECT
            b.game_pk,
            g.game_date,
            g.season,
            b.batter_id,
            b.team_id,
            b.at_bats,
            b.plate_appearances,
            b.hits,
            b.doubles,
            b.triples,
            b.home_runs,
            b.walks,
            b.strikeouts,
            b.hbp,
            b.total_bases
        FROM batter_game_stats b
        JOIN games g ON g.game_pk = b.game_pk
        WHERE g.season BETWEEN ? AND ?
          AND g.status = 'Final'
          AND b.plate_appearances IS NOT NULL
          AND b.plate_appearances > 0
        ORDER BY b.batter_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    pa = df["plate_appearances"].astype("float64").clip(lower=1)
    ab = df["at_bats"].astype("float64").clip(lower=1)
    df["k_rate"]   = (df["strikeouts"].astype("float64") / pa).clip(0, 1)
    df["bb_rate"]  = (df["walks"].astype("float64") / pa).clip(0, 1)
    df["hr_rate"]  = (df["home_runs"].astype("float64") / pa).clip(0, 1)
    df["iso"]      = ((df["total_bases"].fillna(0).astype("float64") - df["hits"].astype("float64")) / ab).clip(0, 2)
    # Approx OBP and SLG from boxscore counts:
    df["obp_proxy"] = ((df["hits"] + df["walks"] + df["hbp"].fillna(0)).astype("float64") / pa).clip(0, 1)
    df["slg_proxy"] = (df["total_bases"].fillna(0).astype("float64") / ab).clip(0, 4)
    df["ops_proxy"] = (df["obp_proxy"] + df["slg_proxy"]).clip(0, 5)
    return df


def build_lineup_quality(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-(team, game) lineup-aggregated rolling quality stats.

    Output columns:
      team_id, game_pk, lineup_avg_k_rate_r30d, lineup_avg_bb_rate_r30d,
      lineup_avg_iso_r30d, lineup_avg_ops_proxy_r30d, lineup_max_iso_r30d,
      lineup_n_active_batters
    """
    panel = _load_batter_panel(season_start, season_end)
    if panel.empty:
        log.warning("lineup_quality.empty_panel")
        return panel

    value_cols = ["k_rate", "bb_rate", "hr_rate", "iso", "obp_proxy", "slg_proxy", "ops_proxy"]
    r30 = rolling_per_entity(
        panel, "batter_id", "game_date", value_cols, window_days=30, min_periods=3
    )
    career = expanding_mean_excluding_current(panel, "batter_id", "game_date", value_cols)
    rolled = pd.concat(
        [
            panel[["batter_id", "game_pk", "team_id"]].reset_index(drop=True),
            r30.reset_index(drop=True),
            career.reset_index(drop=True),
        ],
        axis=1,
    )

    # Pull the actual starting lineup (batting_order 1..9). If lineup table is
    # incomplete for a game we fall back to all batters with PAs.
    lineup_sql = """
        SELECT game_pk, team_id, player_id AS batter_id
        FROM lineups
        WHERE batting_order BETWEEN 1 AND 9
    """
    lineups = query(lineup_sql)
    if lineups.empty:
        return pd.DataFrame()

    merged = lineups.merge(rolled, on=["game_pk", "team_id", "batter_id"], how="left")

    agg_cols = [c for c in merged.columns if c.endswith(("_r30d", "_career"))]
    grouped = (
        merged.groupby(["team_id", "game_pk"], as_index=False)[agg_cols]
        .mean()
        .rename(columns={c: f"lineup_avg_{c}" for c in agg_cols})
    )
    # Power bat indicator: max ISO in the lineup
    max_iso = (
        merged.groupby(["team_id", "game_pk"])["iso_r30d"]
        .max()
        .reset_index()
        .rename(columns={"iso_r30d": "lineup_max_iso_r30d"})
    )
    grouped = grouped.merge(max_iso, on=["team_id", "game_pk"], how="left")
    # How many of the 9 had rolling data
    valid_count = (
        merged.assign(_has=merged["k_rate_r30d"].notna().astype(int))
        .groupby(["team_id", "game_pk"])["_has"]
        .sum()
        .reset_index()
        .rename(columns={"_has": "lineup_n_active_batters"})
    )
    grouped = grouped.merge(valid_count, on=["team_id", "game_pk"], how="left")
    log.info("lineup_quality.built", rows=len(grouped))
    return grouped
