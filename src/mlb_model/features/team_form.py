"""Team-level offensive and defensive form features.

Rolling and expanding rates of runs scored / runs allowed at the team level,
computed from ``team_boxscores`` and ``games``. The output is one row per
(team, game) with leakage-safe rolling features ready to be joined onto the
game-level feature row.

Why this works: while pitcher and bullpen specifics carry most of the signal
in MLB, team offensive form (lineup health, bench depth, recent BABIP luck)
adds 1-2 percentage points of explanatory power on top.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import (
    expanding_mean_excluding_current,
    rolling_per_entity,
)
from mlb_model.logging import get_logger

log = get_logger("features.team_form")


def _load_team_game_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-team-per-game offense + opponent runs (= defense for that game)."""
    sql = """
        SELECT
            b.game_pk,
            g.game_date,
            g.season,
            b.team_id,
            b.is_home,
            b.runs        AS runs_scored,
            opp.runs      AS runs_allowed,
            b.hits,
            b.home_runs,
            b.walks,
            b.strikeouts,
            b.at_bats
        FROM team_boxscores b
        JOIN team_boxscores opp
          ON opp.game_pk = b.game_pk AND opp.team_id != b.team_id
        JOIN games g ON g.game_pk = b.game_pk
        WHERE g.season BETWEEN ? AND ?
          AND g.status = 'Final'
        ORDER BY b.team_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]
    df["k_pct"] = df["strikeouts"] / df["at_bats"].replace(0, pd.NA)
    df["bb_pct"] = df["walks"] / (df["at_bats"] + df["walks"]).replace(0, pd.NA)
    df["iso_proxy"] = df["home_runs"] / df["at_bats"].replace(0, pd.NA)  # crude ISO proxy
    return df


def build_team_form_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Build leakage-safe team-form features."""
    panel = _load_team_game_panel(season_start, season_end)
    if panel.empty:
        log.warning("team_form.empty_panel")
        return panel

    value_cols = [
        "runs_scored",
        "runs_allowed",
        "run_diff",
        "hits",
        "home_runs",
        "walks",
        "strikeouts",
        "k_pct",
        "bb_pct",
        "iso_proxy",
    ]

    r10 = rolling_per_entity(panel, "team_id", "game_date", value_cols, window_count=10, min_periods=3)
    r30 = rolling_per_entity(panel, "team_id", "game_date", value_cols, window_days=30, min_periods=5)
    r60 = rolling_per_entity(panel, "team_id", "game_date", value_cols, window_days=60, min_periods=8)
    season_career = expanding_mean_excluding_current(panel, "team_id", "game_date", value_cols)

    out = pd.concat(
        [
            panel[["team_id", "game_pk", "game_date", "is_home"]].reset_index(drop=True),
            r10.reset_index(drop=True),
            r30.reset_index(drop=True),
            r60.reset_index(drop=True),
            season_career.reset_index(drop=True),
        ],
        axis=1,
    )
    out.columns = [str(c) for c in out.columns]
    log.info("team_form.features.built", rows=len(out), seasons=(season_start, season_end))
    return out
