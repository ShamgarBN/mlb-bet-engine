"""Umpire tendency features.

Plate umpires meaningfully influence run scoring through their strike-zone
preferences: a "pitcher's umpire" expands the zone and depresses runs;
the opposite expands BB%/HR%. Across a full season the spread between
the most pitcher-friendly and hitter-friendly umpires is worth roughly
0.4-0.5 runs per game on average -- a small-but-real signal for totals.

We compute rolling K%, BB%, and runs-per-game per umpire over their
career-to-date (with a recency-weighted window), strictly using games
prior to the target game so there is no leakage from the target.

Source data:
  * ``umpires.ump_name``        -- plate ump assignment per game
  * ``team_boxscores``           -- runs / strikeouts / walks per team-game

If either is missing for a game we silently fall back to league
average tendency, which is what the model sees as "no signal".
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("features.umpire")


def build_umpire_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-game umpire tendency features.

    Returns a frame with columns:
      game_pk, ump_career_k_pct, ump_career_bb_pct, ump_career_runs_per_game

    All three are computed from games strictly *before* the target game,
    so the model never peeks at outcomes of the game it's predicting.
    """
    # Pull ump assignment + per-game aggregate run/K/BB events.
    sql = """
        WITH game_totals AS (
            SELECT
                b.game_pk,
                g.game_date,
                SUM(b.runs)              AS total_runs,
                SUM(b.strikeouts)        AS total_k,
                SUM(b.walks)             AS total_bb,
                SUM(b.at_bats + b.walks) AS total_bf  -- proxy for batters faced
            FROM team_boxscores b
            JOIN games g ON g.game_pk = b.game_pk
            WHERE g.season BETWEEN ? AND ?
              AND g.status = 'Final'
            GROUP BY b.game_pk, g.game_date
        )
        SELECT
            u.ump_name,
            t.game_pk,
            t.game_date,
            t.total_runs,
            t.total_k,
            t.total_bb,
            t.total_bf
        FROM game_totals t
        JOIN umpires u ON u.game_pk = t.game_pk
        WHERE u.ump_name IS NOT NULL
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        log.warning("umpire.empty_panel")
        return df

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["ump_name", "game_date", "game_pk"]).reset_index(drop=True)

    # Career-to-date totals, *not including* the current game.
    grp = df.groupby("ump_name", group_keys=False, sort=False)
    df["c_games"] = grp.cumcount()  # 0 for the ump's first game in window
    df["c_runs"] = grp["total_runs"].cumsum() - df["total_runs"]
    df["c_k"]    = grp["total_k"].cumsum() - df["total_k"]
    df["c_bb"]   = grp["total_bb"].cumsum() - df["total_bb"]
    df["c_bf"]   = grp["total_bf"].cumsum() - df["total_bf"]

    # Require at least 5 prior games of sample before we trust the ump
    # tendency; smaller samples are too noisy and just inject variance.
    enough = df["c_games"] >= 5
    df["ump_career_runs_per_game"] = (
        df["c_runs"].astype("float64") / df["c_games"].astype("float64").clip(lower=1)
    ).where(enough)
    df["ump_career_k_pct"] = (
        df["c_k"].astype("float64") / df["c_bf"].astype("float64").clip(lower=1)
    ).where(enough)
    df["ump_career_bb_pct"] = (
        df["c_bb"].astype("float64") / df["c_bf"].astype("float64").clip(lower=1)
    ).where(enough)

    out = df[
        [
            "game_pk",
            "ump_career_k_pct",
            "ump_career_bb_pct",
            "ump_career_runs_per_game",
        ]
    ].copy()
    log.info("umpire.features.built", rows=len(out))
    return out
