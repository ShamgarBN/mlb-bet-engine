"""Park factor features.

Park factors are computed as the *ratio of runs scored at a venue versus the
league average*, using only prior-season data so today's prediction can't peek
at outcomes happening *in* today's season at that park.

A more nuanced version of this would split by handedness (LHB vs RHB HR
factors) -- left as a TODO once we have batter-level handedness data
ingested.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("features.park")


def compute_park_factors(through_season: int, lookback_seasons: int = 3) -> pd.DataFrame:
    """Return park factors for ``through_season``, using the previous
    ``lookback_seasons`` years of completed games.

    Output columns:
      venue_id, park_run_factor, park_hr_factor, park_k_factor

    Values are *normalized* so league-average = 1.0. >1 means runs/HR/K
    are inflated at that park.
    """
    sql = """
        WITH per_game AS (
            SELECT
                g.venue_id,
                g.season,
                SUM(b.runs)        AS total_runs,
                SUM(b.home_runs)   AS total_hrs,
                SUM(b.strikeouts)  AS total_ks,
                COUNT(DISTINCT b.game_pk) AS games
            FROM team_boxscores b
            JOIN games g ON g.game_pk = b.game_pk
            WHERE g.season BETWEEN ? AND ?
              AND g.status = 'Final'
              AND g.venue_id IS NOT NULL
            GROUP BY g.venue_id, g.season
        ),
        league AS (
            SELECT
                season,
                SUM(total_runs) AS lg_runs,
                SUM(total_hrs)  AS lg_hrs,
                SUM(total_ks)   AS lg_ks,
                SUM(games)      AS lg_games
            FROM per_game
            GROUP BY season
        )
        SELECT
            p.venue_id,
            -- average runs/HR/K per (team-game) at this venue divided by league average
            AVG((p.total_runs::DOUBLE / p.games) /
                (l.lg_runs::DOUBLE / l.lg_games)) AS park_run_factor,
            AVG((p.total_hrs::DOUBLE / p.games) /
                (l.lg_hrs::DOUBLE / l.lg_games))  AS park_hr_factor,
            AVG((p.total_ks::DOUBLE / p.games) /
                (l.lg_ks::DOUBLE / l.lg_games))   AS park_k_factor
        FROM per_game p
        JOIN league l ON l.season = p.season
        GROUP BY p.venue_id
    """
    start = max(2010, through_season - lookback_seasons)
    end = through_season - 1
    df = query(sql, params=(start, end))
    df["through_season"] = through_season
    log.info("park.factors.computed", through=through_season, lookback=lookback_seasons, venues=len(df))
    return df
