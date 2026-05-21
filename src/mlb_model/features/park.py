"""Park factor features.

Park factors are computed as the *ratio of runs scored at a venue versus the
league average*, using only prior-season data so today's prediction can't peek
at outcomes happening *in* today's season at that park.

We additionally compute pitcher-throwing-hand splits: at parks with strong
LHP/RHP asymmetry (e.g. short porches in right field), runs/HR allowed by a
LHP differ meaningfully from RHP. This gives the model a venue-specific
matchup signal it can't get from any other column.
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

    # ------------------------------------------------------------------
    # Handedness split: runs/HR allowed by venue x opposing-SP throws.
    # ``pp.is_home = TRUE`` means the home SP threw, and the *away*
    # team's offense produced the recorded ``runs``. We thus get
    # "park runs allowed when a {L|R}HP starts for the home club".
    # Skip silently if no probable_pitchers rows exist (e.g. very old
    # seasons without the data).
    # ------------------------------------------------------------------
    hand_sql = """
        WITH per_game AS (
            SELECT
                g.venue_id,
                g.season,
                pp.pitcher_throws AS opp_sp_throws,
                SUM(b.runs)      AS total_runs,
                SUM(b.home_runs) AS total_hrs,
                COUNT(DISTINCT b.game_pk) AS games
            FROM team_boxscores b
            JOIN games g            ON g.game_pk = b.game_pk
            JOIN probable_pitchers pp
                                   ON pp.game_pk = b.game_pk
                                  AND pp.team_id != b.team_id   -- the opposing SP
            WHERE g.season BETWEEN ? AND ?
              AND g.status = 'Final'
              AND g.venue_id IS NOT NULL
              AND pp.pitcher_throws IN ('L', 'R')
            GROUP BY g.venue_id, g.season, pp.pitcher_throws
        ),
        league AS (
            SELECT
                season,
                opp_sp_throws,
                SUM(total_runs) AS lg_runs,
                SUM(total_hrs)  AS lg_hrs,
                SUM(games)      AS lg_games
            FROM per_game
            GROUP BY season, opp_sp_throws
        )
        SELECT
            p.venue_id,
            p.opp_sp_throws,
            AVG((p.total_runs::DOUBLE / p.games) /
                NULLIF(l.lg_runs::DOUBLE / l.lg_games, 0)) AS hand_run_factor,
            AVG((p.total_hrs::DOUBLE  / p.games) /
                NULLIF(l.lg_hrs::DOUBLE  / l.lg_games, 0)) AS hand_hr_factor
        FROM per_game p
        JOIN league l ON l.season = p.season AND l.opp_sp_throws = p.opp_sp_throws
        GROUP BY p.venue_id, p.opp_sp_throws
    """
    try:
        hand = query(hand_sql, params=(start, end))
    except Exception:  # noqa: BLE001
        hand = pd.DataFrame(columns=["venue_id", "opp_sp_throws", "hand_run_factor", "hand_hr_factor"])
        log.warning("park.factors.hand_split_failed", through=through_season)

    if not hand.empty:
        # Pivot wide so each row gets two new columns per hand.
        wide = hand.pivot_table(
            index="venue_id",
            columns="opp_sp_throws",
            values=["hand_run_factor", "hand_hr_factor"],
        )
        wide.columns = [f"park_{v}_vs_{h}HP".lower() for v, h in wide.columns]
        wide = wide.reset_index()
        df = df.merge(wide, on="venue_id", how="left")

    log.info(
        "park.factors.computed",
        through=through_season,
        lookback=lookback_seasons,
        venues=len(df),
        with_handedness=("park_hand_run_factor_vs_lhp" in df.columns)
            or any(c.startswith("park_hand_") for c in df.columns),
    )
    return df
