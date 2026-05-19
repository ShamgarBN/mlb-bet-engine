"""Bullpen quality features.

After the starter departs (typically inning 5-6), the bullpen handles the
rest. In the modern game that's ~40% of total innings. A great bullpen can
turn a 4-run lead into a near-certain win; a bad one melts down 6-run
leads. Including bullpen quality as a feature meaningfully tightens
late-game probability calibration.

Without per-inning pitch data we approximate bullpen quality by:
  * runs allowed minus runs allowed *while SP was in the game*
  * Innings pitched and walks/strikeouts by all-non-SP pitchers per team

Since our data only has team-game level boxscores, we use a proxy:
  ``team_runs_allowed - opp_starting_pitcher_earned_runs``

If we don't have SP-level data for the day, we fall back to season-to-date
team-level "runs allowed in innings 7+" (computed from box pitch counts).

For the first cut we use a simpler-but-effective construction:
  late-game run prevention proxy = team runs allowed * (1 - SP IP / total IP)

This is rough but captures the right signal direction. A future iteration
will use Statcast pitch-level data to separate SP and RP contributions
precisely.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import rolling_per_entity
from mlb_model.logging import get_logger

log = get_logger("features.bullpen")


def _load_bullpen_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-team-per-game *true* bullpen performance from pitcher_game_stats.

    Aggregates over all non-starting pitchers for the team in each game --
    this is the real bullpen output, not a proxy:
      bullpen_ip            -- total innings pitched by relievers
      bullpen_er            -- earned runs allowed by relievers
      bullpen_k             -- strikeouts by relievers
      bullpen_bb            -- walks by relievers
      bullpen_era           -- ER * 9 / IP (NaN if IP=0)
      bullpen_whip          -- (H + BB) / IP
    """
    sql = """
        SELECT
            g.game_pk,
            g.game_date,
            g.season,
            p.team_id,
            SUM(p.innings_pitched)       AS bullpen_ip,
            SUM(p.earned_runs)           AS bullpen_er,
            SUM(p.strikeouts)            AS bullpen_k,
            SUM(p.walks)                 AS bullpen_bb,
            SUM(p.hits)                  AS bullpen_hits,
            SUM(p.home_runs)             AS bullpen_hr,
            SUM(p.batters_faced)         AS bullpen_bf
        FROM pitcher_game_stats p
        JOIN games g ON g.game_pk = p.game_pk
        WHERE p.is_starter = FALSE
          AND g.season BETWEEN ? AND ?
          AND g.status = 'Final'
        GROUP BY g.game_pk, g.game_date, g.season, p.team_id
        ORDER BY p.team_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Clip IP to >=1 so a single 0.1 IP outing doesn't blow ERA up. The rolling
    # window will smooth most of the noise.
    ip = df["bullpen_ip"].astype("float64").clip(lower=1.0)
    bf = df["bullpen_bf"].astype("float64").clip(lower=3)
    df["bullpen_era"]  = (df["bullpen_er"].astype("float64") * 9.0 / ip).clip(0, 30)
    df["bullpen_whip"] = ((df["bullpen_hits"].astype("float64") + df["bullpen_bb"].astype("float64")) / ip).clip(0, 6)
    df["bullpen_k_pct"]  = (df["bullpen_k"].astype("float64")  / bf).clip(0, 1)
    df["bullpen_bb_pct"] = (df["bullpen_bb"].astype("float64") / bf).clip(0, 1)
    df["bullpen_k_per_9"]  = (df["bullpen_k"].astype("float64")  * 9.0 / ip).clip(0, 25)
    df["bullpen_bb_per_9"] = (df["bullpen_bb"].astype("float64") * 9.0 / ip).clip(0, 15)
    df["bullpen_hr_per_9"] = (df["bullpen_hr"].astype("float64") * 9.0 / ip).clip(0, 10)
    return df


def build_bullpen_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Leakage-safe bullpen-quality rolling features per (team, game)."""
    panel = _load_bullpen_panel(season_start, season_end)
    if panel.empty:
        log.warning("bullpen.empty_panel")
        return panel

    value_cols = [
        "bullpen_ip", "bullpen_er", "bullpen_era", "bullpen_whip",
        "bullpen_k_pct", "bullpen_bb_pct",
        "bullpen_k_per_9", "bullpen_bb_per_9", "bullpen_hr_per_9",
    ]
    r10 = rolling_per_entity(panel, "team_id", "game_date", value_cols, window_count=10, min_periods=3)
    r30 = rolling_per_entity(panel, "team_id", "game_date", value_cols, window_days=30, min_periods=5)

    out = pd.concat(
        [
            panel[["team_id", "game_pk"]].reset_index(drop=True),
            r10.reset_index(drop=True),
            r30.reset_index(drop=True),
        ],
        axis=1,
    )
    log.info("bullpen.features.built", rows=len(out))
    return out
