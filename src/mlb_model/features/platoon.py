"""Team offensive splits vs LHP / RHP.

When a team that's stacked with right-handed batters faces a left-handed
starter, they typically out-perform their overall numbers. The opposite
is true for lefty-heavy lineups vs RHP. Capturing this matchup signal is
worth 0.5-1.0 percentage points of accuracy on games with extreme
handedness mismatches.

We compute trailing 60-day team runs scored (and K%/BB%/HR rate) when
facing each starter-handedness, and join the appropriate split onto the
prediction row using the *opposing* SP's handedness.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features._rolling import rolling_per_entity
from mlb_model.logging import get_logger

log = get_logger("features.platoon")


def _load_platoon_panel(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-team-per-game offense conditioned on opposing SP handedness.

    Returns one row per (team, game, vs_handedness) with team_runs_scored
    and proxy rates.
    """
    sql = """
        SELECT
            b.game_pk,
            g.game_date,
            g.season,
            b.team_id,
            -- Opposing SP handedness for this team's offense
            CASE WHEN b.is_home = TRUE THEN pp_away.pitcher_throws
                 ELSE pp_home.pitcher_throws END AS vs_sp_throws,
            b.runs       AS team_runs_scored,
            b.hits,
            b.home_runs,
            b.walks,
            b.strikeouts,
            b.at_bats
        FROM team_boxscores b
        JOIN games g ON g.game_pk = b.game_pk
        LEFT JOIN probable_pitchers pp_home ON pp_home.game_pk = b.game_pk AND pp_home.is_home = TRUE
        LEFT JOIN probable_pitchers pp_away ON pp_away.game_pk = b.game_pk AND pp_away.is_home = FALSE
        WHERE g.season BETWEEN ? AND ?
          AND g.status = 'Final'
        ORDER BY b.team_id, g.game_date
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Only keep L / R; null SP handedness is filtered out (we'll fall back
    # to overall team form for those games).
    df = df[df["vs_sp_throws"].isin(["L", "R"])].copy()
    ab = df["at_bats"].astype("float64").clip(lower=1)
    df["platoon_k_pct"]   = (df["strikeouts"].astype("float64") / ab).clip(0, 1)
    df["platoon_bb_rate"] = (df["walks"].astype("float64") / (df["at_bats"] + df["walks"]).clip(lower=1)).clip(0, 1)
    df["platoon_hr_rate"] = (df["home_runs"].astype("float64") / ab).clip(0, 1)
    return df


def build_platoon_features(season_start: int, season_end: int) -> pd.DataFrame:
    """Rolling 60-day team offensive form *split by opposing SP handedness*.

    Output columns:
      team_id, game_pk, platoon_vs_L_runs_scored, platoon_vs_R_runs_scored,
      platoon_vs_L_k_pct, platoon_vs_R_k_pct, etc.
    """
    panel = _load_platoon_panel(season_start, season_end)
    if panel.empty:
        log.warning("platoon.empty_panel")
        return panel

    value_cols = ["team_runs_scored", "platoon_k_pct", "platoon_bb_rate", "platoon_hr_rate"]

    frames: list[pd.DataFrame] = []
    for hand in ("L", "R"):
        side = panel[panel["vs_sp_throws"] == hand].copy()
        if side.empty:
            continue
        rolled = rolling_per_entity(
            side, "team_id", "game_date", value_cols, window_days=60, min_periods=3
        )
        rolled.columns = [c.replace("_r60d", f"_vs_{hand}_r60d") for c in rolled.columns]
        side_out = pd.concat(
            [side[["team_id", "game_pk"]].reset_index(drop=True), rolled.reset_index(drop=True)],
            axis=1,
        )
        frames.append(side_out)

    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    # Re-aggregate to one row per (team, game) -- a game has exactly one
    # opposing handedness, so we just collapse with first-non-null.
    grouped = combined.groupby(["team_id", "game_pk"], as_index=False).first()
    log.info("platoon.features.built", rows=len(grouped))
    return grouped
