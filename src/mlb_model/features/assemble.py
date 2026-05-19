"""Assemble the wide per-game feature table.

This is the integration point: one row per game with everything joined on
``game_pk``. The model trains and predicts on this table.

Naming convention -- every column lives in one of three namespaces:

  home_*  -- home-team-side feature (offense, defense, SP, bullpen, form)
  away_*  -- away-team-side feature
  game_*  -- game-level (park, weather, ump, schedule context, market)

Targets are appended at the end:
  target_home_win       -- bool (None for future games)
  target_total_runs     -- int  (None for future games)
  target_home_score     -- int  (None for future games)
  target_away_score     -- int  (None for future games)
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.features.bullpen import build_bullpen_features
from mlb_model.features.lineup_quality import build_lineup_quality
from mlb_model.features.market import build_market_features
from mlb_model.features.park import compute_park_factors
from mlb_model.features.platoon import build_platoon_features
from mlb_model.features.schedule_context import build_schedule_context
from mlb_model.features.sp_boxscore import build_sp_boxscore_features
from mlb_model.features.starting_pitcher import build_sp_features
from mlb_model.features.team_form import build_team_form_features
from mlb_model.logging import get_logger

log = get_logger("features.assemble")


def _games_skeleton(season_start: int, season_end: int) -> pd.DataFrame:
    """Per-game spine: one row per game with target labels."""
    sql = """
        SELECT
            g.game_pk,
            g.game_date,
            g.season,
            g.home_team_id,
            g.away_team_id,
            g.home_team_abbr,
            g.away_team_abbr,
            g.venue_id,
            g.home_score AS target_home_score,
            g.away_score AS target_away_score,
            CASE WHEN g.home_score IS NOT NULL AND g.away_score IS NOT NULL
                 THEN g.home_score + g.away_score
                 ELSE NULL END AS target_total_runs,
            g.home_win AS target_home_win,
            pp_home.pitcher_id   AS home_sp_id,
            pp_away.pitcher_id   AS away_sp_id,
            pp_home.pitcher_throws AS home_sp_throws,
            pp_away.pitcher_throws AS away_sp_throws,
            w.temp_f, w.humidity_pct, w.pressure_hpa,
            w.wind_speed_mph, w.wind_out_to_cf,
            w.is_dome, u.ump_name
        FROM games g
        LEFT JOIN probable_pitchers pp_home
               ON pp_home.game_pk = g.game_pk AND pp_home.is_home = TRUE
        LEFT JOIN probable_pitchers pp_away
               ON pp_away.game_pk = g.game_pk AND pp_away.is_home = FALSE
        LEFT JOIN weather w ON w.game_pk = g.game_pk
        LEFT JOIN umpires u ON u.game_pk = g.game_pk
        WHERE g.season BETWEEN ? AND ?
        ORDER BY g.game_date, g.game_pk
    """
    df = query(sql, params=(season_start, season_end))
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _merge_sp(skeleton: pd.DataFrame, sp_feat: pd.DataFrame, side: str) -> pd.DataFrame:
    """Merge SP features as ``{side}_sp_<feat>``."""
    if sp_feat.empty:
        return skeleton
    sp_key_col = f"{side}_sp_id"
    sp_rename = {c: f"{side}_sp_{c}" for c in sp_feat.columns if c not in ("pitcher_id", "game_pk", "game_date")}
    sp = sp_feat.rename(columns=sp_rename).rename(columns={"pitcher_id": sp_key_col})
    sp = sp.drop(columns=[c for c in ["game_date"] if c in sp.columns])
    return skeleton.merge(sp, on=["game_pk", sp_key_col], how="left")


def _merge_team_form(skeleton: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    """Merge team-form features as home_* and away_*."""
    if tf.empty:
        return skeleton
    home = tf[tf["is_home"]].copy()
    away = tf[~tf["is_home"]].copy()

    rename_home = {
        c: f"home_team_{c}"
        for c in home.columns
        if c not in ("team_id", "game_pk", "game_date", "is_home")
    }
    rename_away = {
        c: f"away_team_{c}"
        for c in away.columns
        if c not in ("team_id", "game_pk", "game_date", "is_home")
    }
    home = home.rename(columns=rename_home).rename(columns={"team_id": "home_team_id"}).drop(
        columns=["game_date", "is_home"], errors="ignore"
    )
    away = away.rename(columns=rename_away).rename(columns={"team_id": "away_team_id"}).drop(
        columns=["game_date", "is_home"], errors="ignore"
    )

    out = skeleton.merge(home, on=["game_pk", "home_team_id"], how="left")
    out = out.merge(away, on=["game_pk", "away_team_id"], how="left")
    return out


def _merge_schedule(skeleton: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    if sched.empty:
        return skeleton
    home = sched[sched["is_home"]].copy().drop(columns=["is_home"]).rename(
        columns={
            "days_rest": "home_days_rest",
            "travel_miles": "home_travel_miles",
            "is_getaway_day": "home_is_getaway_day",
            "is_doubleheader_leg2": "home_is_doubleheader_leg2",
            "team_id": "home_team_id",
        }
    ).drop(columns=["game_date"], errors="ignore")
    away = sched[~sched["is_home"]].copy().drop(columns=["is_home"]).rename(
        columns={
            "days_rest": "away_days_rest",
            "travel_miles": "away_travel_miles",
            "is_getaway_day": "away_is_getaway_day",
            "is_doubleheader_leg2": "away_is_doubleheader_leg2",
            "team_id": "away_team_id",
        }
    ).drop(columns=["game_date"], errors="ignore")

    out = skeleton.merge(home, on=["game_pk", "home_team_id"], how="left")
    out = out.merge(away, on=["game_pk", "away_team_id"], how="left")
    return out


def _merge_park(skeleton: pd.DataFrame, pf: pd.DataFrame) -> pd.DataFrame:
    if pf.empty:
        return skeleton
    return skeleton.merge(
        pf[["venue_id", "park_run_factor", "park_hr_factor", "park_k_factor"]],
        on="venue_id",
        how="left",
    )


def _merge_market(skeleton: pd.DataFrame, mkt: pd.DataFrame) -> pd.DataFrame:
    if mkt.empty:
        return skeleton
    return skeleton.merge(mkt.drop(columns=["game_date", "season"], errors="ignore"), on="game_pk", how="left")


def build_features_table(season_start: int, season_end: int) -> pd.DataFrame:
    """Assemble the full wide feature table for the season range."""
    log.info("features.assemble.start", start=season_start, end=season_end)
    skeleton = _games_skeleton(season_start, season_end)
    if skeleton.empty:
        log.warning("features.assemble.no_games")
        return skeleton

    # Statcast-based SP features (rich but only available 2015+ and requires
    # statcast ingest)
    sp_features = build_sp_features(season_start, season_end)
    if not sp_features.empty:
        skeleton = _merge_sp(skeleton, sp_features, "home")
        skeleton = _merge_sp(skeleton, sp_features, "away")

    # Boxscore-based SP features (always available; covers ~80% of Statcast signal)
    sp_box = build_sp_boxscore_features(season_start, season_end)
    if not sp_box.empty:
        # Prefix-rename so it doesn't collide with Statcast columns
        sp_box_renamed = sp_box.rename(
            columns={c: f"box_{c}" for c in sp_box.columns if c not in ("pitcher_id", "game_pk", "game_date")}
        )
        skeleton = _merge_sp(skeleton, sp_box_renamed, "home")
        skeleton = _merge_sp(skeleton, sp_box_renamed, "away")

    team_form = build_team_form_features(season_start, season_end)
    skeleton = _merge_team_form(skeleton, team_form)

    bullpen = build_bullpen_features(season_start, season_end)
    if not bullpen.empty:
        # Merge as home_bullpen_* and away_bullpen_*
        bp_cols = [c for c in bullpen.columns if c not in ("team_id", "game_pk")]
        home_bp = bullpen.rename(columns={c: f"home_bullpen_{c}" for c in bp_cols})
        home_bp = home_bp.rename(columns={"team_id": "home_team_id"})
        away_bp = bullpen.rename(columns={c: f"away_bullpen_{c}" for c in bp_cols})
        away_bp = away_bp.rename(columns={"team_id": "away_team_id"})
        skeleton = skeleton.merge(home_bp, on=["game_pk", "home_team_id"], how="left")
        skeleton = skeleton.merge(away_bp, on=["game_pk", "away_team_id"], how="left")

    platoon = build_platoon_features(season_start, season_end)
    if not platoon.empty:
        plt_cols = [c for c in platoon.columns if c not in ("team_id", "game_pk")]
        home_plt = platoon.rename(columns={c: f"home_{c}" for c in plt_cols}).rename(
            columns={"team_id": "home_team_id"}
        )
        away_plt = platoon.rename(columns={c: f"away_{c}" for c in plt_cols}).rename(
            columns={"team_id": "away_team_id"}
        )
        skeleton = skeleton.merge(home_plt, on=["game_pk", "home_team_id"], how="left")
        skeleton = skeleton.merge(away_plt, on=["game_pk", "away_team_id"], how="left")

    lineup_q = build_lineup_quality(season_start, season_end)
    if not lineup_q.empty:
        lq_cols = [c for c in lineup_q.columns if c not in ("team_id", "game_pk")]
        home_lq = lineup_q.rename(columns={c: f"home_{c}" for c in lq_cols}).rename(
            columns={"team_id": "home_team_id"}
        )
        away_lq = lineup_q.rename(columns={c: f"away_{c}" for c in lq_cols}).rename(
            columns={"team_id": "away_team_id"}
        )
        skeleton = skeleton.merge(home_lq, on=["game_pk", "home_team_id"], how="left")
        skeleton = skeleton.merge(away_lq, on=["game_pk", "away_team_id"], how="left")

    sched = build_schedule_context(season_start, season_end)
    skeleton = _merge_schedule(skeleton, sched)

    # Park factors are computed per *target* season using prior years
    park_frames = []
    for s in range(season_start, season_end + 1):
        pf = compute_park_factors(s, lookback_seasons=3)
        if not pf.empty:
            pf["season_apply"] = s
            park_frames.append(pf)
    if park_frames:
        park_all = pd.concat(park_frames, ignore_index=True)
        skeleton = skeleton.merge(
            park_all,
            left_on=["venue_id", "season"],
            right_on=["venue_id", "season_apply"],
            how="left",
        ).drop(columns=["season_apply", "through_season"], errors="ignore")

    market = build_market_features(season_start, season_end)
    skeleton = _merge_market(skeleton, market)

    # Tidy categorical columns
    skeleton["home_sp_throws"] = skeleton["home_sp_throws"].fillna("R")
    skeleton["away_sp_throws"] = skeleton["away_sp_throws"].fillna("R")
    skeleton["is_dome"] = skeleton["is_dome"].fillna(False)

    log.info(
        "features.assemble.complete",
        rows=len(skeleton),
        cols=len(skeleton.columns),
        seasons=(season_start, season_end),
    )
    return skeleton
