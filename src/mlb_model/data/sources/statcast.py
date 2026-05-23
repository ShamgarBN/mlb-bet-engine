"""Statcast pitch-level ingestion (per-pitcher per-game aggregates).

Goes through ``pybaseball`` which scrapes Baseball Savant. Statcast
coverage starts in 2015 — earlier seasons are silently skipped.

We don't keep raw pitch rows in the warehouse (the parquet is large
and rebuilding aggregates is cheap). Instead we compute the per-game,
per-pitcher feature row directly and upsert it.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.statcast")


def _aggregate_pitcher_game(df: pd.DataFrame) -> pd.DataFrame:
    """Roll up pitch-level rows to one row per (game_pk, pitcher_id)."""
    if df is None or df.empty:
        return pd.DataFrame()

    # Coerce numerics the scraper sometimes returns as strings.
    for col in ("release_speed", "release_spin_rate", "estimated_woba_using_speedangle",
                "woba_value", "launch_speed"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived: in-zone (zone 1-9), chase (out-of-zone swing), barrels, hard-hits.
    in_zone = df.get("zone").between(1, 9) if "zone" in df.columns else pd.Series(False, index=df.index)
    is_swing = df.get("description").isin(
        {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
         "hit_into_play", "foul_bunt"}
    ) if "description" in df.columns else pd.Series(False, index=df.index)
    chase = is_swing & ~in_zone
    swinging_strike = df.get("description").isin(
        {"swinging_strike", "swinging_strike_blocked"}
    ) if "description" in df.columns else pd.Series(False, index=df.index)
    called_strike = (df.get("description") == "called_strike") if "description" in df.columns else pd.Series(False, index=df.index)
    is_barrel = (df.get("launch_speed_angle") == 6) if "launch_speed_angle" in df.columns else pd.Series(False, index=df.index)
    is_hard_hit = (df.get("launch_speed") >= 95) if "launch_speed" in df.columns else pd.Series(False, index=df.index)

    df = df.assign(
        _in_zone=in_zone.fillna(False),
        _is_swing=is_swing.fillna(False),
        _chase=chase.fillna(False),
        _ss=swinging_strike.fillna(False),
        _cs=called_strike.fillna(False),
        _barrel=is_barrel.fillna(False),
        _hard=is_hard_hit.fillna(False),
    )

    group = df.groupby(["game_pk", "pitcher"], as_index=False)
    agg = group.agg(
        pitches_thrown=("game_pk", "size"),
        swinging_strikes=("_ss", "sum"),
        called_strikes=("_cs", "sum"),
        in_zone_count=("_in_zone", "sum"),
        chase_count=("_chase", "sum"),
        swing_count=("_is_swing", "sum"),
        avg_velocity=("release_speed", "mean"),
        spin_rate_avg=("release_spin_rate", "mean"),
        xwoba_against=("estimated_woba_using_speedangle", "mean"),
        woba_against=("woba_value", "mean"),
        barrels_against=("_barrel", "sum"),
        hard_hits_against=("_hard", "sum"),
    )
    agg = agg.rename(columns={"pitcher": "pitcher_id"})
    agg["in_zone_pct"] = np.where(
        agg["pitches_thrown"] > 0,
        agg["in_zone_count"].astype(float) / agg["pitches_thrown"],
        np.nan,
    )
    out_of_zone = (agg["pitches_thrown"] - agg["in_zone_count"]).clip(lower=0)
    agg["chase_pct"] = np.where(
        out_of_zone > 0,
        agg["chase_count"].astype(float) / out_of_zone,
        np.nan,
    )
    return agg[[
        "game_pk", "pitcher_id", "pitches_thrown", "swinging_strikes",
        "called_strikes", "in_zone_pct", "chase_pct", "avg_velocity",
        "spin_rate_avg", "xwoba_against", "woba_against", "barrels_against",
        "hard_hits_against",
    ]].astype({
        "game_pk": "int64",
        "pitcher_id": "int64",
        "pitches_thrown": "int64",
        "swinging_strikes": "int64",
        "called_strikes": "int64",
        "barrels_against": "int64",
        "hard_hits_against": "int64",
    })


def ingest_season(season: int) -> int:
    """Pull every pitch in ``season`` from Baseball Savant and upsert aggregates.

    Returns the number of (pitcher, game) rows upserted. Pitch-level pulls
    are heavy: expect 700k+ rows per season. ``pybaseball`` caches the
    HTTP responses, so re-runs are cheap.
    """
    if season < settings.statcast_available_from:
        log.info("statcast.season.skipped", season=season,
                 reason="before_statcast_coverage")
        return 0
    try:
        import pybaseball
    except ImportError:
        log.warning("statcast.pybaseball.missing")
        return 0
    # Persist every Baseball Savant sub-query to ~/.pybaseball/ so a
    # crash mid-pull doesn't lose progress and re-runs are fast.
    try:
        pybaseball.cache.enable()
    except Exception:  # noqa: BLE001 -- cache is best-effort
        log.warning("statcast.cache.enable_failed")

    start = date(season, 3, 15).isoformat()
    end = date(season, 11, 15).isoformat()
    try:
        raw = pybaseball.statcast(start_dt=start, end_dt=end)
    except Exception:  # noqa: BLE001 -- pybaseball is fragile
        log.exception("statcast.fetch.failed", season=season)
        return 0
    if raw is None or raw.empty:
        return 0

    agg = _aggregate_pitcher_game(raw)
    if agg.empty:
        return 0
    return upsert_dataframe(
        agg, "statcast_pitcher_daily", key_columns=["game_pk", "pitcher_id"]
    )
