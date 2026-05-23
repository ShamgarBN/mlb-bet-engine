"""Pipeline orchestration: pull from upstream sources, write to the warehouse.

Two public entry points:

* :func:`pull_date_range` — fast path for the daily / morning sync.
  Schedule + boxscores + probable pitchers for a small window.
* :func:`pull_season` — full backfill for one season. Includes Statcast
  daily summaries (heavy) and weather (best-effort).

Both are idempotent (every write goes through ``upsert_dataframe``) and
return ``{table_name: rows_written}`` dicts so callers can log progress.

:func:`ensure_venues` seeds the static venues table from
``venues_seed.py``; safe to call repeatedly.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

import pandas as pd

from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.venues_seed import seed_venues_df
from mlb_model.data.warehouse import init_schema, upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.pipeline")


def ensure_venues() -> int:
    """Seed the venues table from the static metadata file."""
    init_schema()
    df = seed_venues_df()
    if df.empty:
        return 0
    return upsert_dataframe(df, "venues", key_columns=["team_abbr"])


def _date_range(start: date, end: date) -> Iterable[date]:
    """Inclusive date iterator."""
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _pull_one_day(day: date, *, with_boxscores: bool = True) -> dict[str, int]:
    """Pull schedule + per-game boxscores for one day."""
    counts: dict[str, int] = {}
    raw = mlb_statsapi.fetch_schedule(day, day)
    games_df = mlb_statsapi.normalize_schedule(raw)
    if games_df.empty:
        return counts
    counts["games"] = upsert_dataframe(games_df, "games", key_columns=["game_pk"])

    # Probable pitchers — one row per side, looked up via schedule + people endpoints.
    pp_rows: list[pd.DataFrame] = []
    for pk in games_df["game_pk"].astype(int):
        pp = mlb_statsapi.fetch_probable_pitchers(int(pk))
        if not pp.empty:
            pp_rows.append(pp)
    if pp_rows:
        pp_df = pd.concat(pp_rows, ignore_index=True)
        counts["probable_pitchers"] = upsert_dataframe(
            pp_df, "probable_pitchers", key_columns=["game_pk", "pitcher_id"]
        )

    if not with_boxscores:
        return counts

    # Boxscores for any finalized game.
    finals = games_df[games_df["status"] == "Final"]
    if finals.empty:
        return counts
    tb_total = lp_total = pg_total = bg_total = 0
    for pk in finals["game_pk"].astype(int):
        payload = mlb_statsapi.fetch_boxscore(int(pk))
        if not payload:
            continue
        tb_total += upsert_dataframe(
            mlb_statsapi.normalize_team_boxscore(int(pk), payload),
            "team_boxscores", key_columns=["game_pk", "team_id"],
        )
        lp_total += upsert_dataframe(
            mlb_statsapi.normalize_lineup(int(pk), payload),
            "lineups", key_columns=["game_pk", "team_id", "batting_order"],
        )
        pg_total += upsert_dataframe(
            mlb_statsapi.normalize_pitcher_stats(int(pk), payload),
            "pitcher_game_stats", key_columns=["game_pk", "pitcher_id"],
        )
        bg_total += upsert_dataframe(
            mlb_statsapi.normalize_batter_stats(int(pk), payload),
            "batter_game_stats", key_columns=["game_pk", "batter_id"],
        )
    counts["team_boxscores"] = tb_total
    counts["lineups"] = lp_total
    counts["pitcher_game_stats"] = pg_total
    counts["batter_game_stats"] = bg_total
    return counts


def pull_date_range(start_date: date, end_date: date) -> dict[str, int]:
    """Pull every day in the inclusive range; sum row counts by table."""
    init_schema()
    totals: dict[str, int] = {}
    for d in _date_range(start_date, end_date):
        try:
            day_counts = _pull_one_day(d)
        except Exception:  # noqa: BLE001 -- one bad day shouldn't kill the loop
            log.exception("pipeline.day.failed", date=d.isoformat())
            continue
        for k, v in day_counts.items():
            totals[k] = totals.get(k, 0) + int(v)
        log.info("pipeline.day.done", date=d.isoformat(), counts=day_counts)
    return totals


def pull_season(
    season: int, with_statcast: bool = True, with_weather: bool = True
) -> dict[str, int]:
    """Pull a full season. Statcast + weather are best-effort.

    Statcast ingestion is delegated to ``pybaseball``; we wrap it in a
    try/except because that package occasionally breaks when MLB Stats
    API changes shape. Failure there does NOT prevent the rest of the
    season from landing.
    """
    init_schema()
    ensure_venues()
    # MLB regular season runs late March through early October; we use
    # a generous window so spring training games + early postseason are
    # still picked up if present in the schedule.
    start = date(season, 3, 15)
    end = date(season, 11, 15)
    totals = pull_date_range(start, end)

    if with_statcast:
        try:
            from mlb_model.data.sources import statcast  # optional

            totals["statcast_pitcher_daily"] = statcast.ingest_season(season)
        except ImportError:
            log.info("pipeline.statcast.unavailable")
        except Exception:  # noqa: BLE001
            log.exception("pipeline.statcast.failed", season=season)

    if with_weather:
        try:
            from mlb_model.data.sources import weather as _weather
            from mlb_model.data.warehouse import query

            slate = query(
                """
                SELECT game_pk, scheduled_start, home_team_abbr
                FROM games
                WHERE season = ? AND game_pk NOT IN (SELECT game_pk FROM weather)
                """,
                (int(season),),
            )
            totals["weather"] = _weather.ingest_weather_for_games(
                slate, seed_venues_df()
            )
        except Exception:  # noqa: BLE001
            log.exception("pipeline.weather.failed", season=season)

    return totals
