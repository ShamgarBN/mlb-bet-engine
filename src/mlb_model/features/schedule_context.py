"""Schedule context features: rest days, travel, doubleheader leg, getaway day.

These small-effect features have been shown in academic literature to be
worth roughly 0.5-1.0 percentage points of win-probability accuracy. They're
free to compute from the already-ingested schedule.

Travel distance is great-circle (haversine) between the previous game's
venue and today's venue.
"""

from __future__ import annotations

import math

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("features.schedule")


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    if any(pd.isna([lat1, lon1, lat2, lon2])):
        return 0.0
    r_mi = 3958.7613
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r_mi * math.asin(math.sqrt(a))


def build_schedule_context(season_start: int, season_end: int) -> pd.DataFrame:
    """Return one row per (team, game) with schedule features.

    Columns:
      team_id, game_pk, game_date, is_home,
      days_rest, travel_miles, is_getaway_day, is_doubleheader_leg2
    """
    # Subtle bug we used to ship: ``games.venue_id`` is the MLB Stats
    # API's venue_id, but ``venues.venue_id`` is a synthetic hash of the
    # home-team abbreviation (see ``data/venues_seed.py``). Joining the
    # two produced no matches, so ``venue_lat`` / ``venue_lon`` were
    # always NULL and ``travel_miles`` was always 0 for every team-game.
    # The signal was silently absent from the model.
    #
    # The home venue's lat/lon is uniquely determined by the home team
    # abbr (one park per team, dome+altitude included in the seed), so
    # joining on team abbr gives us the geometry we actually want.
    games = query(
        """
        SELECT g.game_pk, g.game_date, g.season, g.home_team_id, g.away_team_id,
               g.home_team_abbr, g.away_team_abbr,
               g.doubleheader, g.game_number, g.venue_id,
               v.latitude  AS venue_lat,
               v.longitude AS venue_lon
        FROM games g
        LEFT JOIN venues v ON v.team_abbr = g.home_team_abbr
        WHERE g.season BETWEEN ? AND ?
        ORDER BY g.game_date, g.game_pk
        """,
        params=(season_start, season_end),
    )
    if games.empty:
        return games
    games["game_date"] = pd.to_datetime(games["game_date"])

    # Reshape to one row per (team, game)
    home = games.rename(columns={"home_team_id": "team_id"}).assign(is_home=True)[
        [
            "team_id", "game_pk", "game_date", "doubleheader", "game_number",
            "venue_id", "venue_lat", "venue_lon", "is_home",
        ]
    ]
    away = games.rename(columns={"away_team_id": "team_id"}).assign(is_home=False)[
        [
            "team_id", "game_pk", "game_date", "doubleheader", "game_number",
            "venue_id", "venue_lat", "venue_lon", "is_home",
        ]
    ]
    panel = pd.concat([home, away], ignore_index=True).sort_values(["team_id", "game_date", "game_pk"])

    panel["prev_game_date"] = panel.groupby("team_id")["game_date"].shift(1)
    panel["prev_lat"] = panel.groupby("team_id")["venue_lat"].shift(1)
    panel["prev_lon"] = panel.groupby("team_id")["venue_lon"].shift(1)

    panel["days_rest"] = (panel["game_date"] - panel["prev_game_date"]).dt.days.fillna(7).clip(lower=0, upper=14)
    panel["travel_miles"] = panel.apply(
        lambda r: _haversine_miles(r["prev_lat"], r["prev_lon"], r["venue_lat"], r["venue_lon"]),
        axis=1,
    )
    panel["is_doubleheader_leg2"] = (panel["game_number"] == 2).fillna(False)

    # Getaway day = last game of a series for the visiting team. Heuristic:
    # if the next game for this team is at a *different* venue, today was a
    # getaway day.
    panel["next_venue_id"] = panel.groupby("team_id")["venue_id"].shift(-1)
    panel["is_getaway_day"] = (
        (panel["next_venue_id"] != panel["venue_id"])
        & panel["next_venue_id"].notna()
    )

    out = panel[
        [
            "team_id", "game_pk", "game_date", "is_home",
            "days_rest", "travel_miles", "is_getaway_day", "is_doubleheader_leg2",
        ]
    ].reset_index(drop=True)
    log.info("schedule_context.built", rows=len(out))
    return out
