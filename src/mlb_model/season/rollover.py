"""Season lifecycle predicates.

The MLB regular season runs late March -> early October, with playoffs
through early November. We want the weekly retraining job to switch
into "end of season sweep" mode once the regular season finishes for
the year, AND we want to make sure the model handles brand new seasons
(no historical data yet) gracefully.

This module is the single source of truth for "what season is it?"
questions so callers don't have to invent ad-hoc heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls
from typing import Literal

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("season.rollover")

# The regular season has finished for the year if we're past this
# month/day boundary AND the warehouse shows we have N+ finalized
# games from this year. (Just a date check would mis-fire if MLB ever
# extends the schedule.)
_REGULAR_SEASON_END_MONTH = 10
_REGULAR_SEASON_END_DAY = 5
_MIN_FINALIZED_GAMES_FOR_FULL_SEASON = 2200  # ~2430 in a healthy year


SeasonState = Literal["in_progress", "ended", "off_season", "pre_season"]


@dataclass
class SeasonStatus:
    state: SeasonState
    season: int
    finalized_games: int
    description: str


def is_regular_season_over(season: int, *, today: date_cls | None = None) -> bool:
    """True iff ``season``'s regular season is in the books.

    Uses two signals: (1) we're past the calendar boundary for that
    season, and (2) the warehouse has at least roughly a full-season
    number of finalized games. Either one alone gives false positives
    in edge cases (early playoff schedule shifts, partial-season pulls).
    """
    today = today or date_cls.today()
    if today < date_cls(season, _REGULAR_SEASON_END_MONTH, _REGULAR_SEASON_END_DAY):
        return False

    df = query(
        "SELECT COUNT(*) AS n FROM games WHERE season = ? AND status = 'Final'",
        (int(season),),
    )
    if df.empty:
        return False
    return int(df.iloc[0]["n"]) >= _MIN_FINALIZED_GAMES_FOR_FULL_SEASON


def last_completed_season(today: date_cls | None = None) -> int | None:
    """Return the most recent season whose regular season is over.

    Walks back from ``today.year`` until it finds one. Returns None if
    no season in the warehouse has finished yet.
    """
    today = today or date_cls.today()
    for season in range(today.year, today.year - 8, -1):
        if is_regular_season_over(season, today=today):
            return season
    return None


def detect_season_state(today: date_cls | None = None) -> SeasonStatus:
    """High-level summary used by the weekly-train scheduler.

    Returns one of:

    * ``in_progress``  -- a season is actively running and weekly
      retraining is a normal Sunday retrain.
    * ``ended``        -- THIS season's regular season just finished;
      time to trigger the end-of-season sweep.
    * ``off_season``   -- regular season already wrapped, sweep done.
    * ``pre_season``   -- new year before opening day; no games yet.
    """
    today = today or date_cls.today()
    this_year = today.year
    df = query(
        """
        SELECT season, MIN(game_date) AS first_game, MAX(game_date) AS last_game,
               COUNT(*) FILTER (WHERE status = 'Final') AS finalized
        FROM games
        WHERE season = ?
        GROUP BY season
        """,
        (this_year,),
    )

    if df.empty:
        return SeasonStatus(
            state="pre_season",
            season=this_year,
            finalized_games=0,
            description=(
                f"No games on record for {this_year} yet. "
                "First call to morning-sync after opening day will populate it."
            ),
        )

    row = df.iloc[0]
    finalized = int(row["finalized"] or 0)
    if is_regular_season_over(this_year, today=today):
        # Has the end-of-season report already been written? Caller's
        # responsibility to check that (we don't want a circular import).
        return SeasonStatus(
            state="ended",
            season=this_year,
            finalized_games=finalized,
            description=(
                f"{this_year} regular season is complete "
                f"({finalized} finalized games)."
            ),
        )

    return SeasonStatus(
        state="in_progress",
        season=this_year,
        finalized_games=finalized,
        description=f"{this_year} season in progress; {finalized} games finalized.",
    )
