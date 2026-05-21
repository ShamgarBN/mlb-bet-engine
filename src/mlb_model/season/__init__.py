"""Season-level analysis and lifecycle automation.

Two surfaces:

* :mod:`mlb_model.season.end_of_season` -- the once-a-year "full sweep"
  that pulls the just-finished season, runs an extended backtest, and
  writes a deep performance report.
* :mod:`mlb_model.season.rollover` -- predicates for "is the season
  over?" / "is this a new season?" / "should we promote a new
  champion model snapshot?" so the weekly retrain can auto-trigger
  the right behavior in October without code changes.
"""

from mlb_model.season.end_of_season import (
    EndOfSeasonReport,
    run_end_of_season_sweep,
)
from mlb_model.season.rollover import (
    detect_season_state,
    is_regular_season_over,
    last_completed_season,
)

__all__ = [
    "EndOfSeasonReport",
    "detect_season_state",
    "is_regular_season_over",
    "last_completed_season",
    "run_end_of_season_sweep",
]
