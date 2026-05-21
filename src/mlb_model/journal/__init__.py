"""Prediction journal.

This module is the answer to "how well is the model doing right now?".
It records every prediction the model has produced -- not just the
picks the user clicked "log" on -- so model accuracy can be measured
honestly across the whole season, every season, indefinitely.

Two surfaces:

* :mod:`mlb_model.journal.record` -- append a row per (game, market,
  pick) every time ``predict_for_date`` runs. Idempotent within a day:
  re-running predict only adds a new snapshot row, never duplicates.
* :mod:`mlb_model.journal.metrics` -- read the journal, join to
  finalized scores, and compute calibration / rolling accuracy / slice
  breakdowns.

Storage lives at ``data/journal/predictions.parquet`` and is intentionally
NOT under ``data/cache/`` so morning-sync's cache invalidation can't
wipe history.
"""

from mlb_model.journal.record import (
    JOURNAL_PATH,
    record_predictions,
    record_predictions_from_df,
)
from mlb_model.journal.metrics import (
    SeasonSummary,
    calibration_bins,
    grade_journal,
    rolling_accuracy,
    season_summary,
    slice_breakdown,
)

__all__ = [
    "JOURNAL_PATH",
    "SeasonSummary",
    "calibration_bins",
    "grade_journal",
    "record_predictions",
    "record_predictions_from_df",
    "rolling_accuracy",
    "season_summary",
    "slice_breakdown",
]
