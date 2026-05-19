"""Quick backfill: schedule + boxscores for 2022-2024.

Skips Statcast (which takes hours) and weather (which takes a long time too).
Gives us enough data to do a real end-to-end test of training + backtest.
"""

from __future__ import annotations

from mlb_model.data.pipeline import (
    ensure_venues,
    ingest_boxscores_and_lineups,
)
from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.warehouse import init_schema
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_quick")


def main() -> None:
    configure_logging()
    init_schema()
    ensure_venues()
    for season in (2022, 2023, 2024):
        log.info("backfill.schedule", season=season)
        mlb_statsapi.ingest_season(season)
        log.info("backfill.boxscores", season=season)
        # No limit -- pull everything
        counts = ingest_boxscores_and_lineups(season)
        log.info("backfill.season.done", season=season, counts=counts)


if __name__ == "__main__":
    main()
