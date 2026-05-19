"""Extended backfill: 2018-2021 (plus 2025 if available).

Runs everything *except* Statcast (which is much heavier and we'll do
separately). Each season is independent so failures don't propagate.
"""

from __future__ import annotations

import sys

from mlb_model.data.pipeline import (
    ensure_venues,
    ingest_boxscores_and_lineups,
)
from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.warehouse import init_schema
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_extended")

DEFAULT_SEASONS = (2018, 2019, 2020, 2021, 2025)


def main(seasons: tuple[int, ...] = DEFAULT_SEASONS) -> None:
    configure_logging()
    init_schema()
    ensure_venues()
    for season in seasons:
        try:
            log.info("extended.schedule", season=season)
            mlb_statsapi.ingest_season(season)
            log.info("extended.boxscores", season=season)
            counts = ingest_boxscores_and_lineups(season)
            log.info("extended.season.done", season=season, counts=counts)
        except Exception as exc:  # noqa: BLE001
            log.error("extended.season.failed", season=season, error=str(exc))


if __name__ == "__main__":
    seasons = tuple(int(s) for s in sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_SEASONS
    main(seasons)
