"""Backfill historical odds from SBRO for 2014-2021.

SBRO has stopped updating after 2021. For 2022+ we have to live without
historical odds OR provide manual XLSX files in data/raw/odds/.
"""

from __future__ import annotations

import sys

from mlb_model.data.sources import odds_history
from mlb_model.data.warehouse import init_schema
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_odds")


def main(seasons: tuple[int, ...] = tuple(range(2014, 2022))) -> None:
    configure_logging()
    init_schema()
    for season in seasons:
        try:
            n = odds_history.ingest_season(season)
            log.info("odds.season.done", season=season, rows=n)
        except Exception as exc:  # noqa: BLE001
            log.warning("odds.season.failed", season=season, error=str(exc))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seasons = tuple(int(s) for s in sys.argv[1:])
    else:
        seasons = tuple(range(2014, 2022))
    main(seasons)
