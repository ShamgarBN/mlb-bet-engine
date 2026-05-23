"""Ingest historical odds CSVs into the warehouse.

Drop any number of Kaggle (or other) odds CSVs / XLSX files into
``data/raw/odds/`` and run this script. Each file becomes one ``book``
row in ``odds_history`` (the filename becomes the book label).

Usage:
    uv run python scripts/backfill_odds.py
    uv run python scripts/backfill_odds.py path/to/odds.csv
    uv run python scripts/backfill_odds.py path/to/dir --book consensus_close

See RECOVERY.md for a list of public odds datasets and how to format
your CSV if the auto-detector can't read it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from mlb_model.data.sources import odds_csv
from mlb_model.data.warehouse import init_schema
from mlb_model.logging import configure_logging, get_logger

log = get_logger("backfill_odds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*",
        help="One or more CSV/XLSX files OR a directory. "
             "Defaults to data/raw/odds/.",
    )
    parser.add_argument(
        "--book", default=None,
        help="Override the book label (default: derived from filename).",
    )
    args = parser.parse_args()

    configure_logging()
    init_schema()

    targets: list[Path] = (
        [Path(p) for p in args.paths] if args.paths else [Path("data/raw/odds")]
    )

    total = 0
    for path in targets:
        if path.is_dir():
            results = odds_csv.ingest_directory(path)
            for name, n in results.items():
                log.info("odds.file.ingested", file=name, rows=n)
                total += n
        elif path.is_file():
            n = odds_csv.ingest_csv(path, book=args.book)
            log.info("odds.file.ingested", file=str(path), rows=n)
            total += n
        else:
            log.warning("odds.path.missing", path=str(path))

    log.info("odds.backfill.done", total_rows=total)
    if total == 0:
        print("No rows ingested. Drop a CSV/XLSX into data/raw/odds/ and re-run.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
