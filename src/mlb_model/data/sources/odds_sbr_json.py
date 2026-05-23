"""Modern (2022+) odds ingestion from the SBR consensus JSON archive.

The user supplies a JSON dataset under ``data/raw/odds_scraped/`` (see
the README quick-start). This module reads every file under that path,
deduplicates by (date, matchup, book), and upserts to ``odds_history``
with ``book='sbr_consensus'``.

The JSON format we expect is a list of records with these keys (the
public consensus scraper output):
- ``date``: YYYY-MM-DD
- ``home``, ``away``: 2-3 letter abbreviations
- ``ml_home``, ``ml_away``: American odds, closing
- ``rl_home_line``, ``rl_home_price``, ``rl_away_price``
- ``total``, ``total_over_price``, ``total_under_price``
- optionally ``ml_open_home`` / ``ml_open_away`` for line movement
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.odds_sbr_json")


def _archive_dir() -> Path:
    return settings.raw_dir / "odds_scraped"


def _iter_records(root: Path):
    """Yield records from every .json/.ndjson file under ``root``."""
    if not root.exists():
        return
    for path in sorted(root.rglob("*.json")):
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            log.warning("odds_sbr_json.read_failed", path=str(path))
            continue
        # Accept either a top-level list or a {"records": [...]} envelope.
        records = payload if isinstance(payload, list) else payload.get("records", [])
        for r in records:
            if isinstance(r, dict):
                yield r


def _to_int(v) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ingest_dataset() -> int:
    """Ingest every JSON file in the archive root. Returns rows upserted."""
    root = _archive_dir()
    rows: list[dict] = []
    for r in _iter_records(root):
        date_s = r.get("date") or r.get("game_date")
        home = r.get("home") or r.get("home_team_abbr")
        away = r.get("away") or r.get("away_team_abbr")
        if not date_s or not home or not away:
            continue
        rows.append({
            "game_date": pd.to_datetime(date_s).date(),
            "home_team_abbr": str(home),
            "away_team_abbr": str(away),
            "book": "sbr_consensus",
            "ml_open_home": _to_int(r.get("ml_open_home")),
            "ml_open_away": _to_int(r.get("ml_open_away")),
            "ml_close_home": _to_int(r.get("ml_home") or r.get("ml_close_home")),
            "ml_close_away": _to_int(r.get("ml_away") or r.get("ml_close_away")),
            "rl_close_home": _to_float(r.get("rl_home_line") or r.get("rl_close_home")),
            "rl_close_home_price": _to_int(r.get("rl_home_price") or r.get("rl_close_home_price")),
            "rl_close_away_price": _to_int(r.get("rl_away_price") or r.get("rl_close_away_price")),
            "total_close": _to_float(r.get("total") or r.get("total_close")),
            "total_close_over": _to_int(r.get("total_over_price") or r.get("total_close_over")),
            "total_close_under": _to_int(r.get("total_under_price") or r.get("total_close_under")),
        })

    if not rows:
        log.info("odds_sbr_json.no_records", root=str(root))
        return 0
    df = pd.DataFrame(rows)
    # Drop within-batch duplicates so the upsert key-conflict on the same
    # transaction doesn't surprise us.
    df = df.drop_duplicates(subset=["game_date", "home_team_abbr", "away_team_abbr", "book"])
    return upsert_dataframe(
        df, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
