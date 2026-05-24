"""Modern (2022+) odds ingestion from SBR JSON archives.

The user supplies a JSON dataset under ``data/raw/odds_scraped/`` (see
the README quick-start). This module reads every file under that path,
deduplicates by (date, matchup, book), and upserts to ``odds_history``.

Two on-disk shapes are supported:

1. The original flat shape — a top-level list (or ``{"records": [...]}``
   envelope) of records with these keys, one row per (game, book):
   ``date``, ``home``, ``away``, ``ml_home``, ``ml_away``,
   ``rl_home_line``, ``rl_home_price``, ``rl_away_price``,
   ``total``, ``total_over_price``, ``total_under_price`` (plus
   optional ``ml_open_home`` / ``ml_open_away``). Records get
   ``book='sbr_consensus'`` unless they carry their own ``book`` value.

2. The ArnavSaraogi/mlb-odds-scraper dataset shape — a top-level dict
   keyed by ``YYYY-MM-DD``, mapping to a list of games with nested
   ``gameView`` and ``odds.{moneyline,pointspread,totals}`` arrays
   (one entry per sportsbook). Each (game, book) is flattened to one
   row and stamped with the lowercased sportsbook name as ``book``.
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


# Arnav dataset team shortNames -> our 3-letter convention.
_ARNAV_TEAM_ALIASES = {
    "ATH": "OAK",   # Athletics rebrand in 2024+
    "AZ": "ARI",
    "CHW": "CWS",
    "WAS": "WSH",
}
# Skip these — All-Star matchups, not real games.
_ARNAV_SKIP_TEAMS = {"AL", "NL"}
# Game types worth ingesting: regular season + postseason. Skip "S"
# (spring training) and "Unknown".
_ARNAV_KEEP_GAMETYPES = {"R", "D", "L", "W", "F"}


def _normalize_arnav_team(short: str | None) -> str | None:
    if not short:
        return None
    if short in _ARNAV_SKIP_TEAMS:
        return None
    return _ARNAV_TEAM_ALIASES.get(short, short)


def _iter_arnav_records(payload: dict):
    """Yield one flat record per (game, book) from the Arnav dataset shape."""
    for date_s, games in payload.items():
        if not isinstance(games, list):
            continue
        for g in games:
            if not isinstance(g, dict):
                continue
            gv = g.get("gameView") or {}
            if gv.get("gameType") not in _ARNAV_KEEP_GAMETYPES:
                continue
            home = _normalize_arnav_team((gv.get("homeTeam") or {}).get("shortName"))
            away = _normalize_arnav_team((gv.get("awayTeam") or {}).get("shortName"))
            if not (home and away):
                continue

            odds = g.get("odds") or {}
            ml_by_book: dict[str, dict] = {bm["sportsbook"]: bm for bm in (odds.get("moneyline") or []) if bm.get("sportsbook")}
            ps_by_book: dict[str, dict] = {bm["sportsbook"]: bm for bm in (odds.get("pointspread") or []) if bm.get("sportsbook")}
            tot_by_book: dict[str, dict] = {bm["sportsbook"]: bm for bm in (odds.get("totals") or []) if bm.get("sportsbook")}

            for book in set(ml_by_book) | set(ps_by_book) | set(tot_by_book):
                ml = ml_by_book.get(book) or {}
                ps = ps_by_book.get(book) or {}
                tot = tot_by_book.get(book) or {}
                ml_open = ml.get("openingLine") or {}
                ml_close = ml.get("currentLine") or {}
                ps_close = ps.get("currentLine") or {}
                tot_close = tot.get("currentLine") or {}
                yield {
                    "date": date_s,
                    "home": home,
                    "away": away,
                    "book": book,
                    "ml_open_home": ml_open.get("homeOdds"),
                    "ml_open_away": ml_open.get("awayOdds"),
                    "ml_home": ml_close.get("homeOdds"),
                    "ml_away": ml_close.get("awayOdds"),
                    "rl_home_line": ps_close.get("homeSpread"),
                    "rl_home_price": ps_close.get("homeOdds"),
                    "rl_away_price": ps_close.get("awayOdds"),
                    "total": tot_close.get("total"),
                    "total_over_price": tot_close.get("overOdds"),
                    "total_under_price": tot_close.get("underOdds"),
                }


def _is_arnav_shape(payload) -> bool:
    """True if ``payload`` looks like the Arnav date-keyed dataset."""
    if not isinstance(payload, dict) or not payload:
        return False
    sample_key = next(iter(payload))
    if not (isinstance(sample_key, str) and len(sample_key) == 10 and sample_key[4] == "-" and sample_key[7] == "-"):
        return False
    sample_val = payload[sample_key]
    return isinstance(sample_val, list)


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
        if _is_arnav_shape(payload):
            yield from _iter_arnav_records(payload)
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
        book = r.get("book") or "sbr_consensus"
        rows.append({
            "game_date": pd.to_datetime(date_s).date(),
            "home_team_abbr": str(home),
            "away_team_abbr": str(away),
            "book": str(book).lower(),
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
