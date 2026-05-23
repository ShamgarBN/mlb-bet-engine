"""Historical odds ingestion.

Originally backed by the SportsBookReviewsOnline (SBRO) archive. SBRO
stopped updating after the 2021 season, so this source covers 2014-2021.
For 2022+ data we use the SBR JSON archive (see ``odds_sbr_json.py``).

Implementation note: the original scraper used a mix of SBRO HTML pages
and a cached XLSX archive. The XLSX format is preserved here as the
primary path because it survives without network access; the HTML
fallback can be wired in if the archive isn't on disk.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.odds_history")


def _xlsx_path_for_season(season: int) -> Path:
    """Where the SBRO XLSX archive lives on disk."""
    return settings.raw_dir / "odds" / f"mlb_odds_{season}.xlsx"


def _normalize_sbro_xlsx(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Parse the SBRO XLSX format into our ``odds_history`` schema.

    The archive groups one game across two rows (away then home). We
    pivot to one row per game and stamp ``book='sbro_consensus'``.

    If the XLSX schema doesn't match expectations (different vendor,
    different season) we return an empty DataFrame and log — never
    crash the backfill loop.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    expected = {"Date", "VH", "Team", "Final", "Open", "Close", "ML",
                "Run Line", "Run Line Odds", "Total", "Total Odds"}
    if not expected.issubset(set(df.columns)):
        log.warning("odds.xlsx.unexpected_columns",
                    season=season, columns=list(df.columns))
        return pd.DataFrame()

    rows: list[dict] = []
    for i in range(0, len(df) - 1, 2):
        away = df.iloc[i]
        home = df.iloc[i + 1]
        if str(away.get("VH")).upper() != "V" or str(home.get("VH")).upper() != "H":
            continue
        rows.append({
            "game_date": pd.to_datetime(away.get("Date")).date(),
            "home_team_abbr": str(home.get("Team")),
            "away_team_abbr": str(away.get("Team")),
            "book": "sbro_consensus",
            "ml_open_home": _to_int(home.get("Open")),
            "ml_open_away": _to_int(away.get("Open")),
            "ml_close_home": _to_int(home.get("Close")),
            "ml_close_away": _to_int(away.get("Close")),
            "rl_close_home": _to_float(home.get("Run Line")),
            "rl_close_home_price": _to_int(home.get("Run Line Odds")),
            "rl_close_away_price": _to_int(away.get("Run Line Odds")),
            "total_close": _to_float(home.get("Total")),
            "total_close_over": _to_int(home.get("Total Odds")),
            "total_close_under": _to_int(away.get("Total Odds")),
        })
    return pd.DataFrame(rows)


def _to_int(v) -> int | None:
    if v is None or pd.isna(v) or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or pd.isna(v) or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ingest_season(season: int) -> int:
    """Ingest one season of SBRO odds. Returns rows upserted."""
    path = _xlsx_path_for_season(season)
    if not path.exists():
        log.info("odds.season.missing_xlsx", season=season, path=str(path))
        return 0
    try:
        df = pd.read_excel(path)
    except Exception:
        log.exception("odds.season.read_failed", season=season, path=str(path))
        return 0
    rows = _normalize_sbro_xlsx(df, season)
    if rows.empty:
        return 0
    return upsert_dataframe(
        rows, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
