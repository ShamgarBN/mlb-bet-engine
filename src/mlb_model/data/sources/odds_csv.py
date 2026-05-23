"""Generic CSV / XLSX odds ingester.

Designed to consume public historical odds datasets (Kaggle, OpenSports,
or a hand-rolled CSV) without locking the project to any one vendor.

Two ingestion paths:

* :func:`ingest_csv` — point it at a single CSV/XLSX file. Auto-detects
  the column layout (long format with V/H rows, or wide one-row-per-game).
* :func:`ingest_directory` — point it at a directory and it ingests
  every .csv/.xlsx underneath, with the filename becoming the ``book``
  label (so you can keep multiple vintages side by side).

The warehouse schema (``odds_history``) is the contract:
``game_date, home_team_abbr, away_team_abbr, book`` is the primary key,
and the body is open/close moneyline both sides plus run-line and total
markets.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.odds_csv")


# --------------------------------------------------------------------------- #
# Team name → abbreviation                                                    #
# --------------------------------------------------------------------------- #
# Kaggle datasets commonly use full team names ("Boston Red Sox") or city
# names ("Boston"). MLB Stats API uses 3-letter abbreviations. We need
# either to be acceptable, so this map covers both directions.

_NAME_TO_ABBR: dict[str, str] = {
    # Full names
    "arizona diamondbacks": "ARI", "atlanta braves": "ATL",
    "baltimore orioles": "BAL",  "boston red sox": "BOS",
    "chicago cubs": "CHC",       "chicago white sox": "CWS",
    "cincinnati reds": "CIN",    "cleveland guardians": "CLE",
    "cleveland indians": "CLE",  # historical name pre-2022
    "colorado rockies": "COL",   "detroit tigers": "DET",
    "houston astros": "HOU",     "kansas city royals": "KC",
    "los angeles angels": "LAA", "los angeles dodgers": "LAD",
    "miami marlins": "MIA",      "milwaukee brewers": "MIL",
    "minnesota twins": "MIN",    "new york mets": "NYM",
    "new york yankees": "NYY",   "oakland athletics": "OAK",
    "philadelphia phillies": "PHI", "pittsburgh pirates": "PIT",
    "san diego padres": "SD",    "seattle mariners": "SEA",
    "san francisco giants": "SF","st. louis cardinals": "STL",
    "st louis cardinals": "STL",
    "tampa bay rays": "TB",      "texas rangers": "TEX",
    "toronto blue jays": "TOR",  "washington nationals": "WSH",
    # City-only common abbreviations
    "athletics": "OAK", "diamondbacks": "ARI", "d-backs": "ARI",
    "redsox": "BOS",
    # Some datasets use slightly different short codes
    "WAS": "WSH", "KCR": "KC", "CHW": "CWS", "TBR": "TB", "SDP": "SD",
    "SFG": "SF",  "WSN": "WSH", "ANA": "LAA",
}

_VALID_ABBRS = {
    "ARI", "ATL", "BAL", "BOS", "CHC", "CWS", "CIN", "CLE", "COL", "DET",
    "HOU", "KC",  "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SD",  "SEA", "SF",  "STL", "TB",  "TEX", "TOR", "WSH",
}


def _normalize_team(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    # Already a known abbreviation?
    upper = s.upper()
    if upper in _VALID_ABBRS:
        return upper
    if upper in _NAME_TO_ABBR:
        return _NAME_TO_ABBR[upper]
    lower = s.lower().strip()
    if lower in _NAME_TO_ABBR:
        return _NAME_TO_ABBR[lower]
    return None


# --------------------------------------------------------------------------- #
# Column auto-detection                                                       #
# --------------------------------------------------------------------------- #

# Each canonical field is paired with a list of patterns (lowercased,
# matched as a substring against the lowercase column name).
_COLUMN_PATTERNS: dict[str, list[str]] = {
    "date":            ["date", "game_date", "gameday"],
    "home_team":       ["home_team", "home team", "home_name", "home"],
    "away_team":       ["away_team", "away team", "away_name",
                        "visit", "away", "vis_team"],
    "ml_close_home":   ["close_home_ml", "ml_close_home", "home_ml_close",
                        "home_close_ml", "ml home close", "moneyline_home_close",
                        "home_ml", "moneyline_home", "ml_home"],
    "ml_close_away":   ["close_away_ml", "ml_close_away", "away_ml_close",
                        "away_close_ml", "ml away close", "moneyline_away_close",
                        "away_ml", "moneyline_away", "ml_away"],
    "ml_open_home":    ["open_home_ml", "ml_open_home", "home_ml_open"],
    "ml_open_away":    ["open_away_ml", "ml_open_away", "away_ml_open"],
    "total_close":     ["total_close", "close_total", "total", "o/u", "ou_close"],
    "total_close_over":  ["over_close", "close_over", "over_odds", "over_price",
                          "total_over"],
    "total_close_under": ["under_close", "close_under", "under_odds", "under_price",
                          "total_under"],
    "rl_close_home":   ["spread_home", "home_spread", "runline_home",
                        "rl_home_line", "rl_close_home", "spread"],
    "rl_close_home_price": ["home_spread_odds", "spread_home_price",
                            "runline_home_price", "rl_home_price",
                            "rl_close_home_price"],
    "rl_close_away_price": ["away_spread_odds", "spread_away_price",
                            "runline_away_price", "rl_away_price",
                            "rl_close_away_price"],
}


def _auto_map_columns(columns: list[str]) -> dict[str, str]:
    """Return {canonical_field: actual_column_name} for whatever we can match."""
    lookup = {c: c.lower().strip() for c in columns}
    mapping: dict[str, str] = {}
    for canonical, patterns in _COLUMN_PATTERNS.items():
        for col, low in lookup.items():
            if col in mapping.values():
                continue
            if any(p in low for p in patterns):
                mapping[canonical] = col
                break
    return mapping


# --------------------------------------------------------------------------- #
# Long-format (V/H) detection                                                 #
# --------------------------------------------------------------------------- #
# Some archives (SBRO-style XLSX, certain Kaggle dumps) put two rows per
# game — one for the visitor, one for the home team — keyed by a column
# called VH / Home_Visitor / TeamType. We pivot back to one row per game.


def _is_long_format(df: pd.DataFrame) -> bool:
    cols_low = {c.lower().strip() for c in df.columns}
    return any(c in cols_low for c in {"vh", "v/h", "team_type"})


def _pivot_long_to_wide(df: pd.DataFrame) -> pd.DataFrame:
    """Pair away/home rows into single-game wide rows."""
    vh_col = next(
        (c for c in df.columns if c.lower().strip() in {"vh", "v/h", "team_type"}),
        None,
    )
    if vh_col is None:
        return df
    df = df.reset_index(drop=True)
    away_rows = df[df[vh_col].astype(str).str.upper().str.startswith("V")]
    home_rows = df[df[vh_col].astype(str).str.upper().str.startswith("H")]
    n = min(len(away_rows), len(home_rows))
    if n == 0:
        return pd.DataFrame()
    out_rows = []
    for i in range(n):
        a = away_rows.iloc[i]
        h = home_rows.iloc[i]
        row = {
            "date": h.get("Date") or h.get("date"),
            "home_team": h.get("Team") or h.get("team"),
            "away_team": a.get("Team") or a.get("team"),
            "ml_close_home": h.get("Close") or h.get("ML") or h.get("close"),
            "ml_close_away": a.get("Close") or a.get("ML") or a.get("close"),
            "ml_open_home": h.get("Open") or h.get("open"),
            "ml_open_away": a.get("Open") or a.get("open"),
            "rl_close_home": h.get("Run Line") or h.get("Spread"),
            "rl_close_home_price": h.get("Run Line Odds") or h.get("Spread Odds"),
            "rl_close_away_price": a.get("Run Line Odds") or a.get("Spread Odds"),
            "total_close": h.get("Total") or a.get("Total"),
            "total_close_over": h.get("Total Odds") if str(h.get("Total")) > str(a.get("Total")) else a.get("Total Odds"),
            "total_close_under": a.get("Total Odds") if str(h.get("Total")) > str(a.get("Total")) else h.get("Total Odds"),
        }
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# --------------------------------------------------------------------------- #
# Coercion helpers                                                            #
# --------------------------------------------------------------------------- #


def _to_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _to_date(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or v == "":
        return None
    try:
        return pd.to_datetime(v).date()
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Public ingestion                                                            #
# --------------------------------------------------------------------------- #


def _read_any(path: Path) -> pd.DataFrame:
    """Read a CSV or XLSX into a DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported odds file format: {suffix}")


def _book_label_from_path(path: Path) -> str:
    """Derive a stable, readable book label from a filename."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem.lower()).strip("_")
    return f"csv_{stem}"


def ingest_csv(
    path: str | Path,
    book: str | None = None,
    column_overrides: dict[str, str] | None = None,
) -> int:
    """Ingest a single CSV / XLSX file. Returns rows upserted.

    ``book`` defaults to a label derived from the filename so multiple
    vintages co-exist in the warehouse without colliding on the PK.
    Pass ``column_overrides`` if the auto-detector picks the wrong
    column (e.g. ``{"date": "GameDate", "home_team": "HomeName"}``).
    """
    p = Path(path)
    if not p.exists():
        log.warning("odds_csv.missing", path=str(p))
        return 0
    try:
        raw = _read_any(p)
    except Exception:
        log.exception("odds_csv.read_failed", path=str(p))
        return 0
    if raw is None or raw.empty:
        return 0

    book_label = book or _book_label_from_path(p)

    # Long format (V/H pairs) → wide pivot first.
    if _is_long_format(raw):
        raw = _pivot_long_to_wide(raw)
        if raw.empty:
            log.warning("odds_csv.empty_after_pivot", path=str(p))
            return 0
        mapping = {c: c for c in raw.columns}
    else:
        mapping = _auto_map_columns(list(raw.columns))

    if column_overrides:
        mapping.update(column_overrides)

    # Required fields
    for required in ("date", "home_team", "away_team"):
        if required not in mapping:
            log.warning("odds_csv.missing_required_column",
                        path=str(p), missing=required, available=list(raw.columns))
            return 0

    rows: list[dict] = []
    for _, src in raw.iterrows():
        game_date = _to_date(src.get(mapping["date"]))
        home = _normalize_team(src.get(mapping["home_team"]))
        away = _normalize_team(src.get(mapping["away_team"]))
        if not (game_date and home and away):
            continue
        rows.append({
            "game_date": game_date,
            "home_team_abbr": home,
            "away_team_abbr": away,
            "book": book_label,
            "ml_open_home":   _to_int(src.get(mapping.get("ml_open_home"))) if "ml_open_home" in mapping else None,
            "ml_open_away":   _to_int(src.get(mapping.get("ml_open_away"))) if "ml_open_away" in mapping else None,
            "ml_close_home":  _to_int(src.get(mapping.get("ml_close_home"))) if "ml_close_home" in mapping else None,
            "ml_close_away":  _to_int(src.get(mapping.get("ml_close_away"))) if "ml_close_away" in mapping else None,
            "rl_close_home":  _to_float(src.get(mapping.get("rl_close_home"))) if "rl_close_home" in mapping else None,
            "rl_close_home_price": _to_int(src.get(mapping.get("rl_close_home_price"))) if "rl_close_home_price" in mapping else None,
            "rl_close_away_price": _to_int(src.get(mapping.get("rl_close_away_price"))) if "rl_close_away_price" in mapping else None,
            "total_close":         _to_float(src.get(mapping.get("total_close"))) if "total_close" in mapping else None,
            "total_close_over":    _to_int(src.get(mapping.get("total_close_over"))) if "total_close_over" in mapping else None,
            "total_close_under":   _to_int(src.get(mapping.get("total_close_under"))) if "total_close_under" in mapping else None,
        })

    if not rows:
        log.info("odds_csv.no_rows_after_normalize", path=str(p))
        return 0

    df = pd.DataFrame(rows).drop_duplicates(
        subset=["game_date", "home_team_abbr", "away_team_abbr", "book"]
    )
    n = upsert_dataframe(
        df, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
    log.info("odds_csv.ingested", path=str(p), book=book_label, rows=n)
    return n


def ingest_directory(
    root: str | Path = "data/raw/odds",
    book_prefix: str = "",
) -> dict[str, int]:
    """Ingest every .csv/.xlsx under ``root``. Returns {filename: row_count}."""
    root_path = Path(root)
    if not root_path.exists():
        log.info("odds_csv.directory_missing", path=str(root_path))
        return {}

    results: dict[str, int] = {}
    for path in sorted(root_path.rglob("*")):
        if path.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
            continue
        book = f"{book_prefix}{_book_label_from_path(path)}" if book_prefix else None
        results[path.name] = ingest_csv(path, book=book)
    return results
