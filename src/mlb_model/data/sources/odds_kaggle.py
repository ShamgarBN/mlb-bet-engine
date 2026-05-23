"""Ingesters for the specific Kaggle MLB-odds datasets the user has.

Two formats supported:

* **Historical archive (2012-2021)** — Pair of CSVs:
  * ``oddsDataMLB.csv`` — one row per (team's perspective) with moneyline,
    run-line, total, and the opponent's lines too. Lacks V/H markers.
  * ``oddsData.csv``    — one row per (team's perspective) with V/H marker
    but a sparser column set.
  We use ``oddsData.csv`` to determine home/away, then pull the richer
  fields from ``oddsDataMLB.csv``. Stamped as ``book='kaggle_historical'``.

* **Live snapshot (current season)** —
  ``mlb_odds_kaggle_states.csv`` is a long-form per-(event, book, market,
  outcome, snapshot) dump from The Odds API. We pick the *latest*
  snapshot per event-book-market-outcome that arrived before
  ``commence_time`` (i.e., the closing line as of the most recent
  pre-game observation). Stamped as ``book='kaggle_states_<book_key>'``
  so each sportsbook is tracked separately.

Both ingesters are idempotent — keys in ``odds_history`` are
``(game_date, home_team_abbr, away_team_abbr, book)`` so re-runs upsert
in place.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_model.data.sources.odds_csv import _normalize_team
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.odds_kaggle")


# --------------------------------------------------------------------------- #
# Historical archive (oddsDataMLB.csv + oddsData.csv)                         #
# --------------------------------------------------------------------------- #


def _to_int(v) -> int | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NONE"}:
        return None
    try:
        return int(float(s))
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    s = str(v).strip()
    if s == "" or s.upper() in {"NA", "NAN", "NONE"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def ingest_historical(
    archive_dir: str | Path,
    book: str = "kaggle_historical",
) -> int:
    """Ingest the paired oddsDataMLB.csv + oddsData.csv files.

    ``archive_dir`` should contain both files. Returns rows upserted.
    """
    root = Path(archive_dir)
    main_path = root / "oddsDataMLB.csv"
    vh_path = root / "oddsData.csv"

    if not main_path.exists():
        log.warning("kaggle.historical.missing_main", path=str(main_path))
        return 0
    if not vh_path.exists():
        log.warning("kaggle.historical.missing_vh", path=str(vh_path))
        return 0

    try:
        rich = pd.read_csv(main_path)
        vh = pd.read_csv(vh_path)
    except Exception:
        log.exception("kaggle.historical.read_failed", path=str(root))
        return 0

    # Build (date, team) → 'V'/'H' lookup.
    vh = vh[["date", "at", "team"]].copy()
    vh["date"] = pd.to_datetime(vh["date"], errors="coerce").dt.date
    vh = vh.dropna(subset=["date", "team", "at"])
    vh_lookup: dict[tuple, str] = {
        (row.date, str(row.team).strip().upper()): str(row.at).strip().upper()
        for row in vh.itertuples(index=False)
    }
    if not vh_lookup:
        log.warning("kaggle.historical.empty_vh_lookup")
        return 0

    # Walk the rich file; keep one row per game (the home-team perspective).
    rich = rich.copy()
    rich["date"] = pd.to_datetime(rich["date"], errors="coerce").dt.date
    rich = rich.dropna(subset=["date", "team", "opponent"])

    rows: list[dict] = []
    for r in rich.itertuples(index=False):
        team = _normalize_team(r.team)
        opp = _normalize_team(r.opponent)
        if not team or not opp:
            continue

        # Use the raw (unnormalized) abbr from the CSV for the lookup —
        # both files were exported from the same source so they agree on
        # spelling, but normalize to the warehouse abbr for the row.
        side = vh_lookup.get((r.date, str(r.team).strip().upper()))
        if side is None:
            # Try the opponent side as a fallback.
            opp_side = vh_lookup.get((r.date, str(r.opponent).strip().upper()))
            if opp_side == "V":
                side = "H"
            elif opp_side == "H":
                side = "V"
        if side != "H":
            # Skip visitor-perspective rows — we'll get the home row for
            # the same game (or the opponent's row if "team" was the away).
            continue

        rows.append({
            "game_date": r.date,
            "home_team_abbr": team,
            "away_team_abbr": opp,
            "book": book,
            "ml_open_home": None,
            "ml_open_away": None,
            "ml_close_home": _to_int(r.moneyLine),
            "ml_close_away": _to_int(r.oppMoneyLine),
            "rl_close_home": _to_float(r.runLine),
            "rl_close_home_price": _to_int(r.runLineOdds),
            "rl_close_away_price": _to_int(r.oppRunLineOdds),
            "total_close": _to_float(r.total),
            "total_close_over": _to_int(r.overOdds),
            "total_close_under": _to_int(r.underOdds),
        })

    if not rows:
        log.warning("kaggle.historical.no_rows_after_filter")
        return 0
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["game_date", "home_team_abbr", "away_team_abbr", "book"]
    )
    n = upsert_dataframe(
        df, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
    log.info("kaggle.historical.ingested", rows=n, book=book)
    return n


# --------------------------------------------------------------------------- #
# Snapshot dump (mlb_odds_kaggle_states.csv)                                  #
# --------------------------------------------------------------------------- #


def _coerce_american(price) -> int | None:
    """The states file stores American odds in the ``price`` column directly."""
    return _to_int(price)


def ingest_states(
    csv_path: str | Path,
    book_prefix: str = "kaggle_states_",
) -> int:
    """Ingest the long-form Odds-API-shaped snapshot dump.

    For each (event, bookmaker, market, outcome), pick the most recent
    snapshot whose ``effective_time`` ≤ ``commence_time`` (i.e. the line
    as of the latest pre-game observation). Returns total rows upserted.

    One row per (event, bookmaker) lands in ``odds_history`` with
    ``book = '{book_prefix}{bookmaker_key}'``.
    """
    path = Path(csv_path)
    if not path.exists():
        log.warning("kaggle.states.missing", path=str(path))
        return 0

    try:
        df = pd.read_csv(
            path,
            dtype={"price": "string", "point": "string"},
            low_memory=False,
        )
    except Exception:
        log.exception("kaggle.states.read_failed", path=str(path))
        return 0

    # Coerce timestamps; drop rows where we can't tell pre/post-game.
    for col in ("effective_time", "commence_time"):
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    df = df.dropna(subset=["effective_time", "commence_time"])

    # Keep only pre-game observations and the markets we care about.
    df = df[df["effective_time"] <= df["commence_time"]]
    df = df[df["market_key"].isin(["h2h", "spreads", "totals"])]
    if df.empty:
        log.info("kaggle.states.empty_after_filter")
        return 0

    # For each (event, bookmaker, market, outcome): keep latest snapshot.
    df = df.sort_values("effective_time")
    keyed = ["event_id", "bookmaker_key", "market_key", "outcome_name"]
    latest = df.drop_duplicates(subset=keyed, keep="last").copy()

    # Build per-event-per-book wide rows.
    rows: list[dict] = []
    grouped = latest.groupby(["event_id", "bookmaker_key"], sort=False)
    for (_event_id, book_key), g in grouped:
        # commence_time is constant within an event; pick the first.
        commence = g["commence_time"].iloc[0]
        game_date = commence.date()
        # Team names from the event metadata (same across all rows).
        home_team = _normalize_team(g["home_team"].iloc[0])
        away_team = _normalize_team(g["away_team"].iloc[0])
        if not (home_team and away_team):
            continue

        ml_home = ml_away = None
        rl_home = rl_home_price = rl_away_price = None
        total_line = over_price = under_price = None

        for r in g.itertuples(index=False):
            mkt = r.market_key
            outcome = _normalize_team(r.outcome_name) or r.outcome_name
            price = _coerce_american(r.price)
            point = _to_float(r.point)

            if mkt == "h2h":
                if outcome == home_team:
                    ml_home = price
                elif outcome == away_team:
                    ml_away = price
            elif mkt == "spreads":
                if outcome == home_team:
                    rl_home = point
                    rl_home_price = price
                elif outcome == away_team:
                    rl_away_price = price
            elif mkt == "totals":
                label = (str(r.outcome_name) or "").strip().lower()
                if total_line is None and point is not None:
                    total_line = point
                if label == "over":
                    over_price = price
                elif label == "under":
                    under_price = price

        rows.append({
            "game_date": game_date,
            "home_team_abbr": home_team,
            "away_team_abbr": away_team,
            "book": f"{book_prefix}{book_key}",
            "ml_open_home": None,
            "ml_open_away": None,
            "ml_close_home": ml_home,
            "ml_close_away": ml_away,
            "rl_close_home": rl_home,
            "rl_close_home_price": rl_home_price,
            "rl_close_away_price": rl_away_price,
            "total_close": total_line,
            "total_close_over": over_price,
            "total_close_under": under_price,
        })

    if not rows:
        return 0
    out = pd.DataFrame(rows).drop_duplicates(
        subset=["game_date", "home_team_abbr", "away_team_abbr", "book"]
    )
    n = upsert_dataframe(
        out, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
    log.info("kaggle.states.ingested", rows=n, books=out["book"].nunique())
    return n


# --------------------------------------------------------------------------- #
# Bulk discovery                                                              #
# --------------------------------------------------------------------------- #


def ingest_all(root: str | Path = "kaggle-data") -> dict[str, int]:
    """Walk ``root`` and ingest every recognized Kaggle dataset.

    Returns ``{source_label: rows_upserted}``. Safe to re-run.
    """
    root_path = Path(root)
    results: dict[str, int] = {}

    if not root_path.exists():
        log.warning("kaggle.root.missing", path=str(root_path))
        return results

    # Historical archives — there are often duplicate copies (archive/,
    # archive(2)/, etc.). We ingest just the first one we find with a
    # complete pair; the others are bit-identical so re-ingesting is
    # redundant.
    historical_done = False
    for sub in sorted(root_path.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "oddsDataMLB.csv").exists() and (sub / "oddsData.csv").exists():
            if historical_done:
                log.info("kaggle.historical.duplicate_skipped", path=str(sub))
                continue
            n = ingest_historical(sub)
            results[f"historical:{sub.name}"] = n
            if n > 0:
                historical_done = True

    # Snapshot dump
    for path in root_path.rglob("mlb_odds_kaggle_states.csv"):
        n = ingest_states(path)
        results[f"states:{path.parent.name}"] = n

    return results
