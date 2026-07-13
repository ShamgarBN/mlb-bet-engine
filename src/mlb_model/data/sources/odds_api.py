"""The Odds API ingestion (the-odds-api.com).

Free tier provides live + historical odds (recent past) with a generous
request budget. Best used for:

* live odds on today's slate (for in-app display + journal capture)
* opening lines for line-movement features on current games

Requires an API key in ``MLB_ODDS_API_KEY`` env var (free signup at
https://the-odds-api.com/). The var accepts a comma-separated list of
keys: when the active key's monthly quota runs out (HTTP 401
``OUT_OF_USAGE_CREDITS``) the fetch fails over to the next key in the
list. All calls are best-effort; failures log and return 0 so they
never break a pipeline.

Notes on the API:
* ``sport`` for MLB is ``baseball_mlb``.
* Free tier: 500 req/month. One ``/odds`` call returns the full slate
  (one row per game), so a single daily pull is enough for live data.
* The free historical endpoint goes back ~1 year; multi-year backfills
  require a paid tier (or use ``odds_csv`` for historicals).
"""

from __future__ import annotations

import os
from datetime import date, datetime, UTC

import httpx
import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.odds_api")

_BASE_URL = "https://api.the-odds-api.com/v4"
_SPORT = "baseball_mlb"
_DEFAULT_BOOKS = "draftkings,fanduel,betmgm,caesars,pinnacle"


def _api_keys() -> list[str]:
    """Read the API key(s) from settings (which loads PROJECT_ROOT/.env) and
    fall back to the raw env var so an exported shell variable still works.

    Accepts a comma-separated list; order is priority order for failover.
    """
    raw = settings.odds_api_key or os.environ.get("MLB_ODDS_API_KEY") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def _short_team(name: str) -> str | None:
    """The Odds API returns full team names; map to our 3-letter abbrs."""
    from mlb_model.data.sources.odds_csv import _normalize_team
    return _normalize_team(name)


def _american(price: float | None) -> int | None:
    """The Odds API returns decimal odds; convert to American for the warehouse."""
    if price is None:
        return None
    try:
        d = float(price)
    except (TypeError, ValueError):
        return None
    if d <= 1.0:
        return None
    if d >= 2.0:
        return int(round((d - 1.0) * 100.0))
    return int(round(-100.0 / (d - 1.0)))


def _best_market(bookmakers: list[dict], market_key: str) -> dict | None:
    """Pick the first bookmaker that quoted ``market_key`` and return its outcomes."""
    for bm in bookmakers:
        for m in bm.get("markets", []):
            if m.get("key") == market_key and m.get("outcomes"):
                return {"bookmaker": bm.get("key"), "outcomes": m["outcomes"]}
    return None


def fetch_live_slate(books: str = _DEFAULT_BOOKS) -> pd.DataFrame:
    """Pull live odds for upcoming + in-progress MLB games. Returns warehouse-shaped rows."""
    keys = _api_keys()
    if not keys:
        log.info("odds_api.no_api_key")
        return pd.DataFrame()

    url = f"{_BASE_URL}/sports/{_SPORT}/odds"
    payload = None
    for idx, key in enumerate(keys, start=1):
        params = {
            "apiKey": key,
            "regions": "us",
            "markets": "h2h,spreads,totals",
            "oddsFormat": "decimal",
            "bookmakers": books,
        }
        try:
            with httpx.Client(timeout=20.0) as c:
                resp = c.get(url, params=params)
        except Exception as exc:
            # Log the exception type only -- httpx error messages embed the
            # request URL, which carries the API key.
            log.error("odds_api.request_failed", key_index=idx, error=type(exc).__name__)
            return pd.DataFrame()

        # 401 = quota exhausted (OUT_OF_USAGE_CREDITS) or bad key;
        # 429 = per-key rate limit. Both are key-specific -> next key.
        if resp.status_code in (401, 429):
            try:
                error_code = resp.json().get("error_code")
            except Exception:
                error_code = None
            log.warning(
                "odds_api.key_rejected",
                key_index=idx, keys_total=len(keys),
                status=resp.status_code, error_code=error_code,
            )
            continue
        if resp.status_code != 200:
            log.error("odds_api.http_error", key_index=idx, status=resp.status_code)
            return pd.DataFrame()

        payload = resp.json()
        remaining = resp.headers.get("x-requests-remaining")
        if idx > 1:
            log.warning(
                "odds_api.failover_used", key_index=idx, credits_remaining=remaining,
            )
        else:
            log.info("odds_api.fetched", credits_remaining=remaining)
        break

    if payload is None:
        log.error("odds_api.all_keys_exhausted", keys_total=len(keys))
        return pd.DataFrame()

    rows: list[dict] = []
    for game in payload:
        commence = game.get("commence_time")
        if not commence:
            continue
        try:
            game_date = pd.to_datetime(commence).date()
        except Exception:
            continue
        home = _short_team(game.get("home_team"))
        away = _short_team(game.get("away_team"))
        if not (home and away):
            continue

        bms = game.get("bookmakers") or []
        h2h = _best_market(bms, "h2h")
        spread = _best_market(bms, "spreads")
        total = _best_market(bms, "totals")

        ml_home = ml_away = None
        if h2h:
            for outcome in h2h["outcomes"]:
                price = _american(outcome.get("price"))
                if _short_team(outcome.get("name")) == home:
                    ml_home = price
                elif _short_team(outcome.get("name")) == away:
                    ml_away = price

        rl_line = rl_home_price = rl_away_price = None
        if spread:
            for outcome in spread["outcomes"]:
                price = _american(outcome.get("price"))
                point = outcome.get("point")
                if _short_team(outcome.get("name")) == home:
                    rl_line = float(point) if point is not None else None
                    rl_home_price = price
                elif _short_team(outcome.get("name")) == away:
                    rl_away_price = price

        total_line = over_price = under_price = None
        if total:
            for outcome in total["outcomes"]:
                price = _american(outcome.get("price"))
                point = outcome.get("point")
                label = (outcome.get("name") or "").lower()
                if total_line is None and point is not None:
                    total_line = float(point)
                if label == "over":
                    over_price = price
                elif label == "under":
                    under_price = price

        rows.append({
            "game_date": game_date,
            "home_team_abbr": home,
            "away_team_abbr": away,
            "book": "odds_api",
            "ml_open_home": None,
            "ml_open_away": None,
            "ml_close_home": ml_home,
            "ml_close_away": ml_away,
            "rl_close_home": rl_line,
            "rl_close_home_price": rl_home_price,
            "rl_close_away_price": rl_away_price,
            "total_close": total_line,
            "total_close_over": over_price,
            "total_close_under": under_price,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(
        subset=["game_date", "home_team_abbr", "away_team_abbr", "book"]
    )
    return df


def ingest_live_slate() -> int:
    """Fetch + upsert today's live odds. Returns rows written."""
    df = fetch_live_slate()
    if df.empty:
        return 0
    n = upsert_dataframe(
        df, "odds_history",
        key_columns=["game_date", "home_team_abbr", "away_team_abbr", "book"],
    )
    log.info("odds_api.ingested", rows=n, when=datetime.now(UTC).isoformat())
    return n
