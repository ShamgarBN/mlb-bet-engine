"""App-facing service layer for the slumping-slugger page.

The analysis engine (:mod:`mlb_model.analysis.slugger_slump`) makes ~35 live
API calls per run (season totals + a game log and team-transactions per
flagged player + news), so the page never calls it inline. Instead we cache a
JSON snapshot per ``(season, date)`` under ``data/cache/slugger/`` — the page
reads the snapshot; the refresh endpoint recomputes it. Mirrors the existing
predictions-cache pattern in :mod:`mlb_model.app.services`.
"""

from __future__ import annotations

import csv
import json
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Any

from mlb_model.analysis import slugger_slump as ss
from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("app.slugger_service")

CACHE_DIR = settings.cache_dir / "slugger"


def _cache_path(season: int, target: date_cls) -> Path:
    return CACHE_DIR / f"{season}-{target.isoformat()}.json"


def _headline_dict(h) -> dict[str, Any]:
    return {
        "title": h.title,
        "url": h.url,
        "source": h.source,
        "published": h.published.date().isoformat() if h.published else None,
    }


def _status_dict(s: ss.SluggerStatus) -> dict[str, Any]:
    row = ss.status_to_row(s)
    row["news"] = [_headline_dict(h) for h in s.news_headlines]
    return row


_MATCHUP_RANK = {"FAVORABLE": 0, "NEUTRAL": 1, "TOUGH": 2, "NONE": 3}


def _betting_priority(row: dict[str, Any]) -> tuple:
    """Sort key for the UI: bettable cold streaks first, injuries last.

    The page is for finding viable lines, so a player actively playing through
    a cold streak (UNCLEAR, not absent) sorts to the top; a verified IL injury
    isn't bettable and sinks to the bottom. Within the bettable group, a
    FAVORABLE pitching matchup ranks ahead of a tough one — that's the actual
    bounce-back spot — then the longer the HR drought (and bigger the bat).
    """
    label = row["status_label"]
    if label == "INJURY (verified)":
        bucket = 3
    elif label == "EXTERNAL (verified)":
        bucket = 2
    elif row["is_absent"]:
        bucket = 1
    else:
        bucket = 0
    match_rank = _MATCHUP_RANK.get(row.get("matchup_label", "NONE"), 3)
    return (bucket, match_rank, -int(row["drought_games"]), -int(row["home_runs"]))


def compute_snapshot(
    season: int,
    *,
    threshold: int = ss.DEFAULT_HR_THRESHOLD,
    min_drought: int = ss.DEFAULT_MIN_DROUGHT,
    as_of: date_cls | None = None,
    with_news: bool = True,
) -> dict[str, Any]:
    """Run the engine, persist a dated history snapshot, and cache the result."""
    as_of = as_of or date_cls.today()
    hitters = ss.fetch_season_hitting(season)
    breakdown = ss.threshold_breakdown(hitters, season=season, threshold=threshold, as_of=as_of)
    ss.append_history(breakdown)  # feed the moving-percentage series

    statuses = ss.find_slumping_sluggers(
        season,
        threshold=threshold,
        min_drought=min_drought,
        as_of=as_of,
        with_news=with_news,
        hitters=hitters,  # reuse — don't re-pull season totals
    )

    snapshot = {
        "season": season,
        "threshold": threshold,
        "min_drought": min_drought,
        "as_of": as_of.isoformat(),
        "computed_at": datetime.now().isoformat(timespec="seconds"),
        "n_at_threshold": breakdown.n_at_threshold,
        "shares": {
            k: {"label": v.label, "denom_count": v.denom_count, "pct": round(v.pct, 1)}
            for k, v in breakdown.shares.items()
        },
        "n_injury": sum(1 for s in statuses if s.verified_cause == "injury-verified"),
        "n_external": sum(1 for s in statuses if s.verified_cause == "external-verified"),
        "n_unclear": sum(1 for s in statuses if s.verified_cause == "unverified"),
        "n_viable": sum(
            1 for s in statuses if s.verified_cause == "unverified" and not s.is_absent
        ),
        "n_favorable": sum(1 for s in statuses if s.matchup_label == "FAVORABLE"),
        "sluggers": sorted((_status_dict(s) for s in statuses), key=_betting_priority),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(season, as_of).write_text(json.dumps(snapshot, indent=2))
    return snapshot


def load_snapshot(season: int, *, as_of: date_cls | None = None) -> dict[str, Any] | None:
    """Return today's cached snapshot if present, else None."""
    as_of = as_of or date_cls.today()
    path = _cache_path(season, as_of)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 -- defensive cache read
        log.warning("slugger.cache.read_failed", path=str(path), error=str(exc))
        return None


def get_report(
    season: int, *, refresh: bool = False, as_of: date_cls | None = None
) -> dict[str, Any]:
    """Cached snapshot for the page; recomputes on ``refresh`` or cache miss."""
    if not refresh:
        cached = load_snapshot(season, as_of=as_of)
        if cached is not None:
            return cached
    return compute_snapshot(season, as_of=as_of)


def history_series(season: int, *, threshold: int = ss.DEFAULT_HR_THRESHOLD) -> dict[str, list[dict]]:
    """The recorded moving-percentage series, grouped by denominator.

    Returns ``{denominator: [{"x": "2026-06-22", "y": 15.9}, ...]}`` ready to
    hand to Chart.js. Empty dict if no history has been recorded yet.
    """
    if not ss.HISTORY_PATH.exists():
        return {}
    series: dict[str, list[dict]] = {}
    with ss.HISTORY_PATH.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["season"]) != season or int(r["threshold"]) != threshold:
                continue
            series.setdefault(r["denominator"], []).append(
                {"x": r["as_of"], "y": round(float(r["pct"]), 2)}
            )
    for rows in series.values():
        rows.sort(key=lambda p: p["x"])
    return series
