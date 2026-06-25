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

# Bump whenever the snapshot dict / slugger row shape changes. A cached
# snapshot written by an older app version (different schema) is then ignored
# and recomputed, instead of being served to a newer template that expects new
# fields — which would 500. The cache persists in the user's Application
# Support dir across app upgrades, so this guard is essential.
SNAPSHOT_SCHEMA = 2


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
        bucket = 4
    elif label == "EXTERNAL (verified)":
        bucket = 3
    elif row.get("active_today"):
        bucket = 0   # the primary target: bettable cold bat playing today
    elif row.get("plays_today"):
        bucket = 1   # plays today but day-to-day / sitting — questionable
    else:
        bucket = 2   # team off today (or game already done) — not actionable today
    match_rank = _MATCHUP_RANK.get(row.get("matchup_label", "NONE"), 3)
    return (bucket, match_rank, -int(row["drought_games"]), -int(row["home_runs"]))


def compute_snapshot(
    season: int,
    *,
    threshold: int | None = None,
    target_pct: float = ss.DEFAULT_TARGET_PCT,
    min_drought: int = ss.DEFAULT_MIN_DROUGHT,
    as_of: date_cls | None = None,
    with_news: bool = True,
) -> dict[str, Any]:
    """Run the engine, persist a dated history snapshot, and cache the result.

    ``threshold`` defaults to a dynamic bar tracking the top ``target_pct`` of
    qualified hitters by HR (so it rises through the season); pass an int to
    force a fixed bar.
    """
    as_of = as_of or date_cls.today()
    hitters = ss.fetch_season_hitting(season)
    if threshold is None:
        threshold = ss.dynamic_threshold(hitters, target_pct=target_pct)
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
        "schema": SNAPSHOT_SCHEMA,
        "season": season,
        "threshold": threshold,
        "target_pct": target_pct,
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
        "n_active_today": sum(1 for s in statuses if s.active_today),
        "n_favorable": sum(1 for s in statuses if s.active_today and s.matchup_label == "FAVORABLE"),
        "sluggers": sorted((_status_dict(s) for s in statuses), key=_betting_priority),
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(season, as_of).write_text(json.dumps(snapshot, indent=2))
    return snapshot


def load_snapshot(season: int, *, as_of: date_cls | None = None) -> dict[str, Any] | None:
    """Return today's cached snapshot if present AND schema-compatible.

    A snapshot written by an older app version is ignored (returns None →
    recompute) so a newer template never renders against a missing field.
    """
    as_of = as_of or date_cls.today()
    path = _cache_path(season, as_of)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 -- defensive cache read
        log.warning("slugger.cache.read_failed", path=str(path), error=str(exc))
        return None
    if not isinstance(data, dict) or data.get("schema") != SNAPSHOT_SCHEMA:
        log.info("slugger.cache.schema_mismatch", path=str(path),
                 found=data.get("schema") if isinstance(data, dict) else None,
                 expected=SNAPSHOT_SCHEMA)
        return None
    return data


def get_report(
    season: int, *, refresh: bool = False, as_of: date_cls | None = None
) -> dict[str, Any]:
    """Cached snapshot for the page; recomputes on ``refresh`` or cache miss."""
    if not refresh:
        cached = load_snapshot(season, as_of=as_of)
        if cached is not None:
            return cached
    return compute_snapshot(season, as_of=as_of)


def threshold_series(season: int) -> list[dict]:
    """The HR bar over the season — one point per recorded day.

    With a dynamic top-N% bar the "share at threshold" is ~constant by design,
    so the meaningful trend is the bar itself rising. Returns
    ``[{"x": "2026-06-22", "y": 15}, ...]`` (the recorded threshold per day).
    Empty list if no history yet.
    """
    if not ss.HISTORY_PATH.exists():
        return []
    by_day: dict[str, int] = {}
    with ss.HISTORY_PATH.open(newline="") as fh:
        for r in csv.DictReader(fh):
            if int(r["season"]) != season:
                continue
            # One threshold per (day, season); rows repeat it per denominator.
            by_day[r["as_of"]] = int(r["threshold"])
    return [{"x": d, "y": by_day[d]} for d in sorted(by_day)]
