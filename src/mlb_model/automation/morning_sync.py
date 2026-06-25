"""Morning data refresh.

Pulls yesterday's finalized games (boxscores -> final scores -> picks
log can be graded) and today's schedule/probable-pitchers/weather so the
day's predictions are based on the latest available data.

Designed to be idempotent and safe to run repeatedly: every upsert
checks primary keys before writing, and ``last_morning_sync.txt`` makes
"already done today" easy to detect.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from mlb_model.config import settings
from mlb_model.data import pipeline
from mlb_model.data.sources import mlb_statsapi
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("automation.morning_sync")

MARKER_PATH = settings.cache_dir / "last_morning_sync.txt"


def last_run_date() -> date | None:
    """Return the date of the last successful morning sync, or None."""
    if not MARKER_PATH.exists():
        return None
    try:
        text = MARKER_PATH.read_text(encoding="utf-8").strip()
        return datetime.fromisoformat(text).date()
    except (ValueError, OSError):
        return None


def needs_run(today: date | None = None) -> bool:
    """True if we have not synced yet today."""
    today = today or date.today()
    last = last_run_date()
    return last is None or last < today


def _record_success(when: datetime | None = None) -> None:
    when = when or datetime.now()
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(when.isoformat(timespec="seconds"), encoding="utf-8")


def _refresh_finals_for_date(target: date) -> dict[str, int]:
    """Re-fetch schedule + boxscores for one day so finals land in DB."""
    raw = mlb_statsapi.fetch_schedule(target, target)
    games_df = mlb_statsapi.normalize_schedule(raw)
    if games_df.empty:
        log.info("morning_sync.no_schedule", date=target.isoformat())
        return {"games": 0, "team_boxscores": 0, "lineups": 0}

    counts: dict[str, int] = {
        "games": upsert_dataframe(games_df, "games", key_columns=["game_pk"]),
    }

    # For every "Final" game returned, fetch the boxscore so we pick up
    # the actual final score (the schedule endpoint sometimes shows runs
    # but the boxscore is authoritative).
    finals = games_df[games_df["status"] == "Final"]
    if not finals.empty:
        from mlb_model.data.sources import mlb_statsapi as ms

        for pk in finals["game_pk"].astype(int):
            try:
                payload = ms.fetch_boxscore(int(pk))
            except Exception:  # noqa: BLE001 -- log + continue per game
                log.warning("morning_sync.boxscore.failed", game_pk=int(pk))
                continue
            try:
                upsert_dataframe(
                    ms.normalize_team_boxscore(int(pk), payload),
                    "team_boxscores",
                    key_columns=["game_pk", "team_id"],
                )
                upsert_dataframe(
                    ms.normalize_lineup(int(pk), payload),
                    "lineups",
                    key_columns=["game_pk", "team_id", "batting_order"],
                )
                upsert_dataframe(
                    ms.normalize_pitcher_stats(int(pk), payload),
                    "pitcher_game_stats",
                    key_columns=["game_pk", "pitcher_id"],
                )
                upsert_dataframe(
                    ms.normalize_batter_stats(int(pk), payload),
                    "batter_game_stats",
                    key_columns=["game_pk", "batter_id"],
                )
                counts["team_boxscores"] = counts.get("team_boxscores", 0) + 2
            except Exception:  # noqa: BLE001
                log.exception("morning_sync.boxscore.upsert_failed", game_pk=int(pk))

    return counts


def _maybe_notify_premium(today: date) -> None:
    """Emit a single macOS notification if today has any premium picks.

    Uses ``osascript`` so it works out-of-the-box on macOS without a
    third-party dep. We dedupe with a per-day marker so re-running
    morning_sync doesn't spam the user.
    """
    import platform
    import shutil
    import subprocess

    marker = settings.cache_dir / f"notify_premium_{today.isoformat()}.flag"
    if marker.exists():
        return

    # Generate predictions (cached) and count premium picks per market.
    try:
        from mlb_model.predict.daily import predict_for_date

        preds = predict_for_date(today, refresh_data=False)
    except Exception:  # noqa: BLE001
        log.warning("morning_sync.notify.predict_failed")
        return
    if preds is None or preds.empty:
        return

    # Premium = confidence >= 0.70 (matches services._confidence_tier).
    has_prem = (
        (preds.get("confidence_ml", 0).fillna(0) >= 0.70).any()
        or (preds.get("confidence_rl", 0).fillna(0) >= 0.70).any()
        or (preds.get("confidence_ou", 0).fillna(0) >= 0.70).any()
    )
    if not has_prem:
        return

    # On macOS only -- osascript is part of the OS so no install needed.
    if platform.system() != "Darwin" or shutil.which("osascript") is None:
        log.debug("morning_sync.notify.unsupported_platform")
        return

    n_games = int(preds.shape[0])
    title = "MLB Forecast — premium picks today"
    body = f"{n_games} games on the slate include high-conviction picks. Open the app to review."
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'display notification "{body}" with title "{title}" sound name "Glass"',
            ],
            check=False,
            timeout=5,
        )
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(datetime.now().isoformat(timespec="seconds"))
        log.info("morning_sync.notify.sent", date=today.isoformat())
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.notify.osascript_failed")


def _record_slugger_snapshot(today: date) -> Any:
    """Append today's 15+ HR share to the moving-percentage history.

    Lightweight on purpose — one season-totals API call, no per-player game
    logs or news (those run lazily when the Sluggers page is viewed). This is
    what makes the trend chart accumulate a point per day. Skipped out of
    season (no hitters with a PA yet) so we never write junk 0% rows.
    """
    from mlb_model.analysis import slugger_slump as ss

    hitters = ss.fetch_season_hitting(today.year)
    if not any(h.plate_appearances > 0 for h in hitters):
        log.info("morning_sync.slugger.skipped_offseason", season=today.year)
        return {"recorded": False, "reason": "no-pa"}
    # Record the dynamic (top-N%) bar so the trend chart tracks it rising.
    threshold = ss.dynamic_threshold(hitters)
    breakdown = ss.threshold_breakdown(
        hitters, season=today.year, threshold=threshold, as_of=today
    )
    ss.append_history(breakdown)
    return {
        "recorded": True,
        "threshold": threshold,
        "n_at_threshold": breakdown.n_at_threshold,
        "pct_qualified": round(breakdown.shares["qualified"].pct, 1),
    }


def run_morning_sync(today: date | None = None) -> dict[str, Any]:
    """Pull yesterday's finals + today's slate. Returns row counts."""
    today = today or date.today()
    yesterday = today - timedelta(days=1)

    log.info("morning_sync.start", today=today.isoformat(), yesterday=yesterday.isoformat())

    counts: dict[str, Any] = {"yesterday": yesterday.isoformat(), "today": today.isoformat()}

    # 1) Refresh yesterday's finals.
    try:
        counts["yesterday_results"] = _refresh_finals_for_date(yesterday)
    except Exception:  # noqa: BLE001 -- never fail the whole job for one date
        log.exception("morning_sync.yesterday.failed")
        counts["yesterday_results"] = {"error": True}

    # 2) Pull today's schedule + probable pitchers.
    try:
        counts["today_schedule"] = pipeline.pull_date_range(today, today)
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.today.failed")
        counts["today_schedule"] = {"error": True}

    # 3) Weather (best-effort; outdoor only).
    try:
        from mlb_model.data.sources import weather as _weather
        from mlb_model.data.venues_seed import seed_venues_df
        from mlb_model.data.warehouse import query

        games_df = query(
            """
            SELECT game_pk, venue_id, scheduled_start, home_team_abbr
            FROM games
            WHERE game_date = ?
              AND game_pk NOT IN (SELECT game_pk FROM weather)
            """,
            (today,),
        )
        if not games_df.empty:
            counts["weather"] = _weather.ingest_weather_for_games(games_df, seed_venues_df())
        else:
            counts["weather"] = 0
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.weather.failed")
        counts["weather"] = {"error": True}

    # 4) Invalidate today's cached prediction parquet so the next dashboard
    #    load re-runs with the freshly ingested probables / weather.
    cached = settings.cache_dir / "predictions" / f"{today.isoformat()}.parquet"
    try:
        cached.unlink(missing_ok=True)
        counts["prediction_cache_cleared"] = True
    except OSError:
        counts["prediction_cache_cleared"] = False

    # 5) Record the daily 15+ HR share so the Sluggers trend chart
    #    accumulates a point. Best-effort; never fails the sync.
    try:
        counts["slugger_snapshot"] = _record_slugger_snapshot(today)
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.slugger.failed")
        counts["slugger_snapshot"] = {"error": True}

    # macOS desktop notification when today's slate contains a
    # premium-tier pick. Best-effort -- fails silently on Linux, in
    # tests, when osascript is missing, etc. Only fires AFTER the
    # schedule/weather pull, so the predictions we'd otherwise emit
    # reflect today's actual fixtures.
    try:
        _maybe_notify_premium(today)
    except Exception:  # noqa: BLE001
        log.exception("morning_sync.notify.failed")

    # Only record success if the CRITICAL steps worked. Pulling today's
    # schedule is non-negotiable -- without it the user has no slate.
    # Yesterday's finals being late is sub-optimal but not fatal (the
    # auto-grader will pick them up tomorrow). Weather is best-effort
    # at all times.
    #
    # Previously this method always called _record_success() even when
    # the schedule pull blew up, so the dashboard footer showed a green
    # "synced today" date while the user actually had stale data and no
    # way to know.
    critical_failed = isinstance(counts.get("today_schedule"), dict) and counts["today_schedule"].get("error")
    if critical_failed:
        log.warning(
            "morning_sync.degraded",
            today=today.isoformat(),
            counts={k: v for k, v in counts.items() if not isinstance(v, dict)},
        )
        counts["status"] = "failed"
        counts["recorded"] = False
    else:
        # Capture sub-job degradation so the UI can surface "synced but
        # one of the steps fell over" without misrepresenting it as a
        # full failure.
        degraded_steps = [
            step for step in ("yesterday_results", "weather")
            if isinstance(counts.get(step), dict) and counts[step].get("error")
        ]
        _record_success()
        counts["recorded"] = True
        counts["status"] = "degraded" if degraded_steps else "ok"
        if degraded_steps:
            counts["degraded_steps"] = degraded_steps
            log.warning("morning_sync.partial", degraded_steps=degraded_steps)
    log.info("morning_sync.done", **{k: v for k, v in counts.items() if not isinstance(v, dict)})
    return counts
