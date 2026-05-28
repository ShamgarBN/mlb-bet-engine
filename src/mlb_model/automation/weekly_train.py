"""Weekend self-improvement: retrain models on the freshest data.

Once a week we:

1) Pull any completed games since the last training run.
2) Recompute features and re-fit the home/away runs models + totals
   model + calibrators against the rolling training window.
3) Archive the previous ``models/*.joblib`` to ``models/archive/<ts>/``
   so we can roll back if the new model regresses (we keep the most
   recent 5 archives).
4) Run a small validation pass against the past 14 days and refuse to
   promote the new model if its moneyline accuracy drops by more than
   2 percentage points -- protects the user against a bad refit.
5) Write ``data/cache/last_weekly_train.txt`` so the desktop app can
   tell when retraining last completed.

The CLI command ``mlb-model weekly-train`` invokes this, and the
desktop app runs ``needs_run`` lazily on launch (Sunday only) so the
user doesn't have to install a cron job.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("automation.weekly_train")

MARKER_PATH = settings.cache_dir / "last_weekly_train.txt"
ARCHIVE_ROOT = settings.model_dir / "archive"
KEEP_ARCHIVES = 5
ACCURACY_REGRESSION_TOLERANCE_PP = 2.0  # percentage points
TRAINING_WINDOW_YEARS = 6

# Touched while ``run_weekly_train`` is in flight so the UI can disable
# the "Retrain now" button. The contents are the ISO start time so a
# stale lock from a crashed previous run can be detected (>30 min old
# is considered abandoned).
STATUS_PATH = settings.cache_dir / "weekly_train_status.txt"
RESULT_PATH = settings.cache_dir / "weekly_train_result.json"
STALE_RUN_MINUTES = 45


def last_run_date() -> date | None:
    if not MARKER_PATH.exists():
        return None
    try:
        return datetime.fromisoformat(MARKER_PATH.read_text(encoding="utf-8").strip()).date()
    except (ValueError, OSError):
        return None


def needs_run(today: date | None = None, *, min_days: int = 7) -> bool:
    """True on Sundays when the last train was more than ``min_days`` ago."""
    today = today or date.today()
    if today.weekday() != 6:  # 0=Mon ... 6=Sun
        return False
    last = last_run_date()
    return last is None or (today - last).days >= min_days


def _record_success() -> None:
    MARKER_PATH.parent.mkdir(parents=True, exist_ok=True)
    MARKER_PATH.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def _archive_current_models() -> Path | None:
    """Move existing trained artefacts into a timestamped archive folder."""
    if not settings.model_dir.exists():
        return None
    artefacts = [p for p in settings.model_dir.glob("*.joblib")]
    if not artefacts:
        return None

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_ROOT / stamp
    dest.mkdir(parents=True, exist_ok=True)
    for p in artefacts:
        shutil.copy2(p, dest / p.name)
    log.info("weekly_train.archive.created", path=str(dest), files=len(artefacts))

    # Keep only the most recent ``KEEP_ARCHIVES``.
    existing = sorted(
        [p for p in ARCHIVE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    for stale in existing[KEEP_ARCHIVES:]:
        shutil.rmtree(stale, ignore_errors=True)
        log.info("weekly_train.archive.pruned", path=str(stale))
    return dest


def _restore_from_archive(archive_dir: Path) -> None:
    for p in archive_dir.glob("*.joblib"):
        shutil.copy2(p, settings.model_dir / p.name)
    log.warning("weekly_train.archive.restored", path=str(archive_dir))


def _recent_moneyline_accuracy(days: int = 14) -> float | None:
    """Score the *current* models against the most recent finalized games.

    Returns the moneyline accuracy on (today - days, today). ``None`` if
    we don't have enough finalized games to evaluate.
    """
    from mlb_model.data.warehouse import query
    from mlb_model.predict.daily import predict_for_date

    today = date.today()
    start = today - timedelta(days=days)

    # Need at least 30 finalized games for a meaningful number.
    df = query(
        """
        SELECT game_date, COUNT(*) AS n
        FROM games
        WHERE status = 'Final' AND game_date BETWEEN ? AND ?
        GROUP BY game_date
        ORDER BY game_date
        """,
        (start, today),
    )
    if df.empty or int(df["n"].sum()) < 30:
        return None

    hits = 0
    total = 0
    for d in df["game_date"]:
        try:
            preds = predict_for_date(d if isinstance(d, date) else pd.to_datetime(d).date())
        except Exception:  # noqa: BLE001 -- skip days the model can't predict
            continue
        if preds.empty:
            continue
        # Join in actual outcomes.
        outcomes = query(
            """
            SELECT game_pk, home_score, away_score
            FROM games WHERE game_date = ? AND status = 'Final'
            """,
            (d,),
        )
        if outcomes.empty:
            continue
        merged = preds.merge(outcomes, on="game_pk", how="inner")
        merged["pick_home"] = merged["p_home_win"] >= 0.5
        merged["home_won"] = merged["home_score"] > merged["away_score"]
        hits += int(((merged["pick_home"]) == (merged["home_won"])).sum())
        total += int(len(merged))
    if total < 30:
        return None
    return hits / total


def run_weekly_train(through_season: int | None = None) -> dict[str, Any]:
    """Refit models if we have new data; refuse to promote if it regresses."""
    today = date.today()
    through_season = through_season or today.year

    log.info("weekly_train.start", through_season=through_season)

    # 1) Benchmark current models so we can detect regression.
    baseline_acc = _recent_moneyline_accuracy(days=14)
    log.info("weekly_train.baseline", recent_ml_acc=baseline_acc)

    # 2) Archive current models.
    archive = _archive_current_models()

    # 3) Refit. Call the canonical CLI training function directly so the
    #    walk-forward backtest stays the single source of truth -- no
    #    duplicated training pipeline to drift. Direct call (instead of a
    #    subprocess) is also what lets this work inside a PyInstaller
    #    bundle, where ``sys.executable`` is the .app binary and
    #    ``python -m mlb_model`` is not a valid invocation.
    from mlb_model.cli import train as _train_cmd

    train_window_start = through_season - TRAINING_WINDOW_YEARS
    try:
        _train_cmd(through_season=through_season, train_start=train_window_start)
    except SystemExit as exc:  # typer.Exit subclasses SystemExit
        # SystemExit(0) means a clean "early return" from train (e.g.
        # nothing to do). Anything else means training failed.
        if int(getattr(exc, "code", 0) or 0) != 0:
            log.exception("weekly_train.refit.failed")
            if archive is not None:
                _restore_from_archive(archive)
            raise
    except Exception:  # noqa: BLE001 -- restore archive on any failure
        log.exception("weekly_train.refit.failed")
        if archive is not None:
            _restore_from_archive(archive)
        raise

    # 4) Validate the new model. If it regresses materially, restore.
    new_acc = _recent_moneyline_accuracy(days=14)
    log.info("weekly_train.validation", new_ml_acc=new_acc, baseline=baseline_acc)

    if (
        baseline_acc is not None and new_acc is not None
        and (baseline_acc - new_acc) * 100.0 > ACCURACY_REGRESSION_TOLERANCE_PP
    ):
        log.warning(
            "weekly_train.regression.detected",
            baseline=baseline_acc, new=new_acc,
            drop_pp=(baseline_acc - new_acc) * 100.0,
        )
        if archive is not None:
            _restore_from_archive(archive)
        return {
            "promoted": False,
            "reason": "regression",
            "baseline_ml_acc": baseline_acc,
            "new_ml_acc": new_acc,
        }

    # 5) Invalidate cached predictions so the dashboard picks up the new model.
    cache_dir = settings.cache_dir / "predictions"
    if cache_dir.exists():
        for f in cache_dir.glob("*.parquet"):
            f.unlink(missing_ok=True)

    _record_success()
    log.info("weekly_train.done")

    # 6) If the regular season just wrapped and we have NOT yet produced
    #    an end-of-season report for it, kick one off now. This is what
    #    turns the weekly retrain into an annual self-evaluation loop --
    #    the user doesn't have to remember to run it.
    end_of_season_result: dict[str, Any] | None = None
    try:
        end_of_season_result = _maybe_trigger_end_of_season(through_season)
    except Exception:  # noqa: BLE001 -- never let the sweep failure mask a successful retrain
        log.exception("weekly_train.end_of_season.failed")

    return {
        "promoted": True,
        "baseline_ml_acc": baseline_acc,
        "new_ml_acc": new_acc,
        "archive": str(archive) if archive else None,
        "end_of_season": end_of_season_result,
    }


def _maybe_trigger_end_of_season(through_season: int) -> dict[str, Any] | None:
    """If ``through_season``'s regular season is over and we haven't yet
    written its end-of-season report, write it now.

    Returns a small status dict (or None if we skipped). Idempotent --
    the function checks ``reports/end_of_season_<SEASON>/summary.json``
    as a "done" marker so weekly retrains for the rest of the off-season
    don't keep re-running it.
    """
    from mlb_model.season import is_regular_season_over, run_end_of_season_sweep

    if not is_regular_season_over(through_season):
        return None

    report_dir = settings.project_root / "reports" / f"end_of_season_{through_season}"
    marker = report_dir / "summary.json"
    if marker.exists():
        log.info(
            "weekly_train.end_of_season.already_done",
            season=through_season, marker=str(marker),
        )
        return {"skipped": True, "reason": "already_complete", "season": through_season}

    log.info("weekly_train.end_of_season.start", season=through_season)
    report = run_end_of_season_sweep(season=through_season)
    log.info(
        "weekly_train.end_of_season.done",
        season=through_season,
        output_dir=report.output_dir,
    )
    return {
        "season": through_season,
        "output_dir": report.output_dir,
        "report_md": report.report_md_path,
    }


# ---------------------------------------------------------------------------
# UI-driven async retrain (the "Retrain now" button on /season)
# ---------------------------------------------------------------------------


def is_running() -> bool:
    """True iff a retrain is in flight. Stale locks (>STALE_RUN_MINUTES old)
    from a crashed previous process are treated as not-running.
    """
    if not STATUS_PATH.exists():
        return False
    try:
        started = datetime.fromisoformat(STATUS_PATH.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    age_min = (datetime.now() - started).total_seconds() / 60.0
    if age_min > STALE_RUN_MINUTES:
        STATUS_PATH.unlink(missing_ok=True)
        return False
    return True


def last_result() -> dict[str, Any] | None:
    """Return the last ``run_weekly_train`` result dict written by the
    UI-driven background runner. Distinct from ``last_run_date()`` which
    only reports the success marker for CLI/Sunday runs.
    """
    if not RESULT_PATH.exists():
        return None
    try:
        import json
        return json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def start_background_run(through_season: int | None = None) -> bool:
    """Spawn ``run_weekly_train`` in a daemon thread. Returns False if
    a retrain is already in flight (caller should surface a 409).
    """
    import json
    import threading

    if is_running():
        return False
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")

    def _worker() -> None:
        try:
            result = run_weekly_train(through_season=through_season)
            payload = {"ok": True, "finished_at": datetime.now().isoformat(), "result": result}
        except Exception as exc:  # noqa: BLE001 -- want the UI to see failures
            log.exception("weekly_train.background.failed")
            payload = {
                "ok": False,
                "finished_at": datetime.now().isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        try:
            RESULT_PATH.write_text(json.dumps(payload, default=str), encoding="utf-8")
        except OSError:
            log.exception("weekly_train.background.result_write_failed")
        finally:
            STATUS_PATH.unlink(missing_ok=True)

    threading.Thread(target=_worker, daemon=True, name="weekly-train").start()
    return True
