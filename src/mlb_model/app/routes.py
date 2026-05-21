"""HTTP routes for the desktop app.

Three page-level routes (``/``, ``/game/<id>``, ``/performance``) and a
handful of JSON / HTMX-partial endpoints. Everything is rendered with
Jinja2 + Tailwind via CDN; no JS build step.

Security posture: the server binds to 127.0.0.1 only. The pick-logging
endpoint accepts arbitrary fields but treats every value as an opaque
string that is later joined against trusted game_pk integers, so there
is no injection surface against the warehouse.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd  # noqa: F401 -- used by route handlers
from fastapi import APIRouter, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from mlb_model.app import services
from mlb_model.logging import get_logger

log = get_logger("app.routes")

APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=APP_DIR / "templates")

router = APIRouter()


# ---------------------------------------------------------------------------
# Jinja helpers
# ---------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x * 100:.{digits}f}%"


def _odds(prob: float | None) -> str:
    """Convert a probability to American-style odds."""
    if prob is None or prob <= 0 or prob >= 1 or (isinstance(prob, float) and prob != prob):
        return "—"
    if prob >= 0.5:
        return f"-{round(prob / (1 - prob) * 100)}"
    return f"+{round((1 - prob) / prob * 100)}"


def _num(x: float | None, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:.{digits}f}"


def _signed_pp(x: float | None) -> str:
    if x is None or (isinstance(x, float) and (x != x)):
        return "—"
    return f"{x:+.1f}pp"


templates.env.filters["pct"] = _pct
templates.env.filters["odds"] = _odds
templates.env.filters["num"] = _num
templates.env.filters["signed_pp"] = _signed_pp


# ---------------------------------------------------------------------------
# Date helper
# ---------------------------------------------------------------------------


def _parse_date(value: str | None) -> date_cls:
    if not value:
        return datetime.now().date()
    try:
        return date_cls.fromisoformat(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Bad date: {value!r}") from exc


def _automation_status() -> dict[str, str]:
    """Return human-readable last-run dates + a "stale?" flag for the
    two scheduled jobs.

    The footer uses this to color the morning-sync date amber/red when
    we haven't synced in a day or more, instead of blindly painting it
    green just because *some* date was recorded once.
    """
    from datetime import date as _date

    from mlb_model.automation import morning_sync, weekly_train

    def _fmt(d: Any) -> str:
        if d is None:
            return "never"
        return d.isoformat()

    last_sync = morning_sync.last_run_date()
    today = _date.today()
    if last_sync is None:
        sync_state = "never"
    elif last_sync >= today:
        sync_state = "ok"
    elif (today - last_sync).days <= 1:
        sync_state = "yesterday"
    else:
        sync_state = "stale"

    return {
        "morning_sync": _fmt(last_sync),
        "morning_sync_state": sync_state,
        "weekly_train": _fmt(weekly_train.last_run_date()),
    }


def _render(request: Request, template: str, context: dict[str, Any]) -> HTMLResponse:
    """Thin wrapper around Starlette's TemplateResponse.

    The Starlette/Jinja2Templates API requires ``request`` as the first
    positional argument (the older "context-dict carries the request"
    pattern was deprecated). Going through this helper keeps the call
    sites consistent, attaches automation status to every page, and
    prevents the easy mistake of passing a dict as
    the template name (which produces a cryptic "unhashable type: dict"
    error from Jinja2's cache).
    """
    full_context = {**context, "automation_status": _automation_status()}
    return templates.TemplateResponse(request, template, full_context)


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    date: str | None = Query(default=None),
    market: str = Query(default="moneyline"),
    tier: str = Query(default="all"),
) -> HTMLResponse:
    """Today's-picks dashboard."""
    target = _parse_date(date)
    try:
        df = services.get_predictions(target, refresh=False)
    except Exception as exc:  # noqa: BLE001 -- want a clean error page
        log.exception("dashboard.predict_failed")
        return _render(
            request,
            "dashboard.html",
            {
                "target_date": target,
                "picks": [],
                "active_market": market,
                "active_tier": tier,
                "error": str(exc),
                "yesterday": target - timedelta(days=1),
                "tomorrow": target + timedelta(days=1),
                "has_cache": False,
            },
        )

    picks = services.shape_picks(df)
    picks_filtered = _filter_picks(picks, market=market, tier=tier)

    return _render(
        request,
        "dashboard.html",
        {
            "target_date": target,
            "picks": picks_filtered,
            "total_picks": len(picks),
            "active_market": market,
            "active_tier": tier,
            "yesterday": target - timedelta(days=1),
            "tomorrow": target + timedelta(days=1),
            "has_cache": services.load_cached_predictions(target) is not None,
            "error": None,
        },
    )


@router.get("/game/{game_pk}", response_class=HTMLResponse)
def game_detail(
    request: Request,
    game_pk: int,
    date: str | None = Query(default=None),
) -> HTMLResponse:
    target = _parse_date(date)
    detail = services.get_game_detail(target, game_pk)
    if detail is None:
        raise HTTPException(status_code=404, detail="Game not found in today's predictions.")
    return _render(request, "game_detail.html", {"g": detail, "target_date": target})


@router.get("/performance", response_class=HTMLResponse)
def performance(request: Request) -> HTMLResponse:
    df = services.load_backtest()
    season_rows = df.to_dict(orient="records") if not df.empty else []

    summary_2026 = services.load_2026_summary()

    # The backtest CSV is wide -- one row per season with ml_/rl_/ou_
    # columns inline. Headline is a games-weighted average across rows.
    headline: dict[str, Any] = {}
    if not df.empty and "n_games" in df.columns:
        w = df["n_games"].astype(float)
        total = float(w.sum())
        if total > 0:
            for key, label_col in [
                ("moneyline", "ml_accuracy"),
                ("runline", "rl_accuracy"),
                ("total", "ou_accuracy"),
            ]:
                if label_col not in df.columns:
                    continue
                vals = pd.to_numeric(df[label_col], errors="coerce")
                mask = vals.notna()
                if not mask.any():
                    continue
                acc = float((vals[mask] * w[mask]).sum() / float(w[mask].sum()))
                headline[key] = {
                    "accuracy": acc,
                    "sample_size": int(w[mask].sum()),
                }

    return _render(
        request,
        "performance.html",
        {
            "rows": season_rows,
            "headline": headline,
            "season_2026": summary_2026,
        },
    )


@router.get("/season", response_class=HTMLResponse)
def season_view(
    request: Request,
    season: int | None = Query(default=None),
) -> HTMLResponse:
    """Model performance for an entire season -- every prediction it made,
    graded against actual outcomes, broken down by market and tier."""
    data = services.season_performance(season=season)
    return _render(
        request,
        "season.html",
        {
            "season": data["season"],
            "available_seasons": data["available_seasons"],
            "summary": data["summary"],
            "calibration": data["calibration"],
            "rolling": data["rolling"],
            "tier_breakdown": data["tier_breakdown"],
            "recent_results": data["recent_results"],
            "journal_size": data["journal_size"],
            "eos_report": data.get("eos_report"),
        },
    )


@router.get("/api/season/report")
def season_report(season: int = Query(...)) -> Response:
    """Serve the end-of-season markdown report as plain text.

    The /season page shows a link to this endpoint when a report
    exists; clicking it opens the report in a new tab. The file lives
    on disk at ``reports/end_of_season_<season>/report.md``.
    """
    from mlb_model.season.end_of_season import REPORTS_ROOT

    path = REPORTS_ROOT / f"end_of_season_{int(season)}" / "report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No end-of-season report on disk for that year.")
    body = path.read_text(encoding="utf-8")
    return Response(
        content=body,
        media_type="text/markdown; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/log", response_class=HTMLResponse)
def picks_log_view(request: Request) -> HTMLResponse:
    df = services.load_picks_log()
    if not df.empty:
        df = df.sort_values("logged_at", ascending=False)

    summary: dict[str, Any] = {}
    if not df.empty:
        graded = df[df["result"].isin(["win", "loss", "push"])]
        if not graded.empty:
            summary["graded"] = int(len(graded))
            summary["wins"] = int((graded["result"] == "win").sum())
            summary["losses"] = int((graded["result"] == "loss").sum())
            summary["pushes"] = int((graded["result"] == "push").sum())
            win_rate_base = graded["result"].isin(["win", "loss"]).sum()
            summary["win_rate"] = (
                float(summary["wins"]) / float(win_rate_base) if win_rate_base else None
            )
            summary["units"] = float(graded["roi_units"].fillna(0.0).sum())
        else:
            summary["graded"] = 0
        summary["pending"] = int((df["result"] == "pending").sum())
        summary["total"] = int(len(df))

    return _render(
        request,
        "log.html",
        {
            "rows": df.to_dict(orient="records") if not df.empty else [],
            "summary": summary,
        },
    )


# ---------------------------------------------------------------------------
# HTMX partials & JSON
# ---------------------------------------------------------------------------


@router.get("/api/picks", response_class=HTMLResponse)
def api_picks(
    request: Request,
    date: str | None = Query(default=None),
    market: str = Query(default="moneyline"),
    tier: str = Query(default="all"),
) -> HTMLResponse:
    """HTMX partial: just the picks table body (re-render on filter change)."""
    target = _parse_date(date)
    df = services.get_predictions(target, refresh=False)
    picks = services.shape_picks(df)
    picks = _filter_picks(picks, market=market, tier=tier)
    return _render(
        request,
        "partials/picks_table.html",
        {"picks": picks, "target_date": target, "active_market": market},
    )


@router.post("/api/refresh", response_class=HTMLResponse)
def api_refresh(
    request: Request,
    date: str | None = Form(default=None),
    market: str = Form(default="moneyline"),
    tier: str = Form(default="all"),
) -> HTMLResponse:
    """HTMX: refresh schedule + weather + odds, re-run model, return new table.

    Any exception is converted into an in-line error banner (HTTP 200) so
    HTMX can swap the partial cleanly. The dashboard route already does
    this -- this endpoint used to 500 on the same failure modes, which
    HTMX renders as a broken swap. Now the two routes degrade identically.
    """
    target = _parse_date(date)
    log.info("ui.refresh.start", target=target)
    try:
        df = services.compute_predictions(target, refresh=True)
        picks = services.shape_picks(df)
        picks = _filter_picks(picks, market=market, tier=tier)
        return _render(
            request,
            "partials/picks_table.html",
            {"picks": picks, "target_date": target, "active_market": market},
        )
    except Exception as exc:  # noqa: BLE001 -- surface every failure to the user
        log.exception("ui.refresh.failed", target=target)
        return _render(
            request,
            "partials/picks_table.html",
            {
                "picks": [],
                "target_date": target,
                "active_market": market,
                "error": (
                    f"Refresh failed: {type(exc).__name__}: {exc}. "
                    "Check logs/ for details."
                ),
            },
        )


@router.get("/api/game/{game_pk}/distribution")
def api_score_distribution(game_pk: int, date: str | None = None) -> JSONResponse:
    target = _parse_date(date)
    detail = services.get_game_detail(target, game_pk)
    if detail is None:
        raise HTTPException(status_code=404)
    return JSONResponse(detail.score_distribution)


@router.get("/api/performance/seasons")
def api_perf_seasons() -> JSONResponse:
    df = services.load_backtest()
    if df.empty:
        return JSONResponse({"rows": []})
    return JSONResponse({"rows": df.to_dict(orient="records")})


@router.post("/api/picks-log/add")
def api_log_pick(payload: dict[str, Any]) -> JSONResponse:
    """Persist one pick from the dashboard."""
    required = {"game_pk", "market", "pick", "pick_long", "model_prob"}
    missing = required - set(payload.keys())
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing fields: {sorted(missing)}")
    try:
        payload = {
            "game_pk": int(payload["game_pk"]),
            "game_date": str(payload.get("game_date") or ""),
            "away_team_abbr": str(payload.get("away_team_abbr") or ""),
            "home_team_abbr": str(payload.get("home_team_abbr") or ""),
            "market": str(payload["market"]),
            "pick": str(payload["pick"]),
            "pick_long": str(payload["pick_long"]),
            "model_prob": float(payload["model_prob"]),
            "market_prob": (
                float(payload["market_prob"])
                if payload.get("market_prob") not in (None, "")
                else None
            ),
            "edge_pp": (
                float(payload["edge_pp"])
                if payload.get("edge_pp") not in (None, "")
                else None
            ),
            "total_line": (
                float(payload["total_line"])
                if payload.get("total_line") not in (None, "")
                else None
            ),
            "tier": str(payload.get("tier") or ""),
            "stake_units": (
                float(payload["stake_units"])
                if payload.get("stake_units") not in (None, "")
                else 1.0
            ),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"Bad payload: {exc}") from exc
    pick_id = services.append_logged_pick(payload)
    return JSONResponse({"ok": True, "pick_id": pick_id})


@router.post("/api/picks-log/delete")
def api_delete_pick(payload: dict[str, Any]) -> JSONResponse:
    """Delete a single logged pick by ``pick_id``."""
    pick_id = str(payload.get("pick_id") or "").strip()
    if not pick_id:
        raise HTTPException(status_code=422, detail="Missing pick_id")
    ok = services.delete_logged_pick(pick_id)
    if not ok:
        raise HTTPException(status_code=404, detail="pick_id not found")
    return JSONResponse({"ok": True})


@router.post("/api/picks-log/update")
def api_update_pick(payload: dict[str, Any]) -> JSONResponse:
    """Edit a logged pick (currently only ``stake_units``)."""
    pick_id = str(payload.get("pick_id") or "").strip()
    if not pick_id:
        raise HTTPException(status_code=422, detail="Missing pick_id")
    stake: float | None = None
    if payload.get("stake_units") not in (None, ""):
        try:
            stake = float(payload["stake_units"])
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Bad stake_units: {exc}") from exc
        if stake < 0:
            raise HTTPException(status_code=422, detail="stake_units must be >= 0")
    ok = services.update_logged_pick(pick_id, stake_units=stake)
    if not ok:
        raise HTTPException(status_code=404, detail="pick_id not found")
    return JSONResponse({"ok": True})


@router.post("/api/picks-log/clear")
def api_clear_picks() -> JSONResponse:
    """Delete every logged pick."""
    n = services.clear_logged_picks()
    return JSONResponse({"ok": True, "removed": n})


@router.get("/api/totals/probability")
def api_total_prob(
    game_pk: int,
    line: float,
    date: str | None = None,
) -> JSONResponse:
    """Compute P(total runs > line) for an arbitrary line.

    Reads the cached prediction for the day, pulls (home_mean, home_std,
    away_mean, away_std), and re-simulates against the user-provided line.
    """
    target = _parse_date(date)
    df = services.get_predictions(target)
    if df.empty:
        raise HTTPException(status_code=404, detail="No predictions for that date.")
    row = df[df["game_pk"] == game_pk]
    if row.empty:
        raise HTTPException(status_code=404, detail="game_pk not found in predictions.")
    r = row.iloc[0]
    home_mean = float(r["pred_home_runs"])
    away_mean = float(r["pred_away_runs"])
    home_std = float(r["runs_std_home"]) if pd.notna(r.get("runs_std_home")) else (home_mean**0.5) * 1.3
    away_std = float(r["runs_std_away"]) if pd.notna(r.get("runs_std_away")) else (away_mean**0.5) * 1.3
    p_over = services.p_total_over_at_line(home_mean, home_std, away_mean, away_std, float(line))
    return JSONResponse({
        "line": float(line),
        "p_over": p_over,
        "p_under": 1.0 - p_over,
        "pick": "OVER" if p_over >= 0.5 else "UNDER",
        "predicted_total": home_mean + away_mean,
    })


# ---------------------------------------------------------------------------
# CSV exports -- any table on screen can be downloaded as CSV with one click.
# All endpoints are local-only (the app binds to 127.0.0.1); no auth needed.
# ---------------------------------------------------------------------------


def _csv_response(df: pd.DataFrame, filename: str) -> Response:
    """Serialize ``df`` as a CSV download attachment."""
    body = df.to_csv(index=False).encode("utf-8")
    return Response(
        content=body,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": "no-store",
        },
    )


@router.get("/api/export/picks")
def export_picks(date: str | None = Query(default=None)) -> Response:
    """Today's slate (every market, every game) as CSV."""
    target = _parse_date(date)
    df = services.compute_predictions(target, refresh=False)
    return _csv_response(df, f"picks_{target.isoformat()}.csv")


@router.get("/api/export/log")
def export_log() -> Response:
    """Your picks log (with grading + ROI) as CSV."""
    df = services.load_picks_log()
    return _csv_response(df, "picks_log.csv")


@router.get("/api/export/journal")
def export_journal(season: int | None = Query(default=None)) -> Response:
    """The model's prediction journal (all predictions ever made, graded)."""
    from mlb_model.journal import grade_journal

    df = grade_journal(season=season)
    name = f"journal_{season}.csv" if season else "journal_all.csv"
    return _csv_response(df, name)


@router.get("/api/export/backtest")
def export_backtest() -> Response:
    """The most recent backtest results (season-by-season)."""
    df = services.load_backtest()
    return _csv_response(df, "backtest_results.csv")


@router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/favicon.ico")
def favicon() -> RedirectResponse:
    return RedirectResponse(url="/static/favicon.svg")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


_TIER_THRESHOLDS = {
    "all": 0.0,
    "lean": 0.15,
    "edge": 0.30,
    "strong": 0.50,
    "premium": 0.70,
}


def _filter_picks(
    picks: list[services.PickRow],
    *,
    market: str,
    tier: str,
) -> list[services.PickRow]:
    market = market.lower()
    tier = tier.lower()

    out = [p for p in picks if p.market == market]
    threshold = _TIER_THRESHOLDS.get(tier, 0.0)
    out = [p for p in out if p.confidence >= threshold]
    out.sort(key=lambda p: p.confidence, reverse=True)
    return out
