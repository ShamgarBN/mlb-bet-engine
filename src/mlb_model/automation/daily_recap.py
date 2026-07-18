"""Morning scorecard: yesterday's alerted picks, graded, to Discord.

Runs before the 11 AM picks alert (default 8:00 local via LaunchAgent).
Since morning-sync hasn't ingested yesterday's finals yet at that hour,
this job refreshes them itself, settles the props journal, then posts a
compact per-tier win-rate summary:

* Hitter props  -- journal rows at Premium/Strong for hit/HR/TB (exactly
  the set the two daily alerts select from), graded by ``grade_props``.
* Game picks    -- predictions-journal rows at Premium/Strong, minus
  baseline-total picks (the alert never sends those).
* Pitcher K's   -- the alerted high-K spots persisted by the morning
  alert, scored as hit when actual K's reach the rounded estimate.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from typing import Any

import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("automation.daily_recap")

_TIER_TAG = {"premium": "🟢 Premium", "strong": "🟡 Strong"}
_PROP_MARKETS = ("prop_hit", "prop_hr", "prop_tb")


def _pct(wins: int, losses: int) -> str:
    n = wins + losses
    return f"{wins / n * 100:.0f}%" if n else "—"


def prop_summary(target: date_cls) -> dict[str, dict[str, int]]:
    """Per-tier W-L for yesterday's Premium/Strong hit/HR/TB props."""
    from mlb_model.journal.props import _load

    df = _load()
    if df.empty:
        return {}
    day = df[
        (df["game_date"].astype(str).str[:10] == target.isoformat())
        & df["market"].isin(_PROP_MARKETS)
        & df["tier"].isin(("premium", "strong"))
        & df["is_settled"].fillna(False)
    ]
    out: dict[str, dict[str, int]] = {}
    for tier, grp in day.groupby("tier"):
        wins = int(grp["y_win"].fillna(0).sum())
        out[str(tier)] = {"wins": wins, "losses": int(len(grp)) - wins}
    return out


def game_summary(target: date_cls) -> dict[str, dict[str, int]]:
    """Per-tier W-L for yesterday's alerted (Premium/Strong) game picks."""
    from mlb_model.journal.metrics import grade_journal

    g = grade_journal(season=target.year)
    if g.empty:
        return {}
    day = g[
        (g["game_date"].astype(str) == target.isoformat())
        & g["tier"].isin(("premium", "strong"))
    ]
    # The alert drops O/U picks graded against a baseline (non-market) line.
    day = day[
        ~((day["market"] == "total") & (day["total_line_source"] != "market"))
    ]
    day = day[day["result"].isin(("win", "loss"))]
    out: dict[str, dict[str, int]] = {}
    for tier, grp in day.groupby("tier"):
        wins = int((grp["result"] == "win").sum())
        out[str(tier)] = {"wins": wins, "losses": int(len(grp)) - wins}
    return out


def pitcher_summary(target: date_cls) -> list[dict[str, Any]]:
    """Alerted K spots with actuals: hit when actual >= rounded estimate."""
    from mlb_model.automation.alerts import _pitcher_k_log_path
    from mlb_model.data.warehouse import query

    path = _pitcher_k_log_path()
    if not path.exists():
        return []
    logged = pd.read_parquet(path)
    day = logged[logged["game_date"].astype(str) == target.isoformat()]
    day = day[day["pitcher_id"].notna()]
    if day.empty:
        return []
    pks = [int(p) for p in day["pitcher_id"]]
    placeholders = ",".join("?" for _ in pks)
    actuals = query(
        f"""
        SELECT pgs.pitcher_id, MAX(pgs.strikeouts) AS strikeouts
        FROM pitcher_game_stats pgs
        JOIN games g USING (game_pk)
        WHERE g.game_date = ? AND g.status = 'Final'
          AND pgs.pitcher_id IN ({placeholders})
        GROUP BY pgs.pitcher_id
        """,
        (target, *pks),
    )
    k_map = {int(r.pitcher_id): int(r.strikeouts) for r in actuals.itertuples()}
    out = []
    for r in day.itertuples():
        actual = k_map.get(int(r.pitcher_id))
        est = float(r.est_k)
        out.append({
            "pitcher": r.pitcher,
            "team": r.team,
            "est_k": est,
            "actual_k": actual,
            "hit": (actual is not None and actual >= round(est)),
        })
    return out


def build_recap_message(
    target: date_cls,
    props: dict[str, dict[str, int]],
    games: dict[str, dict[str, int]],
    pitchers: list[dict[str, Any]],
) -> str | None:
    """Compact markdown scorecard, or None when nothing settled."""
    if not props and not games and not pitchers:
        return None
    when = target.strftime("%a %b %-d")
    lines = [f"**📊 MLB Forecast — yesterday's scorecard · {when}**"]

    lines.append("\n__Hitter props__")
    if props:
        for tier in ("premium", "strong"):
            if tier in props:
                w, l = props[tier]["wins"], props[tier]["losses"]
                lines.append(f"{_TIER_TAG[tier]}: {w}-{l} ({_pct(w, l)})")
    else:
        lines.append("_none settled_")

    lines.append("\n__Game picks__")
    if games:
        for tier in ("premium", "strong"):
            if tier in games:
                w, l = games[tier]["wins"], games[tier]["losses"]
                lines.append(f"{_TIER_TAG[tier]}: {w}-{l} ({_pct(w, l)})")
    else:
        lines.append("_none alerted_")

    if pitchers:
        graded = [p for p in pitchers if p["actual_k"] is not None]
        hits = sum(1 for p in graded if p["hit"])
        lines.append(f"\n__Pitcher K spots__ ({hits}/{len(graded)} reached estimate)")
        for p in pitchers:
            actual = "—" if p["actual_k"] is None else str(p["actual_k"])
            mark = " ✓" if p["hit"] else ""
            lines.append(f"• {p['pitcher']} ({p['team']}): est {p['est_k']:g} → {actual}{mark}")

    return "\n".join(lines)


def run_daily_recap(
    today: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Grade yesterday and post the scorecard. Deduped once per day."""
    from mlb_model.automation.alerts import _marker_path, post_to_discord

    today = today or date_cls.today()
    target = today - timedelta(days=1)
    marker = _marker_path(today, kind="recap")
    if not dry_run and not force and marker.exists():
        return {"sent": False, "reason": "already-sent-today"}

    # Finals for yesterday usually aren't in the warehouse yet at 8 AM
    # (morning-sync runs at 11) -- pull them now, then settle props.
    try:
        from mlb_model.automation.morning_sync import _refresh_finals_for_date

        _refresh_finals_for_date(target)
    except Exception:  # noqa: BLE001 -- grade whatever is already settled
        log.exception("daily_recap.finals_refresh_failed")
    try:
        from mlb_model.journal.props import grade_props

        grade_props()
    except Exception:  # noqa: BLE001
        log.exception("daily_recap.grade_props_failed")

    props = prop_summary(target)
    games = game_summary(target)
    pitchers = pitcher_summary(target)
    message = build_recap_message(target, props, games, pitchers)
    result: dict[str, Any] = {
        "target": target.isoformat(),
        "props": props, "games": games, "n_pitchers": len(pitchers),
        "message": message,
    }
    if message is None:
        result.update(sent=False, reason="nothing-to-recap")
        return result
    if dry_run:
        result.update(sent=False, reason="dry-run")
        return result

    sent = post_to_discord(message)
    result["sent"] = sent
    if sent:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("sent")
        log.info("daily_recap.sent", target=target.isoformat())
    else:
        result["reason"] = "post-failed-or-no-webhook"
    return result
