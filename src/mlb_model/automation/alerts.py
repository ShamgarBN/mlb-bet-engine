"""Daily high-confidence pick alerts via a Discord webhook.

Gathers the day's **Premium / Strong** picks — game markets (ML / RL / O-U,
confidence ≥ 0.50) and hitter props (hit / HR / TB / K, edge ≥ 5 pp above the
league baseline) — formats them, and POSTs to a Discord webhook. Wired into
``morning_sync`` so it fires once per day after the slate is refreshed.

Best-effort and deduped per day: no webhook configured → no-op; a network
failure is logged, never fatal. Stdlib-only (urllib), no bot to host.
"""

from __future__ import annotations

import json
import urllib.request
from datetime import date as date_cls
from typing import Any

from mlb_model.config import settings
from mlb_model.journal.props import LEAGUE_BASELINES, edge_to_tier
from mlb_model.logging import get_logger

log = get_logger("automation.alerts")

ALERT_TIERS = ("premium", "strong")
_DISCORD_LIMIT = 1900  # leave headroom under Discord's 2000-char message cap

# Strikeouts are handled per-pitcher (not per-batter): a high-K starter would
# otherwise make every opposing hitter a near-identical "1+ K" pick. We project
# the starter's K total as k_pct × expected batters faced.
_PROP_MARKETS = ("prop_hit", "prop_hr", "prop_tb")  # graded picks (K handled separately)
EXPECTED_BATTERS_FACED = 22  # ~5.2 IP starter
# A starter is a "high-K spot" when his strikeout rate clears this (league
# starter average is ~0.22). Based on the probable pitcher alone, so it works
# at 11 AM before lineups are posted.
HIGH_K_PCT = 0.25

_MARKET_LABEL = {"moneyline": "ML", "runline": "RL", "total": "O/U"}
_PROP_LABEL = {"prop_hit": "1+ Hit", "prop_hr": "HR", "prop_tb": "2+ TB"}


# --------------------------------------------------------------------------- #
# Pick selection
# --------------------------------------------------------------------------- #
def select_game_picks(target: date_cls) -> list[dict[str, Any]]:
    """Premium/Strong game-market picks for ``target`` (cached predictions)."""
    from mlb_model.app import services
    from mlb_model.predict.daily import predict_for_date

    try:
        preds = predict_for_date(target, refresh_data=False)
    except Exception:  # noqa: BLE001
        log.warning("alerts.game.predict_failed")
        return []
    if preds is None or preds.empty:
        return []
    picks = [
        p for p in services.shape_picks(preds)
        if p.tier in ALERT_TIERS
        # Drop O/U picks with no real market total — a "UNDER 9*" against the
        # league-average baseline isn't a bettable line (no book posted it yet).
        and not (p.market == "total" and p.total_line_source != "market")
    ]
    picks.sort(key=lambda p: (p.tier != "premium", -(p.confidence or 0)))
    return [
        {
            "tier": p.tier,
            "matchup": f"{p.away_team_abbr} @ {p.home_team_abbr}",
            "market": _MARKET_LABEL.get(p.market, p.market),
            "pick": p.pick_long or p.pick,
            "confidence": p.confidence,
            "edge_pp": p.edge_pp,
        }
        for p in picks
    ]


def todays_matchups(
    target: date_cls, *, record: bool = True, refresh: bool = False,
) -> list[dict[str, Any]]:
    """Scored hitter-prop matchups for ``target`` (records them for grading).

    ``refresh`` busts the per-date cache so lineups posted since the last
    run are picked up (the afternoon pass needs this).
    """
    from mlb_model.scoring.service import get_matchups_for_date

    try:
        matchups = get_matchups_for_date(target, refresh=refresh)
    except Exception:  # noqa: BLE001
        log.warning("alerts.prop.matchups_failed")
        return []
    if not matchups:
        return []
    if record:
        try:
            from mlb_model.journal.props import record_matchups

            record_matchups(matchups)
        except Exception:  # noqa: BLE001 -- journaling must never block the alert
            log.warning("alerts.prop.record_failed")
    return matchups


def select_prop_picks(matchups: list[dict]) -> list[dict[str, Any]]:
    """Premium/Strong hit / HR / TB picks (strikeouts handled per-pitcher)."""
    out: list[dict[str, Any]] = []
    for game in matchups:
        for side_key in ("away", "home"):
            side = game.get(side_key) or {}
            opp = game.get("home" if side_key == "away" else "away") or {}
            opp_sp = (opp.get("starter") or {})
            for b in (side.get("batters") or []):
                for market in _PROP_MARKETS:
                    prob = b.get(market.replace("prop_", "") + "_prob")
                    if prob is None:
                        continue
                    edge_pp = (float(prob) - LEAGUE_BASELINES[market]) * 100.0
                    tier = edge_to_tier(edge_pp)
                    if tier not in ALERT_TIERS:
                        continue
                    out.append({
                        "tier": tier,
                        "player": b.get("name") or "",
                        "team": side.get("team_abbr") or "",
                        "market": _PROP_LABEL[market],
                        "model_prob": float(prob),
                        "edge_pp": edge_pp,
                        "opp_sp": opp_sp.get("name") or "",
                        "opp_throws": opp_sp.get("throws") or "",
                    })
    out.sort(key=lambda p: (p["tier"] != "premium", -p["edge_pp"]))
    return out


def select_pitcher_strikeouts(matchups: list[dict]) -> list[dict[str, Any]]:
    """One projected-K line per high-K starter (k_pct ≥ ``HIGH_K_PCT``).

    Depends only on the probable pitcher (not the lineup), so it's available at
    11 AM before lineups post. Projected K's = k_pct × expected batters faced.
    """
    out: list[dict[str, Any]] = []
    for game in matchups:
        for sp_side in ("away", "home"):
            side = game.get(sp_side) or {}
            opp = game.get("home" if sp_side == "away" else "away") or {}
            sp = side.get("starter") or {}
            name, kpct = sp.get("name"), sp.get("k_pct")
            if not name or kpct is None or float(kpct) < HIGH_K_PCT:
                continue
            out.append({
                "pitcher": name,
                "pitcher_id": sp.get("id"),
                "throws": sp.get("throws") or "",
                "team": side.get("team_abbr") or "",
                "vs_team": opp.get("team_abbr") or "",
                "est_k": round(float(kpct) * EXPECTED_BATTERS_FACED, 1),
            })
    out.sort(key=lambda x: -x["est_k"])
    return out


# --------------------------------------------------------------------------- #
# Message formatting
# --------------------------------------------------------------------------- #
_TIER_EMOJI = {"premium": "🟢", "strong": "🟡"}
# Compact market tags for the grouped prop lines ("1+ Hit" reads fine on
# its own line but wastes width inline).
_MARKET_SHORT = {"1+ Hit": "H", "HR": "HR", "2+ TB": "TB"}


def _surname(name: str) -> str:
    """'Michael Harris II' -> 'Harris II'; single tokens pass through."""
    parts = (name or "").split()
    return " ".join(parts[1:]) if len(parts) > 1 else (name or "")


def _prop_lines(prop_picks: list[dict], *, max_prop: int) -> list[str]:
    """Dense prop section: one line per team-vs-pitcher group.

    ``CWS vs Bieber (R): 🟢 Vargas TB 51% · 🟡 Antonacci H 71%``

    Groups are ordered premium-first then by best edge; the pick budget
    (``max_prop``) fills group by group with a trailing "+N more" line.
    """
    groups: dict[tuple, list[dict]] = {}
    for p in prop_picks:
        key = (p.get("team") or "", p.get("opp_sp") or "", p.get("opp_throws") or "")
        groups.setdefault(key, []).append(p)
    ordered = sorted(
        groups.values(),
        key=lambda ps: (
            all(x["tier"] != "premium" for x in ps),
            -max(x["edge_pp"] for x in ps),
        ),
    )
    # Cap picks per group too, or one hot matchup (a bad pitcher at Coors)
    # eats the whole budget and only 2-3 games make the message. An
    # uncapped render (`alert run --full`) shows every pick.
    group_cap = min(6, max_prop) if max_prop < 10**8 else max_prop
    lines: list[str] = []
    budget = max_prop
    shown = 0
    for ps in ordered:
        if budget <= 0:
            break
        ranked = sorted(ps, key=lambda x: (x["tier"] != "premium", -x["edge_pp"]))
        take = ranked[: min(group_cap, budget)]
        budget -= len(take)
        shown += len(take)
        head = take[0]
        sp = _surname(head.get("opp_sp") or "")
        vs = f" vs {sp} ({head.get('opp_throws') or '?'})" if sp else ""
        picks = " · ".join(
            f"{_TIER_EMOJI.get(x['tier'], '•')} {_surname(x['player'])} "
            f"{_MARKET_SHORT.get(x['market'], x['market'])} {x['model_prob'] * 100:.0f}%"
            for x in take
        )
        extra = f" (+{len(ps) - len(take)})" if len(ps) > len(take) else ""
        lines.append(f"{head['team']}{vs}: {picks}{extra}")
    rest = len(prop_picks) - shown
    if rest > 0:
        lines.append(f"_…and {rest} more (see the app)_")
    return lines


def build_message(
    target: date_cls,
    game_picks: list[dict],
    prop_picks: list[dict],
    pitcher_ks: list[dict] | None = None,
    *,
    uncapped: bool = False,
) -> str | None:
    """Markdown message body, or None if there's nothing to alert on.

    ``uncapped`` disables the per-section caps + "…and N more" truncation so
    the full pick list can be printed (e.g. `alert run --full`) when the Discord
    message was capped.
    """
    pitcher_ks = pitcher_ks or []
    if not game_picks and not prop_picks and not pitcher_ks:
        return None
    big = 10**9
    max_game = big if uncapped else settings.alert_max_game_picks
    max_prop = big if uncapped else settings.alert_max_prop_picks
    when = target.strftime("%a %b %-d")
    lines = [f"**⚾ MLB Forecast — high-confidence picks · {when}**"]

    if game_picks:
        shown = game_picks[:max_game]
        lines.append(f"\n__Game markets__ ({len(game_picks)})")
        for p in shown:
            conf = f" {p['confidence'] * 100:.0f}%" if p.get("confidence") is not None else ""
            edge = f" ({p['edge_pp']:+.0f}pp)" if p.get("edge_pp") is not None else ""
            lines.append(
                f"{_TIER_EMOJI.get(p['tier'], '•')} {p['matchup']} "
                f"{p['market']}: {p['pick']}{conf}{edge}"
            )
        if len(game_picks) > len(shown):
            lines.append(f"_…and {len(game_picks) - len(shown)} more (see the app)_")

    if prop_picks:
        lines.append(f"\n__Hitter props__ ({len(prop_picks)})")
        lines.extend(_prop_lines(prop_picks, max_prop=max_prop))

    if pitcher_ks:
        ks = pitcher_ks[: (big if uncapped else 12)]
        inline = " · ".join(
            f"{_surname(k['pitcher'])} ({k['team']}) {k['est_k']:g}" for k in ks
        )
        lines.append(f"\n__Pitcher Ks (est)__\n{inline}")

    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


def _chunk(content: str, limit: int = _DISCORD_LIMIT) -> list[str]:
    """Split on line boundaries so each chunk fits Discord's message cap."""
    chunks: list[str] = []
    cur = ""
    for line in content.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            chunks.append(cur)
            cur = ""
        cur += (("\n" if cur else "") + line)
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #
def post_to_discord(content: str, *, webhook_url: str | None = None) -> bool:
    """POST a message to the Discord webhook. Returns True on success."""
    url = webhook_url or settings.discord_webhook_url
    if not url:
        log.info("alerts.discord.no_webhook")
        return False
    ok = True
    for chunk in _chunk(content):
        payload = json.dumps({"content": chunk, "username": "MLB Forecast"}).encode()
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json", "User-Agent": settings.http_user_agent},
        )
        try:
            with urllib.request.urlopen(req, timeout=settings.http_timeout_seconds) as resp:
                if resp.status >= 300:
                    ok = False
        except Exception:  # noqa: BLE001
            log.warning("alerts.discord.post_failed")
            ok = False
    return ok


def _pitcher_k_log_path():
    return settings.data_dir / "journal" / "pitcher_k_alerts.parquet"


def _log_pitcher_ks(target: date_cls, pitcher_ks: list[dict]) -> None:
    """Persist the alerted pitcher-K spots so the recap can grade them.

    Best-effort append, deduped by (date, pitcher_id). Without this the
    K section exists only in the Discord message and can't be scored.
    """
    if not pitcher_ks:
        return
    try:
        import pandas as pd

        rows = pd.DataFrame([
            {
                "game_date": target.isoformat(),
                "pitcher_id": k.get("pitcher_id"),
                "pitcher": k.get("pitcher"),
                "team": k.get("team"),
                "vs_team": k.get("vs_team"),
                "est_k": k.get("est_k"),
            }
            for k in pitcher_ks
        ])
        path = _pitcher_k_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["game_date", "pitcher_id"], keep="first")
        rows.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 -- logging picks must never block the alert
        log.warning("alerts.pitcher_k_log.failed")


def _marker_path(target: date_cls, kind: str = "daily"):
    if kind == "daily":
        return settings.cache_dir / f"alert_sent_{target.isoformat()}.flag"
    return settings.cache_dir / f"alert_{kind}_sent_{target.isoformat()}.flag"


def _not_started(scheduled_start, *, now=None) -> bool:
    """True when the game's UTC start time is still in the future.

    Unknown/unparseable starts return True (better to repeat a pick than
    silently drop a live game).
    """
    from datetime import UTC, datetime

    import pandas as pd

    ts = pd.to_datetime(scheduled_start, errors="coerce")
    if ts is pd.NaT:
        return True
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    now = now or datetime.now(UTC)
    return ts > now


def _alerted_props_log_path():
    return settings.data_dir / "journal" / "alerted_props.parquet"


def _log_alerted_props(target: date_cls, prop_picks: list[dict]) -> None:
    """Persist which (player, market) props an alert selected for ``target``.

    This — not the props journal — is the afternoon pass's dedup source.
    The scoring service journals EVERY matchup it scores (page loads, dry
    runs, cache refreshes included), so journal presence can't distinguish
    "alerted this morning" from "merely scored at some point".
    """
    if not prop_picks:
        return
    try:
        import pandas as pd

        rows = pd.DataFrame([
            {
                "game_date": target.isoformat(),
                "player": p.get("player"),
                "market": p.get("market"),
                "tier": p.get("tier"),
            }
            for p in prop_picks
        ])
        path = _alerted_props_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["game_date", "player", "market"], keep="first")
        rows.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001 -- logging picks must never block the alert
        log.warning("alerts.alerted_props_log.failed")


def _alerted_prop_keys(target: date_cls) -> set[tuple[str, str]]:
    """(player, market) pairs already alerted for ``target``."""
    try:
        import pandas as pd

        path = _alerted_props_log_path()
        if not path.exists():
            return set()
        df = pd.read_parquet(path)
        day = df[df["game_date"].astype(str) == target.isoformat()]
        return set(zip(day["player"].astype(str), day["market"].astype(str)))
    except Exception:  # noqa: BLE001 -- dedup is best-effort; worst case we repeat picks
        log.warning("alerts.alerted_props_log.read_failed")
        return set()


def build_afternoon_message(
    target: date_cls, prop_picks: list[dict], *, uncapped: bool = False,
) -> str | None:
    """Markdown for the afternoon lineup-props alert, or None when empty."""
    if not prop_picks:
        return None
    big = 10**9
    max_prop = big if uncapped else settings.alert_max_prop_picks
    when = target.strftime("%a %b %-d")
    lines = [f"**⚾ MLB Forecast — afternoon lineup props · {when}**"]
    lines.append(f"\n__Hitter props__ ({len(prop_picks)})")
    lines.extend(_prop_lines(prop_picks, max_prop=max_prop))
    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


def send_afternoon_props(
    target: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
    uncapped: bool = False,
) -> dict[str, Any]:
    """Second daily pass: alert Premium/Strong hitter props the morning run
    couldn't see because lineups weren't posted yet.

    Re-scores the slate with fresh lineups, then sends only props whose
    (batter, market) wasn't already journaled by the morning run. Silent
    when nothing new qualifies. Deduped once per day via its own marker.
    """
    target = target or date_cls.today()
    marker = _marker_path(target, kind="afternoon")
    if not dry_run and not force and marker.exists():
        return {"sent": False, "reason": "already-sent-today"}

    # New = not in the alerted-props log (what the morning alert actually
    # sent). Scoring/journal state is NOT consulted: the scoring service
    # journals every matchup it scores, so journal presence says nothing
    # about what was alerted. Grading coverage comes from the scoring
    # service's own recording during the refresh below.
    seen = _alerted_prop_keys(target)
    matchups = todays_matchups(target, record=False, refresh=True)
    # Only games that haven't started: by 4:30 the day slate is underway
    # or final, and a prop for a game in progress isn't bettable. This
    # also keeps the morning alert's day-game picks from repeating.
    matchups = [m for m in matchups if _not_started(m.get("scheduled_start"))]
    props = select_prop_picks(matchups)
    new = [p for p in props if (p["player"], p["market"]) not in seen]
    result: dict[str, Any] = {"n_prop": len(props), "n_prop_new": len(new)}

    message = build_afternoon_message(target, new, uncapped=uncapped)
    result["message"] = message
    if message is None:
        result.update(sent=False, reason="no-new-props")
        return result
    if dry_run:
        result.update(sent=False, reason="dry-run")
        return result

    sent = post_to_discord(message)
    result["sent"] = sent
    if sent:
        _log_alerted_props(target, new)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("sent")
        log.info(
            "alerts.afternoon.sent",
            date=target.isoformat(), n_prop=len(props), n_prop_new=len(new),
        )
    else:
        # Deliberately not logged as alerted: a retry (--force) must still
        # see these props as "new" or it would dedupe its own failed attempt.
        result["reason"] = "post-failed-or-no-webhook"
    return result


def send_daily_alert(
    target: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
    uncapped: bool = False,
) -> dict[str, Any]:
    """Gather Premium/Strong picks and post them. Deduped once per day.

    ``force`` ignores the per-day marker; ``dry_run`` builds the message but
    doesn't post (returns it under ``message``); ``uncapped`` includes every
    pick (no per-section cap) for terminal printing.
    """
    target = target or date_cls.today()
    if not dry_run and not force and _marker_path(target).exists():
        return {"sent": False, "reason": "already-sent-today"}

    # Ensure fresh market lines are in the warehouse before selecting game
    # picks, so O/U uses the real posted total instead of the league-average
    # baseline (which renders with a "*"). Best-effort; no-ops without
    # MLB_ODDS_API_KEY. Makes a manual `alert run` self-sufficient rather than
    # depending on morning-sync having ingested odds first.
    try:
        from mlb_model.data.sources import odds_api as _odds_api

        _odds_api.ingest_live_slate()
    except Exception:  # noqa: BLE001 -- odds are best-effort; baseline is the fallback
        log.warning("alerts.odds.ingest_failed")

    game_picks = select_game_picks(target)
    matchups = todays_matchups(target)
    prop_picks = select_prop_picks(matchups)
    pitcher_ks = select_pitcher_strikeouts(matchups)
    message = build_message(target, game_picks, prop_picks, pitcher_ks, uncapped=uncapped)
    result: dict[str, Any] = {
        "n_game": len(game_picks),
        "n_prop": len(prop_picks),
        "n_pitcher_k": len(pitcher_ks),
        "message": message,
    }
    if message is None:
        result.update(sent=False, reason="no-picks")
        return result
    if dry_run:
        result.update(sent=False, reason="dry-run")
        return result

    sent = post_to_discord(message)
    result["sent"] = sent
    if sent:
        marker = _marker_path(target)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("sent")
        _log_pitcher_ks(target, pitcher_ks)
        _log_alerted_props(target, prop_picks)
        log.info("alerts.sent", date=target.isoformat(), n_game=len(game_picks), n_prop=len(prop_picks))
    else:
        result["reason"] = "post-failed-or-no-webhook"
    return result
