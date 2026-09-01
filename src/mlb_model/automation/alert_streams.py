"""Staggered, single-purpose Discord alert streams.

The old 11 AM alert crammed game markets, hitter props and pitcher Ks into one
message for the whole slate; the 4:30 PM pass sent leftover hitter props. This
module splits that into small, independently-deduped messages on a stagger, and
adds two "cold bat" streams:

Early window (games starting before ``early_late_cutoff_hour`` local)
  11:00  pitchers   — high-K starters (:func:`send_pitchers` window="early")
  11:07  hitters    — hit/HR/TB props (:func:`send_hitters` window="early")
  11:15  cold hitters  — active .300 bats, coloured by matchup
  11:30  cold sluggers — active HR bats, coloured by matchup
  11:45  game markets  — ML/RL/O-U, whole slate (:func:`send_game_markets`)

Late window (games starting at/after the cutoff)
  16:30  hitters    — remaining props (deduped vs the early hitters)
  16:37  pitchers   — remaining starters (deduped vs the early pitchers)
  16:45  game markets  — late games only, sent only if the pick changed

Each stream reuses the selection + delivery helpers in
:mod:`mlb_model.automation.alerts` and its per-day dedup logs, so a hitter (or
pitcher) alerted in the morning never repeats in the afternoon. Every stream is
best-effort and no-ops without a configured webhook.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any, Literal

import pandas as pd

from mlb_model.automation import alerts
from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("automation.alert_streams")

Window = Literal["early", "late"]

# Cold-bat matchup → colour. Mirrors the FAVORABLE/NEUTRAL/TOUGH grade the
# hitter_slump / slugger_slump engines already attach; NONE = no probable yet.
_COLD_EMOJI = {"FAVORABLE": "🟢", "NEUTRAL": "🟡", "TOUGH": "🔴", "NONE": "⚪"}
_COLD_RANK = {"FAVORABLE": 0, "NEUTRAL": 1, "TOUGH": 2, "NONE": 3}


# --------------------------------------------------------------------------- #
# Early / late window split (by local game start time)
# --------------------------------------------------------------------------- #
def _is_early(scheduled_start: Any, cutoff_hour: int) -> bool:
    """True if the game starts before ``cutoff_hour`` in the Mac's local time.

    ``scheduled_start`` is stored naive-UTC; we localize then convert to local.
    An unknown/unparseable start counts as early so it is never silently
    dropped (the dedup logs stop it repeating in the late window).
    """
    ts = pd.to_datetime(scheduled_start, errors="coerce")
    if ts is pd.NaT:
        return True
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    local = ts.to_pydatetime().astimezone()  # no arg → system local zone
    return local.hour < cutoff_hour


def _filter_window(matchups: list[dict], window: Window) -> list[dict]:
    cutoff = int(settings.early_late_cutoff_hour)
    early = window == "early"
    return [m for m in matchups if _is_early(m.get("scheduled_start"), cutoff) == early]


def _start_by_matchup(matchups: list[dict]) -> dict[str, Any]:
    """Map ``"AWY @ HOM"`` → scheduled_start, to window-filter game picks."""
    out: dict[str, Any] = {}
    for m in matchups:
        away, home = m.get("away_team_abbr"), m.get("home_team_abbr")
        if away and home:
            out[f"{away} @ {home}"] = m.get("scheduled_start")
    return out


# --------------------------------------------------------------------------- #
# Per-stream dedup (pitchers + game markets; hitters reuse alerts' prop log)
# --------------------------------------------------------------------------- #
def _alerted_pitcher_ids(target: date_cls) -> set[int]:
    """Pitcher ids already alerted today (from the pitcher-K log)."""
    try:
        path = alerts._pitcher_k_log_path()
        if not path.exists():
            return set()
        df = pd.read_parquet(path)
        day = df[df["game_date"].astype(str) == target.isoformat()]
        return {int(x) for x in day["pitcher_id"].dropna()}
    except Exception:  # noqa: BLE001 -- dedup is best-effort
        log.warning("streams.pitcher_dedup.read_failed")
        return set()


def _alerted_games_log_path():
    return settings.data_dir / "journal" / "alerted_games.parquet"


def _log_alerted_games(target: date_cls, picks: list[dict]) -> None:
    if not picks:
        return
    try:
        rows = pd.DataFrame([
            {
                "game_date": target.isoformat(),
                "matchup": p.get("matchup"),
                "market": p.get("market"),
                "pick": p.get("pick"),
                "tier": p.get("tier"),
            }
            for p in picks
        ])
        path = _alerted_games_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            rows = pd.concat([pd.read_parquet(path), rows], ignore_index=True)
        rows = rows.drop_duplicates(subset=["game_date", "matchup", "market", "pick"], keep="first")
        rows.to_parquet(path, index=False)
    except Exception:  # noqa: BLE001
        log.warning("streams.game_log.failed")


def _alerted_game_keys(target: date_cls) -> set[tuple[str, str, str]]:
    """(matchup, market, pick) already alerted today — the 'changed?' basis."""
    try:
        path = _alerted_games_log_path()
        if not path.exists():
            return set()
        df = pd.read_parquet(path)
        day = df[df["game_date"].astype(str) == target.isoformat()]
        return set(zip(day["matchup"].astype(str), day["market"].astype(str), day["pick"].astype(str)))
    except Exception:  # noqa: BLE001
        log.warning("streams.game_dedup.read_failed")
        return set()


# --------------------------------------------------------------------------- #
# Message builders
# --------------------------------------------------------------------------- #
def _header(title: str, target: date_cls) -> str:
    return f"**⚾ MLB Forecast — {title} · {target.strftime('%a %b %-d')}**"


def build_pitcher_message(target: date_cls, pitcher_ks: list[dict], *, window: Window) -> str | None:
    if not pitcher_ks:
        return None
    tag = "early" if window == "early" else "late"
    lines = [_header(f"pitcher strikeouts ({tag})", target)]
    lines.append(f"\n__Projected Ks__ ({len(pitcher_ks)})")
    for k in pitcher_ks:
        vs = f" vs {k['vs_team']}" if k.get("vs_team") else ""
        lines.append(f"• {k['pitcher']} ({k['team']}){vs} — {k['est_k']:g} K")
    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


def build_hitter_message(target: date_cls, prop_picks: list[dict], *, window: Window, uncapped: bool = False) -> str | None:
    if not prop_picks:
        return None
    big = 10**9
    max_prop = big if uncapped else settings.alert_max_prop_picks
    tag = "early" if window == "early" else "late"
    lines = [_header(f"hitter props ({tag})", target)]
    lines.append(f"\n__Hitter props__ ({len(prop_picks)})")
    lines.extend(alerts._prop_lines(prop_picks, max_prop=max_prop))
    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


def build_game_message(target: date_cls, game_picks: list[dict], *, window: Window, uncapped: bool = False) -> str | None:
    if not game_picks:
        return None
    big = 10**9
    max_game = big if uncapped else settings.alert_max_game_picks
    tag = "whole slate" if window == "early" else "late games — updated"
    lines = [_header(f"game markets ({tag})", target)]
    shown = game_picks[:max_game]
    lines.append(f"\n__Game markets__ ({len(game_picks)})")
    for p in shown:
        conf = f" {p['confidence'] * 100:.0f}%" if p.get("confidence") is not None else ""
        edge = f" ({p['edge_pp']:+.0f}pp)" if p.get("edge_pp") is not None else ""
        lines.append(
            f"{alerts._TIER_EMOJI.get(p['tier'], '•')} {p['matchup']} "
            f"{p['market']}: {p['pick']}{conf}{edge}"
        )
    if len(game_picks) > len(shown):
        lines.append(f"_…and {len(game_picks) - len(shown)} more (see the app)_")
    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


def build_cold_message(
    target: date_cls, rows: list[dict], *, title: str, stat_fn, drought_word: str,
) -> str | None:
    """Coloured cold-bat list (🟢 favorable / 🟡 neutral / 🔴 tough)."""
    if not rows:
        return None
    lines = [_header(title, target)]
    lines.append("\n🟢 favorable · 🟡 neutral · 🔴 tough matchup")
    for r in rows:
        label = r.get("matchup_label", "NONE")
        emoji = _COLD_EMOJI.get(label, "⚪")
        sp = alerts._surname(r.get("matchup_pitcher") or "")
        vs = f" vs {sp}" if sp else " — no probable yet"
        lines.append(
            f"{emoji} {r['name']} ({r['team']}) {stat_fn(r)} · "
            f"{r['drought_games']}G {drought_word}{vs}"
        )
    lines.append("\n_Research only — not financial advice._")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Stream senders
# --------------------------------------------------------------------------- #
def _guard(target: date_cls, kind: str, *, force: bool, dry_run: bool):
    """Per-day marker gate shared by every stream. Returns the marker path."""
    marker = alerts._marker_path(target, kind=kind)
    if not dry_run and not force and marker.exists():
        return None
    return marker


def _finish(marker, message: str, *, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        return {"sent": False, "reason": "dry-run", "message": message}
    sent = alerts.post_to_discord(message)
    if sent and marker is not None:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("sent")
    return {"sent": sent, "message": message,
            "reason": None if sent else "post-failed-or-no-webhook"}


def send_pitchers(
    target: date_cls | None = None, *, window: Window = "early",
    force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """High-K starters for the window, deduped vs pitchers already alerted."""
    target = target or date_cls.today()
    kind = f"{window}-pitchers"
    marker = _guard(target, kind, force=force, dry_run=dry_run)
    if marker is None:
        return {"sent": False, "reason": "already-sent-today"}

    matchups = alerts.todays_matchups(target, record=False, refresh=(window == "late"))
    matchups = _filter_window(matchups, window)
    ks = alerts.select_pitcher_strikeouts(matchups)
    seen = _alerted_pitcher_ids(target)
    new = [k for k in ks if k.get("pitcher_id") not in seen]
    result = {"n_total": len(ks), "n_new": len(new)}

    message = build_pitcher_message(target, new, window=window)
    if message is None:
        return {**result, "sent": False, "reason": "nothing-new"}
    out = _finish(marker, message, dry_run=dry_run)
    if out.get("sent"):
        alerts._log_pitcher_ks(target, new)
        log.info("streams.pitchers.sent", window=window, n_new=len(new))
    return {**result, **out}


def send_hitters(
    target: date_cls | None = None, *, window: Window = "early",
    force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Hit/HR/TB props for the window, deduped vs props already alerted."""
    target = target or date_cls.today()
    kind = f"{window}-hitters"
    marker = _guard(target, kind, force=force, dry_run=dry_run)
    if marker is None:
        return {"sent": False, "reason": "already-sent-today"}

    # Record matchups on the early pass (feeds grading); the late pass refreshes
    # to pick up newly-posted lineups. Only alert not-yet-started games.
    matchups = alerts.todays_matchups(target, record=(window == "early"), refresh=(window == "late"))
    matchups = _filter_window(matchups, window)
    matchups = [m for m in matchups if alerts._not_started(m.get("scheduled_start"))]
    props = alerts.select_prop_picks(matchups)
    seen = alerts._alerted_prop_keys(target)
    new = [p for p in props if (p["player"], p["market"]) not in seen]
    result = {"n_total": len(props), "n_new": len(new)}

    message = build_hitter_message(target, new, window=window)
    if message is None:
        return {**result, "sent": False, "reason": "nothing-new"}
    out = _finish(marker, message, dry_run=dry_run)
    if out.get("sent"):
        alerts._log_alerted_props(target, new)
        log.info("streams.hitters.sent", window=window, n_new=len(new))
    return {**result, **out}


def send_game_markets(
    target: date_cls | None = None, *, window: Window = "early",
    force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """ML/RL/O-U picks. Early = whole slate; late = late games, only if the
    (matchup, market, pick) differs from what the early message already sent."""
    target = target or date_cls.today()
    kind = f"{window}-games"
    marker = _guard(target, kind, force=force, dry_run=dry_run)
    if marker is None:
        return {"sent": False, "reason": "already-sent-today"}

    # Re-ingest odds only in the afternoon (catch line movement); the morning
    # sync already ingested for the early pass.
    if window == "late":
        try:
            from mlb_model.data.sources import odds_api as _odds_api
            _odds_api.ingest_live_slate()
        except Exception:  # noqa: BLE001 -- best-effort
            log.warning("streams.games.odds_ingest_failed")

    picks = alerts.select_game_picks(target)
    if window == "late":
        starts = _start_by_matchup(alerts.todays_matchups(target, record=False))
        cutoff = int(settings.early_late_cutoff_hour)
        picks = [p for p in picks if not _is_early(starts.get(p["matchup"]), cutoff)]
        seen = _alerted_game_keys(target)
        picks = [p for p in picks if (p["matchup"], p["market"], p["pick"]) not in seen]
    result = {"n": len(picks)}

    message = build_game_message(target, picks, window=window)
    if message is None:
        reason = "no-change" if window == "late" else "no-picks"
        return {**result, "sent": False, "reason": reason}
    out = _finish(marker, message, dry_run=dry_run)
    if out.get("sent"):
        _log_alerted_games(target, picks)
        log.info("streams.games.sent", window=window, n=len(picks))
    return {**result, **out}


def _cold_rows(snapshot: dict, list_key: str) -> list[dict]:
    """Active-today cold bats from a service snapshot, best matchup first."""
    rows = [r for r in snapshot.get(list_key, []) if r.get("active_today")]
    rows.sort(key=lambda r: (_COLD_RANK.get(r.get("matchup_label", "NONE"), 3),
                             -int(r.get("drought_games", 0))))
    return rows


def send_cold_hitters(
    target: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
    refresh: bool = True,
) -> dict[str, Any]:
    """Active .300 bats in a hit drought, coloured by next-pitcher matchup."""
    target = target or date_cls.today()
    marker = _guard(target, "cold-hitters", force=force, dry_run=dry_run)
    if marker is None:
        return {"sent": False, "reason": "already-sent-today"}

    from mlb_model.app import hitter_service

    snapshot = hitter_service.get_report(target.year, refresh=refresh, as_of=target)
    rows = _cold_rows(snapshot, "hitters")
    result = {"n": len(rows)}
    message = build_cold_message(
        target, rows, title="cold .300 bats",
        stat_fn=lambda r: f".{r['batting_average'] * 1000:.0f}",
        drought_word="hitless",
    )
    if message is None:
        return {**result, "sent": False, "reason": "none-active-today"}
    out = _finish(marker, message, dry_run=dry_run)
    if out.get("sent"):
        log.info("streams.cold_hitters.sent", n=len(rows))
    return {**result, **out}


def send_cold_sluggers(
    target: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
    refresh: bool = True,
) -> dict[str, Any]:
    """Active HR bats in a homer drought, coloured by next-pitcher matchup."""
    target = target or date_cls.today()
    marker = _guard(target, "cold-sluggers", force=force, dry_run=dry_run)
    if marker is None:
        return {"sent": False, "reason": "already-sent-today"}

    from mlb_model.app import slugger_service

    snapshot = slugger_service.get_report(target.year, refresh=refresh, as_of=target)
    rows = _cold_rows(snapshot, "sluggers")
    result = {"n": len(rows)}
    message = build_cold_message(
        target, rows, title="cold sluggers",
        stat_fn=lambda r: f"{r['home_runs']} HR",
        drought_word="cold",
    )
    if message is None:
        return {**result, "sent": False, "reason": "none-active-today"}
    out = _finish(marker, message, dry_run=dry_run)
    if out.get("sent"):
        log.info("streams.cold_sluggers.sent", n=len(rows))
    return {**result, **out}


# --------------------------------------------------------------------------- #
# Dispatcher (one entry point for the CLI / LaunchAgents)
# --------------------------------------------------------------------------- #
_STREAMS = {
    "early-pitchers": lambda t, **kw: send_pitchers(t, window="early", **kw),
    "early-hitters": lambda t, **kw: send_hitters(t, window="early", **kw),
    "cold-hitters": send_cold_hitters,
    "cold-sluggers": send_cold_sluggers,
    "early-games": lambda t, **kw: send_game_markets(t, window="early", **kw),
    "late-hitters": lambda t, **kw: send_hitters(t, window="late", **kw),
    "late-pitchers": lambda t, **kw: send_pitchers(t, window="late", **kw),
    "late-games": lambda t, **kw: send_game_markets(t, window="late", **kw),
}
STREAM_NAMES = tuple(_STREAMS)


def run_stream(
    name: str, target: date_cls | None = None, *, force: bool = False, dry_run: bool = False,
) -> dict[str, Any]:
    """Run one named stream. Raises KeyError for an unknown name."""
    if name not in _STREAMS:
        raise KeyError(f"unknown stream {name!r}; choose from {', '.join(STREAM_NAMES)}")
    return _STREAMS[name](target, force=force, dry_run=dry_run)
