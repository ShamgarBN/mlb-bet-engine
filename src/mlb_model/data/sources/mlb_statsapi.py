"""MLB Stats API wrapper (official endpoint, free, no auth required).

Goes through the ``MLB-StatsAPI`` PyPI package, which handles URL building
and pagination. Functions here normalize the library's flat dicts into
the wide DataFrame schemas the warehouse expects.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from typing import Any

import pandas as pd
import statsapi  # MLB-StatsAPI

from mlb_model.logging import get_logger

log = get_logger("data.sources.mlb_statsapi")


# --------------------------------------------------------------------------- #
# Team-id ↔ abbreviation map                                                  #
# --------------------------------------------------------------------------- #
# MLB Stats API returns team_id (numeric) in most endpoints but team_abbr only
# from the boxscore endpoint. We hard-code the map so the schedule normalizer
# can produce abbreviations from the schedule endpoint alone.

_TEAM_ID_TO_ABBR: dict[int, str] = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC", 113: "CIN",
    114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",  119: "LAD",
    120: "WSH", 121: "NYM", 133: "OAK", 134: "PIT", 135: "SD",  136: "SEA",
    137: "SF",  138: "STL", 139: "TB",  140: "TEX", 141: "TOR", 142: "MIN",
    143: "PHI", 144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


def _team_abbr(team_id: int | None) -> str | None:
    if team_id is None:
        return None
    return _TEAM_ID_TO_ABBR.get(int(team_id))


# --------------------------------------------------------------------------- #
# Schedule                                                                    #
# --------------------------------------------------------------------------- #


def fetch_schedule(start_date: date, end_date: date) -> dict[str, Any]:
    """Pull schedule entries for an inclusive date range.

    Returns a dict shaped ``{"games": [...]}`` so downstream code can
    pattern-match on the wrapper key. The library returns a flat list;
    we wrap it for forward compatibility with raw-payload consumers.
    """
    try:
        games = statsapi.schedule(
            start_date=str(start_date), end_date=str(end_date)
        )
    except Exception:  # noqa: BLE001 -- network call; never crash the pipeline
        log.exception("statsapi.fetch_schedule.failed", start=str(start_date), end=str(end_date))
        return {"games": []}
    return {"games": games}


def _parse_status(raw: str | None) -> str:
    """Collapse the StatsAPI status values to the strings our SQL filters use."""
    s = (raw or "").strip()
    if s in {"Final", "Game Over", "Completed Early"}:
        return "Final"
    if s in {"In Progress", "Manager challenge", "Warmup", "Pre-Game"}:
        return "In Progress"
    if s in {"Postponed", "Cancelled", "Suspended", "Suspended: Rain"}:
        return s
    return s or "Scheduled"


def normalize_schedule(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a schedule payload to a warehouse-shaped DataFrame."""
    games = payload.get("games") if isinstance(payload, dict) else payload
    if not games:
        return pd.DataFrame()

    rows = []
    for g in games:
        try:
            game_pk = int(g.get("game_id") or g.get("gamePk"))
        except (TypeError, ValueError):
            continue
        home_id = g.get("home_id")
        away_id = g.get("away_id")
        game_date_s = g.get("game_date")
        season = None
        try:
            season = int(str(game_date_s)[:4]) if game_date_s else None
        except (TypeError, ValueError):
            season = None

        home_score = g.get("home_score")
        away_score = g.get("away_score")
        status = _parse_status(g.get("status"))
        home_win: bool | None = None
        if status == "Final" and home_score is not None and away_score is not None:
            try:
                home_win = int(home_score) > int(away_score)
            except (TypeError, ValueError):
                home_win = None

        rows.append({
            "game_pk": game_pk,
            "game_date": game_date_s,
            "season": season,
            "scheduled_start": g.get("game_datetime"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team_abbr": _team_abbr(home_id),
            "away_team_abbr": _team_abbr(away_id),
            "venue_id": g.get("venue_id"),
            "venue_name": g.get("venue_name"),
            "status": status,
            "home_score": home_score if home_score not in ("", None) else None,
            "away_score": away_score if away_score not in ("", None) else None,
            "home_win": home_win,
            "doubleheader": g.get("doubleheader"),
            "game_number": g.get("game_num"),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


# --------------------------------------------------------------------------- #
# Boxscore                                                                    #
# --------------------------------------------------------------------------- #


def fetch_boxscore(game_pk: int) -> dict[str, Any]:
    """Pull the full boxscore payload for one game."""
    try:
        return statsapi.boxscore_data(int(game_pk))
    except Exception:
        log.exception("statsapi.fetch_boxscore.failed", game_pk=int(game_pk))
        return {}


def _safe_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _ip_to_float(ip: Any) -> float | None:
    """MLB reports innings as '5.2' meaning 5 + 2/3. Convert to a real float."""
    if ip is None or ip == "":
        return None
    s = str(ip)
    if "." not in s:
        try:
            return float(s)
        except ValueError:
            return None
    whole, frac = s.split(".", 1)
    try:
        whole_i = int(whole)
        frac_i = int(frac[:1] or "0")
        return whole_i + frac_i / 3.0
    except ValueError:
        return None


def normalize_team_boxscore(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    """Extract per-team batting totals (one row per side)."""
    if not payload:
        return pd.DataFrame()
    team_info = payload.get("teamInfo", {})
    rows = []
    for side in ("home", "away"):
        totals = payload.get(f"{side}BattingTotals", {})
        team_id = team_info.get(side, {}).get("id")
        if team_id is None:
            continue
        rows.append({
            "game_pk": int(game_pk),
            "team_id": int(team_id),
            "is_home": side == "home",
            "runs": _safe_int(totals.get("r")),
            "hits": _safe_int(totals.get("h")),
            "home_runs": _safe_int(totals.get("hr")),
            "doubles": _safe_int(totals.get("doubles")),
            "triples": _safe_int(totals.get("triples")),
            "walks": _safe_int(totals.get("bb")),
            "strikeouts": _safe_int(totals.get("k")),
            "at_bats": _safe_int(totals.get("ab")),
            "plate_appearances": None,
            "total_bases": None,
            "left_on_base": _safe_int(totals.get("lob")),
        })
    return pd.DataFrame(rows)


def normalize_lineup(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    """Extract starting lineups. battingOrder is encoded as '100', '200', …"""
    if not payload:
        return pd.DataFrame()
    team_info = payload.get("teamInfo", {})
    rows = []
    for side in ("home", "away"):
        team_id = team_info.get(side, {}).get("id")
        if team_id is None:
            continue
        for b in payload.get(f"{side}Batters", []):
            order_raw = b.get("battingOrder")
            if not order_raw:
                continue
            try:
                order_int = int(order_raw)
            except (TypeError, ValueError):
                continue
            # 100,200,...,900 are starters; 101,201,... are substitutions
            if order_int % 100 != 0:
                continue
            slot = order_int // 100
            if slot < 1 or slot > 9:
                continue
            pid = b.get("personId")
            if not pid:
                continue
            rows.append({
                "game_pk": int(game_pk),
                "team_id": int(team_id),
                "batting_order": slot,
                "player_id": int(pid),
                "position": b.get("position") or None,
                "bats": None,
            })
    return pd.DataFrame(rows)


def normalize_pitcher_stats(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    """Per-pitcher stat lines. The first appearance per side is the starter."""
    if not payload:
        return pd.DataFrame()
    team_info = payload.get("teamInfo", {})
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team_id = team_info.get(side, {}).get("id")
        if team_id is None:
            continue
        pitchers = payload.get(f"{side}Pitchers", [])
        # Skip the header row (first entry, personId=0)
        real = [p for p in pitchers if p.get("personId")]
        for i, p in enumerate(real):
            pid = p.get("personId")
            if not pid:
                continue
            rows.append({
                "game_pk": int(game_pk),
                "pitcher_id": int(pid),
                "team_id": int(team_id),
                "is_starter": i == 0,
                "innings_pitched": _ip_to_float(p.get("ip")),
                "batters_faced": None,
                "hits": _safe_int(p.get("h")),
                "runs": _safe_int(p.get("r")),
                "earned_runs": _safe_int(p.get("er")),
                "strikeouts": _safe_int(p.get("k")),
                "walks": _safe_int(p.get("bb")),
                "home_runs": _safe_int(p.get("hr")),
                "pitches_thrown": _safe_int(p.get("p")),
                "strikes_thrown": _safe_int(p.get("s")),
            })
    return pd.DataFrame(rows)


def normalize_batter_stats(game_pk: int, payload: dict[str, Any]) -> pd.DataFrame:
    """Per-batter stat lines for all participants (starters + substitutions)."""
    if not payload:
        return pd.DataFrame()
    team_info = payload.get("teamInfo", {})
    rows: list[dict[str, Any]] = []
    for side in ("home", "away"):
        team_id = team_info.get(side, {}).get("id")
        if team_id is None:
            continue
        for b in payload.get(f"{side}Batters", []):
            pid = b.get("personId")
            if not pid:
                continue
            ab = _safe_int(b.get("ab")) or 0
            bb = _safe_int(b.get("bb")) or 0
            h = _safe_int(b.get("h")) or 0
            d = _safe_int(b.get("doubles")) or 0
            t = _safe_int(b.get("triples")) or 0
            hr = _safe_int(b.get("hr")) or 0
            singles = max(0, h - d - t - hr)
            total_bases = singles + 2 * d + 3 * t + 4 * hr
            rows.append({
                "game_pk": int(game_pk),
                "batter_id": int(pid),
                "team_id": int(team_id),
                "at_bats": ab,
                "plate_appearances": ab + bb,
                "hits": h,
                "doubles": d,
                "triples": t,
                "home_runs": hr,
                "walks": bb,
                "strikeouts": _safe_int(b.get("k")) or 0,
                "hbp": 0,
                "total_bases": total_bases,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Probable pitchers (handedness via /people endpoint)                         #
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=2048)
def _player_throws(person_id: int) -> str | None:
    """Look up a pitcher's throwing hand. Cached per process."""
    try:
        info = statsapi.get(
            "person", {"personId": int(person_id), "hydrate": "currentTeam"}
        )
    except Exception:
        return None
    try:
        return info["people"][0]["pitchHand"]["code"]
    except (KeyError, IndexError, TypeError):
        return None


def fetch_probable_pitchers(game_pk: int) -> pd.DataFrame:
    """Resolve home/away probable pitcher IDs + throw hand for one game."""
    try:
        sched = statsapi.schedule(game_id=int(game_pk))
    except Exception:
        return pd.DataFrame()
    if not sched:
        return pd.DataFrame()
    g = sched[0]
    rows = []
    for side in ("home", "away"):
        name = g.get(f"{side}_probable_pitcher")
        if not name:
            continue
        try:
            results = statsapi.lookup_player(name)
        except Exception:
            results = []
        if not results:
            continue
        pid = results[0].get("id")
        if not pid:
            continue
        rows.append({
            "game_pk": int(game_pk),
            "team_id": g.get(f"{side}_id"),
            "pitcher_id": int(pid),
            "pitcher_name": name,
            "is_home": side == "home",
            "pitcher_throws": _player_throws(int(pid)),
        })
    return pd.DataFrame(rows)
