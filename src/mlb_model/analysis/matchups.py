"""Favorable-matchup layer for the slumping-slugger report.

For each cold slugger, look up their team's next *unplayed* game, find the
opposing probable pitcher, and grade how hittable that pitcher is — with a
bias toward power, since the use case is a HR bounce-back. A slumping slugger
facing a homer-prone pitcher who is soft against their handedness is the
prime bet; one facing a stingy ace is not.

All data is live from the MLB Stats API (the warehouse only runs through
spring). Best-effort throughout: any lookup failure yields a NONE verdict so
the rest of the report still renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import statsapi

from mlb_model.logging import get_logger

log = get_logger("analysis.matchups")

# League baselines (mirror mlb_model.scoring.hitter — kept local to avoid
# importing the scoring stack here).
LG_OPS_ALLOWED = 0.731
LG_HR9 = 1.20

# Statuses that mean the game has started or finished — skip them; we want
# the next game a player could actually bat in.
_STARTED_OR_DONE = {
    "Final", "Game Over", "In Progress", "Completed Early", "Suspended",
    "Postponed", "Cancelled",
}


@dataclass(slots=True)
class PitcherProfile:
    pitcher_id: int
    name: str
    throws: str | None          # 'L' | 'R'
    innings: float
    home_runs: int
    hr9: float | None
    ops_allowed: float | None
    vs_l_ops: float | None       # OPS allowed to left-handed batters
    vs_r_ops: float | None       # OPS allowed to right-handed batters


@dataclass(slots=True)
class TeamMatchup:
    game_date: date
    opp_pitcher: PitcherProfile


@dataclass(slots=True)
class MatchupVerdict:
    label: str                   # FAVORABLE | NEUTRAL | TOUGH | NONE
    opp_pitcher: str | None = None
    opp_throws: str | None = None
    game_date: date | None = None
    ops_allowed: float | None = None   # vs the batter's effective hand
    hr9: float | None = None
    sample_ip: float | None = None
    edge: float | None = None
    detail: str = "No upcoming probable pitcher"


def _f(v) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _ip_to_float(ip) -> float:
    """MLB innings are '16.1' meaning 16 + 1/3, not 16.1."""
    s = str(ip or "0")
    if "." in s:
        whole, _, frac = s.partition(".")
        return int(whole or 0) + {"0": 0.0, "1": 1 / 3, "2": 2 / 3}.get(frac, 0.0)
    return float(s or 0)


def fetch_pitcher_profile(pitcher_id: int, season: int) -> PitcherProfile | None:
    """Season pitching line + vs-handedness OPS splits for one pitcher."""
    try:
        res = statsapi.get(
            "people",
            {
                "personIds": int(pitcher_id),
                "hydrate": (
                    f"stats(group=[pitching],type=[season,statSplits],"
                    f"sitCodes=[vl,vr],season={int(season)})"
                ),
            },
        )
        person = res["people"][0]
    except Exception:  # noqa: BLE001 -- best-effort
        log.warning("matchup.pitcher.fetch_failed", pitcher_id=int(pitcher_id))
        return None

    throws = (person.get("pitchHand") or {}).get("code")
    name = person.get("fullName", str(pitcher_id))
    ip = 0.0
    hr = 0
    hr9 = ops = None
    vs_l = vs_r = None
    for block in person.get("stats", []):
        disp = block.get("type", {}).get("displayName")
        for sp in block.get("splits", []):
            st = sp.get("stat", {})
            if disp == "season":
                ip = _ip_to_float(st.get("inningsPitched"))
                hr = int(st.get("homeRuns", 0) or 0)
                hr9 = _f(st.get("homeRunsPer9"))
                ops = _f(st.get("ops"))
            elif disp == "statSplits":
                code = sp.get("split", {}).get("code")
                if code == "vl":
                    vs_l = _f(st.get("ops"))
                elif code == "vr":
                    vs_r = _f(st.get("ops"))
    return PitcherProfile(
        pitcher_id=int(pitcher_id), name=name, throws=throws, innings=ip,
        home_runs=hr, hr9=hr9, ops_allowed=ops, vs_l_ops=vs_l, vs_r_ops=vs_r,
    )


@dataclass(slots=True)
class TeamDay:
    """A team's situation on the selected day."""

    plays_today: bool                  # has a not-yet-started game on `as_of`
    matchup: TeamMatchup | None        # next unplayed game's opposing pitcher


def _first_matchup(sched: list[dict], team_id: int, season: int,
                   as_of: date) -> TeamMatchup | None:
    """First upcoming game (with a known opposing probable), graded-ready."""
    for g in sched:
        if g.get("status") in _STARTED_OR_DONE:
            continue
        # The opposing probable is on the side our team is NOT.
        is_home = g.get("home_id") == int(team_id)
        opp_name = g.get("away_probable_pitcher") if is_home else g.get("home_probable_pitcher")
        if not opp_name:
            continue
        try:
            matches = statsapi.lookup_player(opp_name)
            if not matches:
                continue
            opp_id = matches[0]["id"]
        except Exception:  # noqa: BLE001
            continue
        profile = fetch_pitcher_profile(opp_id, season)
        if profile is None:
            continue
        try:
            gd = date.fromisoformat(str(g.get("game_date"))[:10])
        except ValueError:
            gd = as_of
        return TeamMatchup(game_date=gd, opp_pitcher=profile)
    return None


def team_status(team_id: int, season: int, *, as_of: date | None = None,
                window_days: int = 6) -> TeamDay:
    """Whether the team has a (bettable) game today, plus its next matchup.

    One schedule fetch powers both. ``plays_today`` is True only when there's
    a game on ``as_of`` that hasn't started yet — i.e. an actionable, pre-game
    spot. A game already in progress / final today counts as "not playing".
    """
    as_of = as_of or date.today()
    try:
        sched = statsapi.schedule(
            start_date=as_of.isoformat(),
            end_date=(as_of + timedelta(days=window_days)).isoformat(),
            team=int(team_id),
        )
    except Exception:  # noqa: BLE001
        log.warning("matchup.schedule.fetch_failed", team_id=int(team_id))
        return TeamDay(plays_today=False, matchup=None)

    today_iso = as_of.isoformat()
    plays_today = any(
        str(g.get("game_date"))[:10] == today_iso and g.get("status") not in _STARTED_OR_DONE
        for g in sched
    )
    return TeamDay(plays_today=plays_today, matchup=_first_matchup(sched, team_id, season, as_of))


def next_team_matchup(team_id: int, season: int, *, as_of: date | None = None,
                      window_days: int = 6) -> TeamMatchup | None:
    """Back-compat wrapper: just the next matchup (see :func:`team_status`)."""
    return team_status(team_id, season, as_of=as_of, window_days=window_days).matchup


def batter_hands(player_ids: list[int]) -> dict[int, str]:
    """Batting side ('L'/'R'/'S') for many players in one call."""
    if not player_ids:
        return {}
    try:
        ids = ",".join(str(int(p)) for p in player_ids)
        res = statsapi.get("people", {"personIds": ids})
        return {
            p["id"]: (p.get("batSide") or {}).get("code", "R")
            for p in res.get("people", [])
        }
    except Exception:  # noqa: BLE001
        log.warning("matchup.batter_hands.failed")
        return {}


def grade_for_batter(batter_hand: str | None, tm: TeamMatchup | None) -> MatchupVerdict:
    """Grade how favorable the matchup is for a HR bounce-back.

    Combines the pitcher's season HR/9 with the OPS he allows to the batter's
    effective hand (a switch hitter takes the platoon side vs the pitcher).
    Positive edge = hittable / homer-prone = FAVORABLE.
    """
    if tm is None:
        return MatchupVerdict(label="NONE")
    p = tm.opp_pitcher

    # Effective batter hand: switch hitters bat opposite the pitcher.
    hand = batter_hand or "R"
    if hand == "S":
        hand = "R" if p.throws == "L" else "L"
    ops_vs = p.vs_l_ops if hand == "L" else p.vs_r_ops
    ops_used = ops_vs if ops_vs is not None else p.ops_allowed

    hr9 = p.hr9
    parts = []
    edge = 0.0
    if ops_used is not None:
        edge += (ops_used - LG_OPS_ALLOWED) / 0.120
        parts.append(f"{ops_used:.3f} OPS vs {hand}HB")
    if hr9 is not None:
        edge += (hr9 - LG_HR9) / 0.90
        parts.append(f"{hr9:.2f} HR/9")

    if ops_used is None and hr9 is None:
        label = "NONE"
    elif edge >= 0.8:
        label = "FAVORABLE"
    elif edge <= -0.8:
        label = "TOUGH"
    else:
        label = "NEUTRAL"

    detail = f"vs {p.name} ({p.throws or '?'}HP)"
    if parts:
        detail += " — " + ", ".join(parts)
    if p.innings < 20:
        detail += f" (small sample, {p.innings:.0f} IP)"

    return MatchupVerdict(
        label=label,
        opp_pitcher=p.name,
        opp_throws=p.throws,
        game_date=tm.game_date,
        ops_allowed=ops_used,
        hr9=hr9,
        sample_ip=p.innings,
        edge=round(edge, 2),
        detail=detail,
    )
