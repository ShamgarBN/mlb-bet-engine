"""Track big home-run hitters who have gone cold — and explain why.

The hunch: the season's premier power bats (15+ HR) occasionally hit a
HR drought. Some droughts are noise; some are an injury or an external
factor (benching, optioning, personal leave). This module finds the
slumping sluggers and tries to tell the two apart.

Pipeline
--------
1. :func:`fetch_season_hitting` — pull every hitter's season HR/PA/G totals
   live from the MLB Stats API (deduping traded players).
2. :func:`threshold_breakdown` — compute the share of players at/above the
   HR threshold across three denominators (all-PA, has-HR, qualified).
   :func:`append_history` persists one snapshot per run so the percentage
   can be tracked as a *moving number* through the season.
3. :func:`fetch_gamelog` / :func:`hr_drought` — per-player game log and the
   trailing count of consecutive *appeared* games with no HR.
4. :func:`find_slumping_sluggers` — assemble the flagged list, attaching an
   absence signal (gamelog gap) and a live-news verdict (injury/external).

Everything that touches the network is best-effort and logged; a single
failed player never aborts the run. The data structures are plain
dataclasses so the desktop app can consume the same engine later.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import statsapi

from mlb_model.analysis import matchups, news, transactions
from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("analysis.slugger_slump")

DEFAULT_MIN_DROUGHT = 5
# A regular/qualified hitter accrues ~3.1 PA per team game; by midseason the
# everyday bats are well past 200 PA. We use a fixed floor rather than the
# official 3.1*games rule because it is robust to mid-run schedule drift and
# matches the "regular hitter" denominator we report to the user.
QUALIFIED_PA_FLOOR = 200

# The HR bar is dynamic: each run it tracks the top ``DEFAULT_TARGET_PCT`` of
# qualified hitters by HR, so the list stays "elite power" as the league
# accumulates homers through the season. ``DEFAULT_HR_FLOOR`` keeps it from
# dropping below the original early-season standard. (A fixed bar can still be
# forced via the ``threshold=`` arg.)
DEFAULT_TARGET_PCT = 10.0
DEFAULT_HR_FLOOR = 15
DEFAULT_HR_THRESHOLD = DEFAULT_HR_FLOOR  # back-compat alias for callers/tests

HISTORY_PATH = settings.processed_dir / "slugger_hr_pct_history.csv"


def dynamic_threshold(
    hitters: list["HitterSeason"],
    *,
    target_pct: float = DEFAULT_TARGET_PCT,
    floor: int = DEFAULT_HR_FLOOR,
    pa_floor: int = QUALIFIED_PA_FLOOR,
) -> int:
    """HR bar that captures roughly the top ``target_pct`` of qualified hitters.

    Computed from the HR total of the qualified hitter sitting at the
    ``target_pct`` rank (e.g. the 22nd of 221 for 10%), then floored at
    ``floor``. Integer HR totals + ties mean the captured share is approximate
    and lands on whole-HR steps; the floor guarantees the bar only ever rises
    above the early-season standard, never below it.
    """
    qual = sorted(
        (h.home_runs for h in hitters if h.plate_appearances >= pa_floor),
        reverse=True,
    )
    if not qual:
        return floor
    import math

    idx = min(len(qual) - 1, max(0, math.ceil(target_pct / 100.0 * len(qual)) - 1))
    return max(floor, int(qual[idx]))


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HitterSeason:
    player_id: int
    name: str
    team: str | None
    team_id: int | None
    home_runs: int
    plate_appearances: int
    at_bats: int
    games: int


@dataclass(slots=True)
class DenomShare:
    label: str
    denom_count: int
    n_at_threshold: int

    @property
    def pct(self) -> float:
        return 100.0 * self.n_at_threshold / self.denom_count if self.denom_count else 0.0


@dataclass(slots=True)
class ThresholdBreakdown:
    season: int
    threshold: int
    as_of: date
    shares: dict[str, DenomShare]

    @property
    def n_at_threshold(self) -> int:
        # Identical across denominators (numerator is the same set of players).
        return next(iter(self.shares.values())).n_at_threshold if self.shares else 0


@dataclass(slots=True)
class SluggerStatus:
    player_id: int
    name: str
    team: str | None
    home_runs: int
    games: int
    drought_games: int
    last_hr_date: date | None
    last_game_date: date | None
    days_since_last_game: int | None
    is_absent: bool
    # Authoritative status from the MLB transactions feed.
    verified_cause: str = "unverified"  # injury-verified | external-verified | unverified
    il_type: str | None = None
    il_start: date | None = None
    injury_note: str | None = None
    external_move: str | None = None
    move_date: date | None = None
    # Supplementary context only — never changes the verdict.
    news_cause: str = "n/a"
    news_headlines: list[news.Headline] = field(default_factory=list)
    # Next-game pitching matchup (only graded for bettable/UNCLEAR players).
    matchup_label: str = "NONE"  # FAVORABLE | NEUTRAL | TOUGH | NONE
    matchup_pitcher: str | None = None
    matchup_throws: str | None = None
    matchup_date: date | None = None
    matchup_ops: float | None = None
    matchup_hr9: float | None = None
    matchup_edge: float | None = None
    matchup_detail: str = ""
    # Whether the player's team has a not-yet-started game on the selected day.
    plays_today: bool = False

    @property
    def active_today(self) -> bool:
        """The primary betting target: a bettable cold bat playing today.

        Top-N% HR, in a slump, cause UNCLEAR (not on the IL / optioned), team
        has a pre-game game today, and the player is actually in the mix (not
        sitting out / day-to-day)."""
        return (
            self.verified_cause == "unverified"
            and self.plays_today
            and not self.is_absent
        )

    @property
    def status_label(self) -> str:
        """Short tag for tables / UI badges."""
        if self.verified_cause == "injury-verified":
            return "INJURY (verified)"
        if self.verified_cause == "external-verified":
            return "EXTERNAL (verified)"
        return "UNCLEAR"

    @property
    def advisory(self) -> str:
        """One-line read. Injuries/externals are only ever asserted when MLB
        logged the move; everything else is honestly left UNCLEAR."""
        if self.verified_cause == "injury-verified":
            since = f" since {self.il_start.isoformat()}" if self.il_start else ""
            note = f" — {self.injury_note}" if self.injury_note else ""
            il = self.il_type or "IL"
            return f"INJURY (verified) — on the {il} IL{since}{note}"
        if self.verified_cause == "external-verified":
            when = f" on {self.move_date.isoformat()}" if self.move_date else ""
            return f"EXTERNAL (verified) — {self.external_move}{when}"
        # Unverified: report the cold streak honestly, noting an absence the
        # transactions feed didn't explain (likely a benching / day-to-day).
        if self.is_absent:
            return (
                f"UNCLEAR — no appearance in {self.days_since_last_game} days but no IL "
                "stint or roster move on file (likely day-to-day / benched)"
            )
        return "UNCLEAR — cold streak, no IL stint or roster move on file"


# --------------------------------------------------------------------------- #
# 1. Season hitting totals
# --------------------------------------------------------------------------- #
def fetch_season_hitting(season: int) -> list[HitterSeason]:
    """Pull every hitter's season totals, deduping players who were traded.

    Traded players appear once per team in the splits; we sum their lines so
    a player's HR total reflects the whole season, not one stint.
    """
    res = statsapi.get(
        "stats",
        {
            "stats": "season",
            "group": "hitting",
            "season": int(season),
            "sportId": 1,
            "limit": 3000,
            "playerPool": "all",
        },
    )
    splits = res.get("stats", [{}])[0].get("splits", []) if res.get("stats") else []

    agg: dict[int, HitterSeason] = {}
    for s in splits:
        player = s.get("player", {})
        pid = player.get("id")
        if pid is None:
            continue
        st = s.get("stat", {})
        team_obj = s.get("team") or {}
        team = team_obj.get("abbreviation") or team_obj.get("name")
        team_id = team_obj.get("id")
        cur = agg.get(pid)
        if cur is None:
            agg[pid] = HitterSeason(
                player_id=pid,
                name=player.get("fullName", str(pid)),
                team=team,
                team_id=team_id,
                home_runs=int(st.get("homeRuns", 0) or 0),
                plate_appearances=int(st.get("plateAppearances", 0) or 0),
                at_bats=int(st.get("atBats", 0) or 0),
                games=int(st.get("gamesPlayed", 0) or 0),
            )
        else:
            cur.home_runs += int(st.get("homeRuns", 0) or 0)
            cur.plate_appearances += int(st.get("plateAppearances", 0) or 0)
            cur.at_bats += int(st.get("atBats", 0) or 0)
            cur.games += int(st.get("gamesPlayed", 0) or 0)
            # Keep the most recent team for a traded player.
            if team:
                cur.team = team
                cur.team_id = team_id
    return list(agg.values())


# --------------------------------------------------------------------------- #
# 2. Threshold percentage (the "moving number")
# --------------------------------------------------------------------------- #
def threshold_breakdown(
    hitters: list[HitterSeason],
    *,
    season: int,
    threshold: int = DEFAULT_HR_THRESHOLD,
    as_of: date | None = None,
) -> ThresholdBreakdown:
    """Share of players at/above the HR threshold across three denominators."""
    as_of = as_of or date.today()
    at_thr = [h for h in hitters if h.home_runs >= threshold]
    n = len(at_thr)

    with_pa = [h for h in hitters if h.plate_appearances > 0]
    with_hr = [h for h in hitters if h.home_runs >= 1]
    qualified = [h for h in hitters if h.plate_appearances >= QUALIFIED_PA_FLOOR]

    shares = {
        "all_pa": DenomShare("All players with ≥1 PA", len(with_pa), n),
        "has_hr": DenomShare("Players with ≥1 HR", len(with_hr), n),
        "qualified": DenomShare(
            f"Regular hitters (≥{QUALIFIED_PA_FLOOR} PA)", len(qualified), n
        ),
    }
    return ThresholdBreakdown(season=season, threshold=threshold, as_of=as_of, shares=shares)


def append_history(breakdown: ThresholdBreakdown, *, path: Path | None = None) -> Path:
    """Append one snapshot row per denominator; idempotent for a given day.

    Re-running on the same date overwrites that date's rows so the file holds
    exactly one snapshot per (as_of, season, threshold, denominator).
    """
    path = path or HISTORY_PATH  # resolved at call time so the global is patchable
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["as_of", "season", "threshold", "denominator", "denom_count", "n_at_threshold", "pct"]

    rows: list[dict] = []
    if path.exists():
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))

    key = (breakdown.as_of.isoformat(), str(breakdown.season), str(breakdown.threshold))
    rows = [
        r for r in rows
        if (r["as_of"], r["season"], r["threshold"]) != key
    ]
    for label_key, share in breakdown.shares.items():
        rows.append(
            {
                "as_of": breakdown.as_of.isoformat(),
                "season": str(breakdown.season),
                "threshold": str(breakdown.threshold),
                "denominator": label_key,
                "denom_count": str(share.denom_count),
                "n_at_threshold": str(share.n_at_threshold),
                "pct": f"{share.pct:.4f}",
            }
        )
    rows.sort(key=lambda r: (r["as_of"], r["season"], r["threshold"], r["denominator"]))
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


# --------------------------------------------------------------------------- #
# 3. Per-player game log + drought
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GameLogEntry:
    game_date: date
    home_runs: int


def fetch_gamelog(player_id: int, season: int) -> list[GameLogEntry]:
    """Chronological hitting game log for a player; [] on any failure."""
    try:
        res = statsapi.get(
            "person",
            {
                "personId": int(player_id),
                "hydrate": f"stats(group=[hitting],type=[gameLog],season={int(season)})",
            },
        )
        stats = res.get("people", [{}])[0].get("stats", [])
        if not stats:
            return []
        splits = stats[0].get("splits", [])
    except Exception:  # noqa: BLE001 -- one bad player shouldn't abort the run
        log.warning("gamelog.fetch.failed", player_id=int(player_id))
        return []

    out: list[GameLogEntry] = []
    for s in splits:
        raw = s.get("date")
        if not raw:
            continue
        try:
            d = datetime.strptime(raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        hr = int(s.get("stat", {}).get("homeRuns", 0) or 0)
        out.append(GameLogEntry(game_date=d, home_runs=hr))
    out.sort(key=lambda e: e.game_date)
    return out


def hr_drought(gamelog: list[GameLogEntry]) -> int:
    """Trailing consecutive *appeared* games with zero home runs."""
    n = 0
    for entry in reversed(gamelog):
        if entry.home_runs > 0:
            break
        n += 1
    return n


def last_hr_date(gamelog: list[GameLogEntry]) -> date | None:
    for entry in reversed(gamelog):
        if entry.home_runs > 0:
            return entry.game_date
    return None


# --------------------------------------------------------------------------- #
# 4. Assemble the slumping-slugger report
# --------------------------------------------------------------------------- #
def absence_verdict(
    days_since: int | None,
    games_missed: int | None,
    *,
    absence_days: int = 4,
    absence_games: int = 3,
) -> bool:
    """Is a player "absent" (hidden IL stint / benching signal)?

    Absent = idle for ``absence_days``+ calendar days AND the team played
    ``absence_games``+ games without him. Calendar days alone can't tell a
    benched player from a paused league — the All-Star break idles everyone
    for 4-5 days and used to flag the entire majors as absent. Counting
    TEAM games missed is break-proof and a sharper signal.

    ``games_missed`` is None when the team schedule couldn't be fetched;
    fall back to the calendar-only heuristic in that case.
    """
    if days_since is None or days_since < absence_days:
        return False
    if games_missed is None:
        return True
    return games_missed >= absence_games


def find_slumping_sluggers(
    season: int,
    *,
    threshold: int = DEFAULT_HR_THRESHOLD,
    min_drought: int = DEFAULT_MIN_DROUGHT,
    as_of: date | None = None,
    absence_days: int = 4,
    absence_games: int = 3,
    with_news: bool = True,
    with_matchups: bool = True,
    hitters: list[HitterSeason] | None = None,
) -> list[SluggerStatus]:
    """Find threshold sluggers in a HR drought of ``min_drought``+ games.

    A player is also flagged (and reported regardless of drought length) when
    absent: idle ``absence_days``+ calendar days while his team played
    ``absence_games``+ games without him (see :func:`absence_verdict`) — an
    absence usually means an IL stint or benching, which is exactly the
    "external factor" signal.
    """
    as_of = as_of or date.today()
    hitters = hitters if hitters is not None else fetch_season_hitting(season)
    sluggers = [h for h in hitters if h.home_runs >= threshold]
    log.info("slugger.universe", n=len(sluggers), threshold=threshold)

    # Team schedules for the absence check, fetched lazily and cached per
    # team (only players past the calendar prefilter need one).
    sched_cache: dict[int, list[date] | None] = {}

    def _games_missed(team_id: int | None, last_game: date) -> int | None:
        if team_id is None:
            return None
        if team_id not in sched_cache:
            sched_cache[team_id] = matchups.team_completed_game_dates(
                team_id, as_of - timedelta(days=30), as_of
            )
        dates = sched_cache[team_id]
        if dates is None:
            return None
        return sum(1 for d in dates if d > last_game)

    flagged: list[tuple[HitterSeason, SluggerStatus]] = []
    for h in sluggers:
        glog = fetch_gamelog(h.player_id, season)
        drought = hr_drought(glog)
        last_game = glog[-1].game_date if glog else None
        days_since = (as_of - last_game).days if last_game else None
        is_absent = False
        if days_since is not None and days_since >= absence_days:
            is_absent = absence_verdict(
                days_since,
                _games_missed(h.team_id, last_game),
                absence_days=absence_days,
                absence_games=absence_games,
            )

        # Flag if cold (drought) OR absent. An absent player has a frozen
        # drought count but is the most interesting case for the user.
        if drought < min_drought and not is_absent:
            continue

        flagged.append(
            (
                h,
                SluggerStatus(
                    player_id=h.player_id,
                    name=h.name,
                    team=h.team,
                    home_runs=h.home_runs,
                    games=h.games,
                    drought_games=drought,
                    last_hr_date=last_hr_date(glog),
                    last_game_date=last_game,
                    days_since_last_game=days_since,
                    is_absent=is_absent,
                ),
            )
        )

    # --- Authoritative verification from the transactions feed ---
    # Fetch each team's transactions once, then replay per player.
    tx_cache: dict[int, list[dict]] = {}
    for hitter, status in flagged:
        if hitter.team_id is None:
            continue
        if hitter.team_id not in tx_cache:
            tx_cache[hitter.team_id] = transactions.fetch_team_transactions(
                hitter.team_id, season
            )
        verdict = transactions.verify_status(
            tx_cache[hitter.team_id], hitter.player_id, as_of=as_of
        )
        status.verified_cause = verdict.cause
        status.il_type = verdict.il_type
        status.il_start = verdict.il_start
        status.injury_note = verdict.injury_note
        status.external_move = verdict.external_move
        status.move_date = verdict.move_date

    statuses = [s for _h, s in flagged]

    if with_news:
        for status in statuses:
            verdict_n = news.assess(status.name, team=status.team)
            status.news_cause = verdict_n.cause
            # Show the most relevant evidence: headlines that name the player,
            # most recent first; fall back to the raw feed if none match.
            surname = news._surname(status.name)
            relevant = [h for h in verdict_n.headlines if surname in h.title.lower()]
            relevant.sort(
                key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            status.news_headlines = (relevant or verdict_n.headlines)[:3]

    if with_matchups:
        # Only the bettable (UNCLEAR) players can actually play — skip the IL /
        # optioned crowd. One batter-hand batch + one schedule fetch per team.
        bettable = [(h, s) for h, s in flagged if s.verified_cause == "unverified"]
        hands = matchups.batter_hands([h.player_id for h, _ in bettable])
        td_cache: dict[int, matchups.TeamDay] = {}
        for hitter, status in bettable:
            if hitter.team_id is None:
                continue
            if hitter.team_id not in td_cache:
                td_cache[hitter.team_id] = matchups.team_status(
                    hitter.team_id, season, as_of=as_of
                )
            team_day = td_cache[hitter.team_id]
            status.plays_today = team_day.plays_today
            v = matchups.grade_for_batter(hands.get(hitter.player_id), team_day.matchup)
            status.matchup_label = v.label
            status.matchup_pitcher = v.opp_pitcher
            status.matchup_throws = v.opp_throws
            status.matchup_date = v.game_date
            status.matchup_ops = v.ops_allowed
            status.matchup_hr9 = v.hr9
            status.matchup_edge = v.edge
            status.matchup_detail = v.detail

    # Most concerning first for the CLI scan: verified injuries, then
    # absences, then long droughts. (The web page re-sorts for betting value,
    # surfacing favorable matchups — see app.slugger_service._betting_priority.)
    statuses.sort(
        key=lambda s: (
            s.verified_cause != "injury-verified",
            not s.is_absent,
            -(s.days_since_last_game or 0),
            -s.drought_games,
        )
    )
    return statuses


def status_to_row(s: SluggerStatus) -> dict:
    """Flatten a SluggerStatus to a CSV/JSON-friendly dict (drops headlines objs)."""
    row = asdict(s)
    row.pop("news_headlines", None)
    for key in ("last_hr_date", "last_game_date", "il_start", "move_date", "matchup_date"):
        val = getattr(s, key)
        row[key] = val.isoformat() if val else ""
    row["status_label"] = s.status_label
    row["active_today"] = s.active_today
    row["advisory"] = s.advisory
    row["top_headline"] = s.news_headlines[0].title if s.news_headlines else ""
    return row
