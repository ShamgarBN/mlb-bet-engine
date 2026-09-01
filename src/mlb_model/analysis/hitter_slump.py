"""Track .300+ hitters who have gone cold — and explain why.

The sibling of :mod:`mlb_model.analysis.slugger_slump`. Where that module
hunts elite *power* bats in a HR drought, this one hunts elite *contact* bats
— qualified .300 hitters — who have gone hitless for several straight games.
A .300 hitter riding an 0-for-3-games skid is due for regression *if* nothing
is wrong; the same machinery that grades the slugger report tells the two
apart (injury / benching / roster move vs a plain cold streak) and, for the
still-bettable bats, grades the next pitching matchup and whether they play
today.

Pipeline
--------
1. :func:`fetch_season_hitting` — every hitter's season H/AB/PA/G totals live
   from the MLB Stats API (deduping traded players); batting average derived.
2. :func:`threshold_breakdown` — the share of qualified hitters batting at/above
   the average threshold (the ".300 club").
3. :func:`fetch_gamelog` / :func:`hit_drought` — per-player game log and the
   trailing count of consecutive *appeared* games with no hit.
4. :func:`find_slumping_hitters` — assemble the flagged list, attaching an
   absence signal, a transactions-verified cause, live news, and a hit-oriented
   next-game matchup grade.

The absence, transactions, news and schedule layers are shared with the
slugger report; only the "elite bat" definition (average, not HR) and the
drought stat (hits, not homers) differ.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

import statsapi

from mlb_model.analysis import matchups, news, transactions
from mlb_model.analysis.slugger_slump import absence_verdict
from mlb_model.logging import get_logger

log = get_logger("analysis.hitter_slump")

DEFAULT_MIN_DROUGHT = 3
# The season-average benchmark for an "elite contact bat". Fixed, unlike the
# slugger's moving HR bar — .300 is a stable, well-understood standard, so we
# don't chase a moving percentile.
DEFAULT_AVG_THRESHOLD = 0.300
# ".300 hitter" implies a regular. Same fixed PA floor the slugger report uses
# for its "qualified" denominator — robust to mid-run schedule drift.
QUALIFIED_PA_FLOOR = 200


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class HitterSeason:
    player_id: int
    name: str
    team: str | None
    team_id: int | None
    hits: int
    at_bats: int
    plate_appearances: int
    games: int

    @property
    def avg(self) -> float:
        return self.hits / self.at_bats if self.at_bats else 0.0


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
    threshold: float
    as_of: date
    shares: dict[str, DenomShare]

    @property
    def n_at_threshold(self) -> int:
        return next(iter(self.shares.values())).n_at_threshold if self.shares else 0


@dataclass(slots=True)
class HitterStatus:
    player_id: int
    name: str
    team: str | None
    batting_average: float
    hits: int
    at_bats: int
    games: int
    drought_games: int
    last_hit_date: date | None
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
    matchup_edge: float | None = None
    matchup_detail: str = ""
    # Whether the player's team has a not-yet-started game on the selected day.
    plays_today: bool = False

    @property
    def active_today(self) -> bool:
        """The primary target: a bettable cold bat playing today.

        A .300 hitter in a hit drought, cause UNCLEAR (not on the IL /
        optioned), team has a pre-game game today, and the player is actually
        in the mix (not sitting out / day-to-day)."""
        return (
            self.verified_cause == "unverified"
            and self.plays_today
            and not self.is_absent
        )

    @property
    def status_label(self) -> str:
        if self.verified_cause == "injury-verified":
            return "INJURY (verified)"
        if self.verified_cause == "external-verified":
            return "EXTERNAL (verified)"
        return "UNCLEAR"

    @property
    def advisory(self) -> str:
        if self.verified_cause == "injury-verified":
            since = f" since {self.il_start.isoformat()}" if self.il_start else ""
            note = f" — {self.injury_note}" if self.injury_note else ""
            il = self.il_type or "IL"
            return f"INJURY (verified) — on the {il} IL{since}{note}"
        if self.verified_cause == "external-verified":
            when = f" on {self.move_date.isoformat()}" if self.move_date else ""
            return f"EXTERNAL (verified) — {self.external_move}{when}"
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

    Traded players appear once per team in the splits; we sum their lines so a
    player's average reflects the whole season, not one stint.
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
                hits=int(st.get("hits", 0) or 0),
                at_bats=int(st.get("atBats", 0) or 0),
                plate_appearances=int(st.get("plateAppearances", 0) or 0),
                games=int(st.get("gamesPlayed", 0) or 0),
            )
        else:
            cur.hits += int(st.get("hits", 0) or 0)
            cur.at_bats += int(st.get("atBats", 0) or 0)
            cur.plate_appearances += int(st.get("plateAppearances", 0) or 0)
            cur.games += int(st.get("gamesPlayed", 0) or 0)
            if team:
                cur.team = team
                cur.team_id = team_id
    return list(agg.values())


# --------------------------------------------------------------------------- #
# 2. Threshold share (the ".300 club")
# --------------------------------------------------------------------------- #
def threshold_breakdown(
    hitters: list[HitterSeason],
    *,
    season: int,
    threshold: float = DEFAULT_AVG_THRESHOLD,
    pa_floor: int = QUALIFIED_PA_FLOOR,
    as_of: date | None = None,
) -> ThresholdBreakdown:
    """Share of qualified hitters batting at/above ``threshold``.

    Batting average is only meaningful for a regular, so — unlike the slugger
    report's three denominators — we report a single one: qualified hitters.
    """
    as_of = as_of or date.today()
    qualified = [h for h in hitters if h.plate_appearances >= pa_floor]
    at_thr = [h for h in qualified if h.avg >= threshold]
    shares = {
        "qualified": DenomShare(
            f"Regular hitters (≥{pa_floor} PA)", len(qualified), len(at_thr)
        ),
    }
    return ThresholdBreakdown(season=season, threshold=threshold, as_of=as_of, shares=shares)


# --------------------------------------------------------------------------- #
# 3. Per-player game log + drought
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class GameLogEntry:
    game_date: date
    hits: int


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
        hits = int(s.get("stat", {}).get("hits", 0) or 0)
        out.append(GameLogEntry(game_date=d, hits=hits))
    out.sort(key=lambda e: e.game_date)
    return out


def hit_drought(gamelog: list[GameLogEntry]) -> int:
    """Trailing consecutive *appeared* games with zero hits."""
    n = 0
    for entry in reversed(gamelog):
        if entry.hits > 0:
            break
        n += 1
    return n


def last_hit_date(gamelog: list[GameLogEntry]) -> date | None:
    for entry in reversed(gamelog):
        if entry.hits > 0:
            return entry.game_date
    return None


# --------------------------------------------------------------------------- #
# 4. Assemble the slumping-hitter report
# --------------------------------------------------------------------------- #
def find_slumping_hitters(
    season: int,
    *,
    threshold: float = DEFAULT_AVG_THRESHOLD,
    min_drought: int = DEFAULT_MIN_DROUGHT,
    pa_floor: int = QUALIFIED_PA_FLOOR,
    as_of: date | None = None,
    absence_days: int = 4,
    absence_games: int = 3,
    with_news: bool = True,
    with_matchups: bool = True,
    hitters: list[HitterSeason] | None = None,
) -> list[HitterStatus]:
    """Find qualified .300+ hitters in a hit drought of ``min_drought``+ games.

    A player is also flagged (regardless of drought length) when absent: idle
    ``absence_days``+ calendar days while his team played ``absence_games``+
    games without him — usually an IL stint or benching.
    """
    as_of = as_of or date.today()
    hitters = hitters if hitters is not None else fetch_season_hitting(season)
    elite = [
        h for h in hitters if h.plate_appearances >= pa_floor and h.avg >= threshold
    ]
    log.info("hitter.universe", n=len(elite), threshold=threshold)

    # Team schedules for the absence check, fetched lazily and cached per team.
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

    flagged: list[tuple[HitterSeason, HitterStatus]] = []
    for h in elite:
        glog = fetch_gamelog(h.player_id, season)
        drought = hit_drought(glog)
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
                HitterStatus(
                    player_id=h.player_id,
                    name=h.name,
                    team=h.team,
                    batting_average=round(h.avg, 3),
                    hits=h.hits,
                    at_bats=h.at_bats,
                    games=h.games,
                    drought_games=drought,
                    last_hit_date=last_hit_date(glog),
                    last_game_date=last_game,
                    days_since_last_game=days_since,
                    is_absent=is_absent,
                ),
            )
        )

    # --- Authoritative verification from the transactions feed ---
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
            surname = news._surname(status.name)
            relevant = [h for h in verdict_n.headlines if surname in h.title.lower()]
            relevant.sort(
                key=lambda h: h.published or datetime.min.replace(tzinfo=timezone.utc),
                reverse=True,
            )
            status.news_headlines = (relevant or verdict_n.headlines)[:3]

    if with_matchups:
        # Only the bettable (UNCLEAR) players can actually play — skip the IL /
        # optioned crowd. Grade the matchup for a *hit* bounce-back (power=False).
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
            v = matchups.grade_for_batter(
                hands.get(hitter.player_id), team_day.matchup, power=False
            )
            status.matchup_label = v.label
            status.matchup_pitcher = v.opp_pitcher
            status.matchup_throws = v.opp_throws
            status.matchup_date = v.game_date
            status.matchup_ops = v.ops_allowed
            status.matchup_edge = v.edge
            status.matchup_detail = v.detail

    # Most concerning first for the CLI scan: verified injuries, then absences,
    # then long droughts. (The web page re-sorts for betting value.)
    statuses.sort(
        key=lambda s: (
            s.verified_cause != "injury-verified",
            not s.is_absent,
            -(s.days_since_last_game or 0),
            -s.drought_games,
        )
    )
    return statuses


def status_to_row(s: HitterStatus) -> dict:
    """Flatten a HitterStatus to a CSV/JSON-friendly dict (drops headlines objs)."""
    row = asdict(s)
    row.pop("news_headlines", None)
    for key in ("last_hit_date", "last_game_date", "il_start", "move_date", "matchup_date"):
        val = getattr(s, key)
        row[key] = val.isoformat() if val else ""
    row["status_label"] = s.status_label
    row["active_today"] = s.active_today
    row["advisory"] = s.advisory
    row["top_headline"] = s.news_headlines[0].title if s.news_headlines else ""
    return row
