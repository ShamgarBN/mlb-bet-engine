"""Data layer for the Matchups page.

Hybrid approach (per the build plan):
  * Season aggregates  -> SQL over existing warehouse tables
                          (batter_game_stats, pitcher_game_stats, lineups).
                          Splits are joined to the opposing SP's handedness
                          via probable_pitchers + the batter's own ``bats``
                          field on the lineups table.
  * Today's lineups    -> live pull from the MLB Stats API per game_pk.
                          We don't bake these into the warehouse because
                          managers shuffle right up to first pitch.

Everything returns plain dataclasses / dicts -- no warehouse coupling
beyond the SQL helpers in this file. Scoring lives in :mod:`hitter`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as date_cls

import httpx
import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger
from mlb_model.scoring.hitter import BatterInputs, Hand, PitcherInputs

log = get_logger("scoring.data")


# ---------------------------------------------------------------------------
# Season batter aggregates (overall + split vs L/R)
# ---------------------------------------------------------------------------

_BATTER_SEASON_SQL = """
WITH per_game AS (
    SELECT
        bgs.batter_id,
        bgs.at_bats, bgs.plate_appearances, bgs.hits, bgs.doubles,
        bgs.triples, bgs.home_runs, bgs.walks, bgs.strikeouts, bgs.hbp,
        bgs.total_bases,
        -- The SP for the OTHER team is the one this batter faced.
        CASE WHEN bgs.team_id = g.home_team_id THEN pp_a.pitcher_throws
             ELSE pp_h.pitcher_throws END AS opp_sp_throws
    FROM batter_game_stats bgs
    JOIN games g USING (game_pk)
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY game_pk, is_home ORDER BY pitcher_id) AS rn
        FROM probable_pitchers
    ) pp_h ON pp_h.game_pk = g.game_pk AND pp_h.is_home = TRUE  AND pp_h.rn = 1
    LEFT JOIN (
        SELECT *, ROW_NUMBER() OVER (PARTITION BY game_pk, is_home ORDER BY pitcher_id) AS rn
        FROM probable_pitchers
    ) pp_a ON pp_a.game_pk = g.game_pk AND pp_a.is_home = FALSE AND pp_a.rn = 1
    WHERE EXTRACT(YEAR FROM g.game_date) = ?
      AND g.status = 'Final'
)
SELECT
    batter_id,
    -- Overall season
    SUM(at_bats) AS ab,
    SUM(plate_appearances) AS pa,
    SUM(hits) AS h,
    SUM(doubles) AS d2b,
    SUM(triples) AS d3b,
    SUM(home_runs) AS hr,
    SUM(walks) AS bb,
    SUM(strikeouts) AS k,
    SUM(hbp) AS hbp,
    SUM(total_bases) AS tb,
    -- Split vs LHP (rows where opposing SP was L)
    SUM(CASE WHEN opp_sp_throws = 'L' THEN at_bats ELSE 0 END) AS ab_vL,
    SUM(CASE WHEN opp_sp_throws = 'L' THEN plate_appearances ELSE 0 END) AS pa_vL,
    SUM(CASE WHEN opp_sp_throws = 'L' THEN hits ELSE 0 END) AS h_vL,
    SUM(CASE WHEN opp_sp_throws = 'L' THEN total_bases ELSE 0 END) AS tb_vL,
    SUM(CASE WHEN opp_sp_throws = 'L' THEN strikeouts ELSE 0 END) AS k_vL,
    SUM(CASE WHEN opp_sp_throws = 'L' THEN home_runs ELSE 0 END) AS hr_vL,
    -- Split vs RHP
    SUM(CASE WHEN opp_sp_throws = 'R' THEN at_bats ELSE 0 END) AS ab_vR,
    SUM(CASE WHEN opp_sp_throws = 'R' THEN plate_appearances ELSE 0 END) AS pa_vR,
    SUM(CASE WHEN opp_sp_throws = 'R' THEN hits ELSE 0 END) AS h_vR,
    SUM(CASE WHEN opp_sp_throws = 'R' THEN total_bases ELSE 0 END) AS tb_vR,
    SUM(CASE WHEN opp_sp_throws = 'R' THEN strikeouts ELSE 0 END) AS k_vR,
    SUM(CASE WHEN opp_sp_throws = 'R' THEN home_runs ELSE 0 END) AS hr_vR
FROM per_game
GROUP BY 1
"""


def batter_season_stats(season: int) -> pd.DataFrame:
    """One row per batter with season totals + splits vs L/R SPs.

    Returns a DataFrame indexed by ``batter_id`` containing both overall
    season stats and per-handedness splits. Rate stats (avg/obp/etc.) are
    *not* pre-computed -- :func:`batter_inputs_for` does that lazily so
    we control the league-prior shrinkage in one place.
    """
    df = query(_BATTER_SEASON_SQL, (season,))
    if df.empty:
        return df
    return df.set_index("batter_id")


# ---------------------------------------------------------------------------
# Season pitcher aggregates (overall + split allowed vs LHB / RHB)
# ---------------------------------------------------------------------------

_PITCHER_SEASON_SQL = """
WITH per_game AS (
    SELECT
        pgs.pitcher_id,
        pgs.innings_pitched, pgs.batters_faced, pgs.hits, pgs.runs,
        pgs.earned_runs, pgs.strikeouts, pgs.walks, pgs.home_runs
    FROM pitcher_game_stats pgs
    JOIN games g USING (game_pk)
    WHERE EXTRACT(YEAR FROM g.game_date) = ?
      AND g.status = 'Final'
      AND pgs.is_starter = TRUE
)
SELECT
    pitcher_id,
    SUM(innings_pitched) AS ip,
    SUM(batters_faced)   AS bf,
    SUM(hits)            AS h_allowed,
    SUM(earned_runs)     AS er,
    SUM(strikeouts)      AS k,
    SUM(walks)           AS bb,
    SUM(home_runs)       AS hr_allowed
FROM per_game GROUP BY 1
"""

# Splits: what each pitcher allowed broken down by the batter's hand.
# Joins pitcher_game_stats to batter_game_stats via game_pk so we can
# attribute each batter outcome to the pitcher who faced them. Approximate
# (it credits the starter for all opposing-team PAs in his appearance),
# but good enough for handedness-split rate estimation.
_PITCHER_SPLITS_SQL = """
WITH starts AS (
    SELECT pgs.pitcher_id, pgs.team_id AS pitcher_team, pgs.game_pk
    FROM pitcher_game_stats pgs
    JOIN games g USING (game_pk)
    WHERE EXTRACT(YEAR FROM g.game_date) = ?
      AND g.status = 'Final'
      AND pgs.is_starter = TRUE
),
opp_batters AS (
    -- Look up each PA's hand from the lineups table for that game.
    SELECT s.pitcher_id, bgs.at_bats, bgs.plate_appearances, bgs.hits,
           bgs.home_runs, bgs.strikeouts, bgs.total_bases,
           l.bats AS bat_hand
    FROM starts s
    JOIN batter_game_stats bgs
      ON bgs.game_pk = s.game_pk AND bgs.team_id != s.pitcher_team
    LEFT JOIN lineups l
      ON l.game_pk = bgs.game_pk AND l.player_id = bgs.batter_id
)
SELECT
    pitcher_id,
    SUM(CASE WHEN bat_hand = 'L' THEN at_bats        ELSE 0 END) AS ab_vL,
    SUM(CASE WHEN bat_hand = 'L' THEN plate_appearances ELSE 0 END) AS pa_vL,
    SUM(CASE WHEN bat_hand = 'L' THEN hits           ELSE 0 END) AS h_vL,
    SUM(CASE WHEN bat_hand = 'L' THEN home_runs      ELSE 0 END) AS hr_vL,
    SUM(CASE WHEN bat_hand = 'L' THEN strikeouts     ELSE 0 END) AS k_vL,
    SUM(CASE WHEN bat_hand = 'L' THEN total_bases    ELSE 0 END) AS tb_vL,
    SUM(CASE WHEN bat_hand = 'R' THEN at_bats        ELSE 0 END) AS ab_vR,
    SUM(CASE WHEN bat_hand = 'R' THEN plate_appearances ELSE 0 END) AS pa_vR,
    SUM(CASE WHEN bat_hand = 'R' THEN hits           ELSE 0 END) AS h_vR,
    SUM(CASE WHEN bat_hand = 'R' THEN home_runs      ELSE 0 END) AS hr_vR,
    SUM(CASE WHEN bat_hand = 'R' THEN strikeouts     ELSE 0 END) AS k_vR,
    SUM(CASE WHEN bat_hand = 'R' THEN total_bases    ELSE 0 END) AS tb_vR
FROM opp_batters
GROUP BY 1
"""


def pitcher_season_stats(season: int) -> pd.DataFrame:
    """Pitcher headline stats joined with splits vs L/R batters."""
    head = query(_PITCHER_SEASON_SQL, (season,))
    splits = query(_PITCHER_SPLITS_SQL, (season,))
    if head.empty:
        return head
    df = head.merge(splits, on="pitcher_id", how="left")
    return df.set_index("pitcher_id")


# ---------------------------------------------------------------------------
# Live lineup pull (MLB Stats API)
# ---------------------------------------------------------------------------

@dataclass
class LineupBatter:
    player_id: int
    full_name: str
    bats: Hand | None      # Effective hand vs today's SP (switch hitters resolved)
    bats_raw: str | None   # Raw code from MLB API: 'L' / 'R' / 'S' (switch)
    batting_order: int
    position: str | None

    @property
    def is_switch(self) -> bool:
        return self.bats_raw == "S"


@dataclass
class LineupTeam:
    team_id: int
    team_abbr: str
    starter_id: int | None
    starter_name: str | None
    starter_throws: Hand | None
    batters: list[LineupBatter]


@dataclass
class GameLineups:
    game_pk: int
    game_date: date_cls
    venue_name: str | None
    home: LineupTeam
    away: LineupTeam


_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game/{pk}/feed/live"


def _hand_from_dict(h: dict | None) -> Hand | None:
    code = (h or {}).get("code")
    return code if code in ("L", "R") else None


def _bat_code(h: dict | None) -> str | None:
    """Raw MLB Stats API hand code: 'L', 'R', 'S' (switch), or None."""
    code = (h or {}).get("code")
    return code if code in ("L", "R", "S") else None


def _effective_bat_hand(bat_code: str | None, sp_throws: Hand | None) -> Hand | None:
    """Resolve a switch hitter's effective hand vs today's SP.

    A switch hitter bats opposite to the pitcher (L vs RHP, R vs LHP).
    """
    if bat_code in ("L", "R"):
        return bat_code
    if bat_code == "S" and sp_throws in ("L", "R"):
        return "R" if sp_throws == "L" else "L"
    return None


def fetch_game_lineups(game_pk: int, *, timeout: float = 8.0) -> GameLineups | None:
    """Pull confirmed (or "as-of-now") lineups + probable SPs for one game.

    Returns ``None`` on any error -- the matchups page falls back gracefully
    to season aggregates when the lineup isn't posted yet.
    """
    try:
        with httpx.Client(timeout=timeout) as c:
            r = c.get(_FEED_URL.format(pk=game_pk))
            r.raise_for_status()
            payload = r.json()
    except Exception:  # noqa: BLE001 -- never let a single bad game kill the page
        log.exception("matchups.lineup.fetch_failed", game_pk=game_pk)
        return None

    game_data = payload.get("gameData") or {}
    teams_data = game_data.get("teams") or {}
    venue = (game_data.get("venue") or {}).get("name")
    game_date_str = (game_data.get("datetime") or {}).get("officialDate")
    try:
        game_date = date_cls.fromisoformat(game_date_str) if game_date_str else date_cls.today()
    except ValueError:
        game_date = date_cls.today()

    live = payload.get("liveData") or {}
    box = (live.get("boxscore") or {}).get("teams") or {}
    players = (game_data.get("players") or {})

    def _build_team(side: str) -> LineupTeam | None:
        td = teams_data.get(side) or {}
        bt = box.get(side) or {}
        team_id = td.get("id")
        if team_id is None:
            return None
        # Probable / actual starter
        sp = bt.get("pitchers") or []
        starter_id = sp[0] if sp else None
        if starter_id is None:
            # Try gameData.probablePitchers fallback
            pp = (game_data.get("probablePitchers") or {}).get(side) or {}
            starter_id = pp.get("id")
        starter_meta = players.get(f"ID{starter_id}") if starter_id else None
        starter_name = (starter_meta or {}).get("fullName")
        starter_throws = _hand_from_dict((starter_meta or {}).get("pitchHand"))

        # Opposing SP's hand drives switch-hitter resolution. Build the
        # other side first when we're on the second pass; for the first
        # pass we resolve once we have both sides (see post-loop fixup).
        order_ids = bt.get("battingOrder") or []
        batters: list[LineupBatter] = []
        for i, pid_str in enumerate(order_ids[:9], start=1):
            try:
                pid = int(pid_str)
            except (TypeError, ValueError):
                continue
            meta = players.get(f"ID{pid}") or {}
            raw = _bat_code(meta.get("batSide"))
            pos = ((meta.get("primaryPosition") or {}).get("abbreviation"))
            batters.append(LineupBatter(
                player_id=pid,
                full_name=meta.get("fullName") or f"#{pid}",
                bats=None,           # resolved below once both starters known
                bats_raw=raw,
                batting_order=i,
                position=pos,
            ))
        return LineupTeam(
            team_id=int(team_id),
            team_abbr=td.get("abbreviation") or td.get("teamCode") or "",
            starter_id=int(starter_id) if starter_id else None,
            starter_name=starter_name,
            starter_throws=starter_throws,
            batters=batters,
        )

    home = _build_team("home")
    away = _build_team("away")
    if home is None or away is None:
        return None
    # Resolve each batter's effective hand against the opposing SP.
    for b in home.batters:
        b.bats = _effective_bat_hand(b.bats_raw, away.starter_throws)
    for b in away.batters:
        b.bats = _effective_bat_hand(b.bats_raw, home.starter_throws)
    return GameLineups(
        game_pk=int(game_pk),
        game_date=game_date,
        venue_name=venue,
        home=home,
        away=away,
    )


# ---------------------------------------------------------------------------
# Convert warehouse rows -> dataclasses the scoring module consumes
# ---------------------------------------------------------------------------

def _safe(num, den) -> float | None:
    try:
        n = float(num); d = float(den)
        # NaN-aware: any NaN means we can't compute
        if n != n or d != d or d <= 0:
            return None
        return n / d
    except (TypeError, ValueError):
        return None


def _val(row: dict, key: str, default=0):
    """Read a row column, treating NaN as the default. Warehouse LEFT JOINs
    leave NaN where a side has no data; we never want to feed NaN into
    arithmetic or ``int()``.
    """
    if row is None:
        return default
    v = row.get(key, default)
    if v is None:
        return default
    try:
        if v != v:  # NaN
            return default
    except TypeError:
        pass
    return v


def batter_inputs_for(
    batter_row,
    bats: Hand | None,
    opposing_sp_throws: Hand | None,
) -> BatterInputs:
    """Project a warehouse batter row into the scoring dataclass."""
    if batter_row is None:
        return BatterInputs(bats=bats)
    ab = _val(batter_row, "ab")
    pa = _val(batter_row, "pa")
    h = _val(batter_row, "h")
    bb = _val(batter_row, "bb")
    hbp = _val(batter_row, "hbp")
    sf = 0  # not in our warehouse; small effect
    tb = _val(batter_row, "tb")
    k = _val(batter_row, "k")

    season_avg = _safe(h, ab)
    season_obp = _safe(h + bb + hbp, ab + bb + hbp + sf) if pa else None
    season_slg = _safe(tb, ab)
    season_ops = (season_obp + season_slg) if (season_obp is not None and season_slg is not None) else None
    season_iso = (season_slg - season_avg) if (season_avg is not None and season_slg is not None) else None
    season_k_pct = _safe(k, pa)
    season_bb_pct = _safe(bb, pa)

    # Split vs the opposing SP's hand
    if opposing_sp_throws in ("L", "R"):
        suffix = "vL" if opposing_sp_throws == "L" else "vR"
        sab = _val(batter_row, f"ab_{suffix}")
        spa = _val(batter_row, f"pa_{suffix}")
        sh  = _val(batter_row, f"h_{suffix}")
        stb = _val(batter_row, f"tb_{suffix}")
        sk  = _val(batter_row, f"k_{suffix}")
    else:
        sab = spa = sh = stb = sk = 0

    split_avg = _safe(sh, sab)
    split_slg = _safe(stb, sab)
    split_iso = (split_slg - split_avg) if (split_avg is not None and split_slg is not None) else None
    split_k_pct = _safe(sk, spa)

    return BatterInputs(
        bats=bats,
        season_avg=season_avg, season_obp=season_obp, season_slg=season_slg,
        season_ops=season_ops, season_iso=season_iso,
        season_k_pct=season_k_pct, season_bb_pct=season_bb_pct,
        season_pa=int(pa),
        split_avg=split_avg, split_slg=split_slg, split_iso=split_iso,
        split_k_pct=split_k_pct, split_pa=int(spa),
        # Statcast quality stats not in our warehouse yet (graceful None).
    )


def pitcher_inputs_for(pitcher_row, throws: Hand | None) -> PitcherInputs:
    """Project a warehouse pitcher row into the scoring dataclass."""
    if pitcher_row is None:
        return PitcherInputs(throws=throws)
    ip = float(_val(pitcher_row, "ip"))
    bf = int(_val(pitcher_row, "bf"))
    # ``pitcher_game_stats.batters_faced`` is null in the warehouse for
    # historical reasons (ingest never populated it). Fall back to the
    # league-average ratio of 4.3 PA/IP so rate stats still work.
    if bf == 0 and ip > 0:
        bf = int(round(ip * 4.3))
    er = _val(pitcher_row, "er")
    bb = _val(pitcher_row, "bb")
    hra = _val(pitcher_row, "hr_allowed")
    k = _val(pitcher_row, "k")

    era = (er * 9.0 / ip) if ip > 0 else None
    # FIP ≈ (13*HR + 3*(BB+HBP) - 2*K)/IP + ~3.10 constant (we don't have
    # HBP allowed in pitcher_game_stats; ignore the small bias).
    fip = ((13.0 * hra + 3.0 * bb - 2.0 * k) / ip + 3.10) if ip > 0 else None
    k_pct = (k / bf) if bf > 0 else None
    bb_pct = (bb / bf) if bf > 0 else None
    hr9 = (hra * 9.0 / ip) if ip > 0 else None

    def _split(prefix: str) -> dict[str, float | None]:
        ab = _val(pitcher_row, f"ab_{prefix}")
        pa = _val(pitcher_row, f"pa_{prefix}")
        h = _val(pitcher_row, f"h_{prefix}")
        hr = _val(pitcher_row, f"hr_{prefix}")
        kk = _val(pitcher_row, f"k_{prefix}")
        tb = _val(pitcher_row, f"tb_{prefix}")
        avg = _safe(h, ab)
        slg = _safe(tb, ab)
        # Without BB-vs-hand we can't compute true OBP per hand; use a
        # league-relative AVG-anchored approximation (OBP ≈ AVG + .070).
        obp_est = (avg + 0.070) if avg is not None else None
        ops = (obp_est + slg) if (obp_est is not None and slg is not None) else None
        kpct = _safe(kk, pa)
        # HR/9-vs-hand: needs IP-vs-hand which we don't track. Approximate
        # via HR rate per PA times 38 PA/9-IP (league avg).
        hr_per_pa = _safe(hr, pa)
        hr9_h = (hr_per_pa * 38.0) if hr_per_pa is not None else None
        return {"ops": ops, "avg": avg, "hr9": hr9_h, "k_pct": kpct}

    vL = _split("vL")
    vR = _split("vR")
    return PitcherInputs(
        throws=throws,
        era=era, fip=fip, k_pct=k_pct, bb_pct=bb_pct, hr9=hr9,
        batters_faced=bf,
        vs_lhb_ops=vL["ops"], vs_rhb_ops=vR["ops"],
        vs_lhb_avg=vL["avg"], vs_rhb_avg=vR["avg"],
        vs_lhb_hr9=vL["hr9"], vs_rhb_hr9=vR["hr9"],
        vs_lhb_k_pct=vL["k_pct"], vs_rhb_k_pct=vR["k_pct"],
    )
