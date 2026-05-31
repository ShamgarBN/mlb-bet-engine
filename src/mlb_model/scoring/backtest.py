"""Hitter-prop backtest with strict as-of-date features (no lookahead).

Strategy:
  * One SQL query per season computes **cumulative** batter + pitcher
    aggregates using ``SUM() OVER (... ROWS BETWEEN UNBOUNDED PRECEDING
    AND 1 PRECEDING)``. That excludes the row's own game, so the
    features for game G use stats strictly from games before G.
  * Splits vs hand are attributed via the opposing-SP's throws (looked
    up from ``probable_pitchers``, just like the prod data layer).
  * Actual outcomes (1+ H, 1+ HR, TB>=2, 1+ K) come from the current
    row's ``batter_game_stats``.
  * Scoring re-uses the same :func:`mlb_model.scoring.hitter.score_matchup`
    the live app calls -- no separate "research model" to drift.

We intentionally backtest the **core algorithm** (current season only,
no prior-year blending, no Statcast quality stats). The point is to
measure the design's signal-to-noise on the same inputs the live app
has on opening day -- adding the blending later only improves things,
so the MVP backtest sets a conservative floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import math
import numpy as np
import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger
from mlb_model.scoring.hitter import (
    BatterInputs, PitcherInputs, score_matchup,
)

log = get_logger("scoring.backtest")


# ---------------------------------------------------------------------------
# As-of cumulative aggregates -- one SQL pass per season
# ---------------------------------------------------------------------------

_BATTER_ASOF_SQL = """
WITH bgs_dated AS (
    SELECT bgs.batter_id, bgs.game_pk, bgs.team_id,
           bgs.at_bats, bgs.plate_appearances, bgs.hits, bgs.doubles,
           bgs.triples, bgs.home_runs, bgs.walks, bgs.strikeouts, bgs.hbp,
           bgs.total_bases,
           g.game_date, g.season,
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
    WHERE g.status = 'Final' AND g.season = ?
)
SELECT
    batter_id, game_pk, game_date, opp_sp_throws,
    -- Cumulative AS-OF totals (excluding current row)
    SUM(at_bats)            OVER w AS ab,
    SUM(plate_appearances)  OVER w AS pa,
    SUM(hits)               OVER w AS h,
    SUM(walks)              OVER w AS bb,
    SUM(hbp)                OVER w AS hbp,
    SUM(total_bases)        OVER w AS tb,
    SUM(strikeouts)         OVER w AS k,
    SUM(home_runs)          OVER w AS hr,
    -- Splits vs LHP
    SUM(CASE WHEN opp_sp_throws='L' THEN at_bats             ELSE 0 END) OVER w AS ab_vL,
    SUM(CASE WHEN opp_sp_throws='L' THEN plate_appearances   ELSE 0 END) OVER w AS pa_vL,
    SUM(CASE WHEN opp_sp_throws='L' THEN hits                ELSE 0 END) OVER w AS h_vL,
    SUM(CASE WHEN opp_sp_throws='L' THEN total_bases         ELSE 0 END) OVER w AS tb_vL,
    SUM(CASE WHEN opp_sp_throws='L' THEN strikeouts          ELSE 0 END) OVER w AS k_vL,
    -- Splits vs RHP
    SUM(CASE WHEN opp_sp_throws='R' THEN at_bats             ELSE 0 END) OVER w AS ab_vR,
    SUM(CASE WHEN opp_sp_throws='R' THEN plate_appearances   ELSE 0 END) OVER w AS pa_vR,
    SUM(CASE WHEN opp_sp_throws='R' THEN hits                ELSE 0 END) OVER w AS h_vR,
    SUM(CASE WHEN opp_sp_throws='R' THEN total_bases         ELSE 0 END) OVER w AS tb_vR,
    SUM(CASE WHEN opp_sp_throws='R' THEN strikeouts          ELSE 0 END) OVER w AS k_vR,
    -- Current-game outcomes (for grading)
    hits AS y_h, home_runs AS y_hr, total_bases AS y_tb, strikeouts AS y_k,
    plate_appearances AS y_pa
FROM bgs_dated
WINDOW w AS (
    PARTITION BY batter_id, season ORDER BY game_date, game_pk
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
"""

_PITCHER_ASOF_SQL = """
WITH starts AS (
    SELECT pgs.pitcher_id, pgs.team_id AS pitcher_team, pgs.game_pk,
           pgs.innings_pitched, pgs.batters_faced,
           pgs.strikeouts, pgs.walks, pgs.home_runs, pgs.hits, pgs.earned_runs,
           g.game_date, g.season
    FROM pitcher_game_stats pgs
    JOIN games g USING (game_pk)
    WHERE g.status = 'Final' AND pgs.is_starter = TRUE AND g.season = ?
),
opp_batters AS (
    SELECT s.pitcher_id, s.game_pk, s.game_date, s.season,
           SUM(bgs.at_bats)            AS ab_g,
           SUM(bgs.plate_appearances)  AS pa_g,
           SUM(bgs.hits)               AS h_g,
           SUM(bgs.home_runs)          AS hr_g,
           SUM(bgs.strikeouts)         AS k_g,
           SUM(bgs.total_bases)        AS tb_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.at_bats           ELSE 0 END) AS ab_vL_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.plate_appearances ELSE 0 END) AS pa_vL_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.hits              ELSE 0 END) AS h_vL_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.home_runs         ELSE 0 END) AS hr_vL_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.strikeouts        ELSE 0 END) AS k_vL_g,
           SUM(CASE WHEN l.bats='L' THEN bgs.total_bases       ELSE 0 END) AS tb_vL_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.at_bats           ELSE 0 END) AS ab_vR_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.plate_appearances ELSE 0 END) AS pa_vR_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.hits              ELSE 0 END) AS h_vR_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.home_runs         ELSE 0 END) AS hr_vR_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.strikeouts        ELSE 0 END) AS k_vR_g,
           SUM(CASE WHEN l.bats='R' THEN bgs.total_bases       ELSE 0 END) AS tb_vR_g
    FROM starts s
    JOIN batter_game_stats bgs ON bgs.game_pk = s.game_pk AND bgs.team_id != s.pitcher_team
    LEFT JOIN lineups l ON l.game_pk = bgs.game_pk AND l.player_id = bgs.batter_id
    GROUP BY 1,2,3,4
),
joined AS (
    SELECT s.pitcher_id, s.game_pk, s.game_date, s.season,
           s.innings_pitched, s.batters_faced, s.strikeouts, s.walks, s.home_runs,
           o.ab_g, o.pa_g, o.h_g, o.hr_g, o.k_g, o.tb_g,
           o.ab_vL_g, o.pa_vL_g, o.h_vL_g, o.hr_vL_g, o.k_vL_g, o.tb_vL_g,
           o.ab_vR_g, o.pa_vR_g, o.h_vR_g, o.hr_vR_g, o.k_vR_g, o.tb_vR_g
    FROM starts s
    LEFT JOIN opp_batters o ON o.pitcher_id = s.pitcher_id AND o.game_pk = s.game_pk
)
SELECT
    pitcher_id, game_pk, game_date,
    SUM(innings_pitched) OVER w AS ip,
    SUM(batters_faced)   OVER w AS bf,
    SUM(strikeouts)      OVER w AS k,
    SUM(walks)           OVER w AS bb,
    SUM(home_runs)       OVER w AS hr_allowed,
    SUM(ab_vL_g)  OVER w AS ab_vL, SUM(pa_vL_g) OVER w AS pa_vL,
    SUM(h_vL_g)   OVER w AS h_vL,  SUM(hr_vL_g) OVER w AS hr_vL,
    SUM(k_vL_g)   OVER w AS k_vL,  SUM(tb_vL_g) OVER w AS tb_vL,
    SUM(ab_vR_g)  OVER w AS ab_vR, SUM(pa_vR_g) OVER w AS pa_vR,
    SUM(h_vR_g)   OVER w AS h_vR,  SUM(hr_vR_g) OVER w AS hr_vR,
    SUM(k_vR_g)   OVER w AS k_vR,  SUM(tb_vR_g) OVER w AS tb_vR
FROM joined
WINDOW w AS (
    PARTITION BY pitcher_id, season ORDER BY game_date, game_pk
    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
)
"""

# Map game -> opposing-SP. Same probable-pitchers logic, kept per-row so
# we can join one batter-game row to the correct SP-aggregate row.
_OPP_SP_SQL = """
WITH pp_one AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY game_pk, is_home ORDER BY pitcher_id) AS rn
    FROM probable_pitchers
)
SELECT g.game_pk, g.home_team_id, g.away_team_id,
       pp_h.pitcher_id AS home_sp_id, pp_h.pitcher_throws AS home_sp_throws,
       pp_a.pitcher_id AS away_sp_id, pp_a.pitcher_throws AS away_sp_throws,
       g.venue_name
FROM games g
LEFT JOIN pp_one pp_h ON pp_h.game_pk = g.game_pk AND pp_h.is_home = TRUE  AND pp_h.rn = 1
LEFT JOIN pp_one pp_a ON pp_a.game_pk = g.game_pk AND pp_a.is_home = FALSE AND pp_a.rn = 1
WHERE g.status = 'Final' AND g.season = ?
"""


def _team_to_opp_sp(row, opp_map: dict) -> tuple[int | None, str | None]:
    """Look up the opposing-team starter id + throws for a (game_pk, batter_team_id)."""
    info = opp_map.get(row["game_pk"])
    if info is None:
        return None, None
    if row["batter_team_id"] == info["home_team_id"]:
        return info["away_sp_id"], info["away_sp_throws"]
    return info["home_sp_id"], info["home_sp_throws"]


def _safe(num, den) -> float | None:
    try:
        n = float(num); d = float(den)
        if n != n or d != d or d <= 0:
            return None
        return n / d
    except (TypeError, ValueError):
        return None


def _val(row: dict, key: str, default=0):
    v = row.get(key, default)
    if v is None:
        return default
    try:
        if v != v:
            return default
    except TypeError:
        pass
    return v


def _batter_inputs(b_row, opp_throws):
    """Build BatterInputs from an as-of row + opposing SP hand."""
    ab = _val(b_row, "ab"); pa = _val(b_row, "pa")
    h = _val(b_row, "h"); bb = _val(b_row, "bb"); hbp = _val(b_row, "hbp")
    tb = _val(b_row, "tb"); k = _val(b_row, "k")
    season_avg = _safe(h, ab)
    season_obp = _safe(h + bb + hbp, ab + bb + hbp) if pa else None
    season_slg = _safe(tb, ab)
    season_iso = (season_slg - season_avg) if (season_avg is not None and season_slg is not None) else None
    season_k_pct = _safe(k, pa)
    if opp_throws in ("L", "R"):
        sf = "vL" if opp_throws == "L" else "vR"
        sab = _val(b_row, f"ab_{sf}"); spa = _val(b_row, f"pa_{sf}")
        sh = _val(b_row, f"h_{sf}"); stb = _val(b_row, f"tb_{sf}")
        sk = _val(b_row, f"k_{sf}")
    else:
        sab = spa = sh = stb = sk = 0
    split_avg = _safe(sh, sab)
    split_slg = _safe(stb, sab)
    split_iso = (split_slg - split_avg) if (split_avg is not None and split_slg is not None) else None
    split_k_pct = _safe(sk, spa)
    # NOTE: bats hand looked up below from lineups; passed in by caller
    return BatterInputs(
        bats=None,  # filled in by caller
        season_avg=season_avg, season_obp=season_obp, season_slg=season_slg,
        season_iso=season_iso, season_k_pct=season_k_pct, season_pa=int(pa),
        split_avg=split_avg, split_slg=split_slg, split_iso=split_iso,
        split_k_pct=split_k_pct, split_pa=int(spa),
    )


def _pitcher_inputs(p_row, throws):
    ip = float(_val(p_row, "ip"))
    bf = int(_val(p_row, "bf"))
    if bf == 0 and ip > 0:
        bf = int(round(ip * 4.3))
    bb = _val(p_row, "bb"); hra = _val(p_row, "hr_allowed"); k = _val(p_row, "k")
    era = None
    fip = ((13.0*hra + 3.0*bb - 2.0*k) / ip + 3.10) if ip > 0 else None
    k_pct = (k / bf) if bf > 0 else None
    bb_pct = (bb / bf) if bf > 0 else None
    hr9 = (hra * 9.0 / ip) if ip > 0 else None

    def _split(prefix):
        ab = _val(p_row, f"ab_{prefix}"); pa = _val(p_row, f"pa_{prefix}")
        h = _val(p_row, f"h_{prefix}"); hr = _val(p_row, f"hr_{prefix}")
        kk = _val(p_row, f"k_{prefix}"); tb = _val(p_row, f"tb_{prefix}")
        avg = _safe(h, ab); slg = _safe(tb, ab)
        obp_est = (avg + 0.070) if avg is not None else None
        ops = (obp_est + slg) if (obp_est is not None and slg is not None) else None
        kpct = _safe(kk, pa)
        hr_per_pa = _safe(hr, pa)
        hr9_h = (hr_per_pa * 38.0) if hr_per_pa is not None else None
        return ops, avg, hr9_h, kpct

    vL_ops, vL_avg, vL_hr9, vL_k = _split("vL")
    vR_ops, vR_avg, vR_hr9, vR_k = _split("vR")
    return PitcherInputs(
        throws=throws, era=era, fip=fip, k_pct=k_pct, bb_pct=bb_pct, hr9=hr9,
        batters_faced=bf,
        vs_lhb_ops=vL_ops, vs_rhb_ops=vR_ops, vs_lhb_avg=vL_avg, vs_rhb_avg=vR_avg,
        vs_lhb_hr9=vL_hr9, vs_rhb_hr9=vR_hr9, vs_lhb_k_pct=vL_k, vs_rhb_k_pct=vR_k,
    )


# ---------------------------------------------------------------------------
# Run + grade
# ---------------------------------------------------------------------------

@dataclass
class SeasonResult:
    season: int
    rows: pd.DataFrame  # one row per scored (batter, game) with score + outcome


def run_season(season: int) -> SeasonResult:
    """Score every batter-game in ``season`` against the opposing SP."""
    log.info("hitter_backtest.season.start", season=season)
    b = query(_BATTER_ASOF_SQL, (season,))
    p = query(_PITCHER_ASOF_SQL, (season,))
    opp = query(_OPP_SP_SQL, (season,))
    lineups = query(
        """
        SELECT game_pk, player_id AS batter_id, bats
        FROM lineups WHERE game_pk IN (SELECT game_pk FROM games WHERE season = ? AND status='Final')
        """,
        (season,),
    )
    log.info("hitter_backtest.season.loaded",
             season=season, batter_rows=len(b), pitcher_rows=len(p),
             games=len(opp), lineup_rows=len(lineups))

    # Build lookup: (game_pk, batter_id) -> bats; (game_pk) -> opp SP info
    bats_lookup = dict(zip(
        list(zip(lineups["game_pk"], lineups["batter_id"])),
        lineups["bats"],
    ))
    # Park factors are joined per game via the static lookup in the
    # scoring service. Doing it here keeps the backtest one-process.
    from mlb_model.scoring.service import park_factor_for

    opp_map = {
        int(row.game_pk): {
            "home_team_id": int(row.home_team_id), "away_team_id": int(row.away_team_id),
            "home_sp_id": int(row.home_sp_id) if pd.notna(row.home_sp_id) else None,
            "home_sp_throws": row.home_sp_throws,
            "away_sp_id": int(row.away_sp_id) if pd.notna(row.away_sp_id) else None,
            "away_sp_throws": row.away_sp_throws,
            "park_factor": park_factor_for(row.venue_name),
        }
        for row in opp.itertuples()
    }
    p_lookup: dict[tuple[int, int], dict] = {
        (int(r["pitcher_id"]), int(r["game_pk"])): r
        for r in p.to_dict(orient="records")
    }

    # Need batter_team_id to figure out which SP they faced -- pull from
    # the original batter_game_stats join.
    bteam = query(
        """
        SELECT bgs.batter_id, bgs.game_pk, bgs.team_id AS batter_team_id
        FROM batter_game_stats bgs JOIN games g USING (game_pk)
        WHERE g.status='Final' AND g.season = ?
        """,
        (season,),
    )
    team_lookup = {(int(r.batter_id), int(r.game_pk)): int(r.batter_team_id)
                   for r in bteam.itertuples()}

    out_rows = []
    skipped = {"no_opp": 0, "no_pitcher_row": 0, "no_pa": 0, "low_season_pa": 0}
    for r in b.itertuples():
        gpk = int(r.game_pk); bid = int(r.batter_id)
        opp_info = opp_map.get(gpk)
        team_id = team_lookup.get((bid, gpk))
        if opp_info is None or team_id is None:
            skipped["no_opp"] += 1
            continue
        if team_id == opp_info["home_team_id"]:
            sp_id, sp_throws = opp_info["away_sp_id"], opp_info["away_sp_throws"]
        else:
            sp_id, sp_throws = opp_info["home_sp_id"], opp_info["home_sp_throws"]
        if sp_id is None:
            skipped["no_opp"] += 1
            continue
        p_row = p_lookup.get((sp_id, gpk))
        if p_row is None:
            skipped["no_pitcher_row"] += 1
            continue
        b_row = r._asdict()
        bi = _batter_inputs(b_row, sp_throws)
        # Resolve bats (switch hitters bat opposite of pitcher hand)
        raw_bats = bats_lookup.get((gpk, bid))
        if raw_bats == "S" and sp_throws in ("L", "R"):
            bi.bats = "R" if sp_throws == "L" else "L"
        elif raw_bats in ("L", "R"):
            bi.bats = raw_bats
        pi = _pitcher_inputs(p_row, sp_throws)
        s = score_matchup(bi, pi, park_factor=opp_info["park_factor"])
        # Build outcomes
        y_pa = _val(b_row, "y_pa")
        if y_pa <= 0:
            skipped["no_pa"] += 1
            continue
        # Min-PA filter: skip rows where the batter had ≤ 5 season PA going
        # into the game. They're too noisy to score and were the source of
        # the bin-0 calibration inversion on the hit market.
        if int(_val(b_row, "pa")) < 5:
            skipped["low_season_pa"] = skipped.get("low_season_pa", 0) + 1
            continue
        out_rows.append({
            "game_pk": gpk, "batter_id": bid, "game_date": r.game_date,
            "hit_score": s.hit, "hr_score": s.hr,
            "tb_score": s.total_bases, "k_score": s.strikeout,
            "y_hit": 1 if _val(b_row, "y_h") > 0 else 0,
            "y_hr":  1 if _val(b_row, "y_hr") > 0 else 0,
            "y_tb":  1 if _val(b_row, "y_tb") >= 2 else 0,
            "y_k":   1 if _val(b_row, "y_k") > 0 else 0,
            "season_pa_to_date": int(_val(b_row, "pa")),
            "pa_in_game": int(y_pa),
            "bats": bi.bats, "throws": sp_throws,
        })
    log.info("hitter_backtest.season.scored", season=season,
             scored=len(out_rows), skipped=skipped)
    return SeasonResult(season=season, rows=pd.DataFrame(out_rows))


# ---------------------------------------------------------------------------
# Calibration + headline metrics
# ---------------------------------------------------------------------------

def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def score_to_prob(scores: np.ndarray, *, lg_rate: float, slope: float = 0.18) -> np.ndarray:
    """Map 0-10 scores to probabilities centered at ``lg_rate`` for score=5.

    Logistic: p = sigmoid(logit(lg_rate) + slope * (score - 5))
    """
    s = np.asarray(scores, dtype=float)
    z = _logit(lg_rate) + slope * (s - 5.0)
    return 1.0 / (1.0 + np.exp(-z))


def calibration_table(df: pd.DataFrame, score_col: str, y_col: str, n_bins: int = 10) -> pd.DataFrame:
    """Bin scores by quantile, report empirical hit rate per bin."""
    sub = df[[score_col, y_col]].dropna()
    if sub.empty:
        return pd.DataFrame()
    sub = sub.copy()
    sub["bin"] = pd.qcut(sub[score_col], q=n_bins, duplicates="drop", labels=False)
    g = sub.groupby("bin").agg(
        n=(y_col, "size"),
        score_mean=(score_col, "mean"),
        actual_rate=(y_col, "mean"),
    ).reset_index()
    return g


def headline_metrics(df: pd.DataFrame, score_col: str, y_col: str, lg_rate: float) -> dict:
    sub = df[[score_col, y_col]].dropna()
    if sub.empty:
        return {"n": 0}
    y = sub[y_col].to_numpy(dtype=float)
    p = score_to_prob(sub[score_col].to_numpy(), lg_rate=lg_rate)
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-12
    p_c = np.clip(p, eps, 1 - eps)
    log_loss = float(-np.mean(y * np.log(p_c) + (1 - y) * np.log(1 - p_c)))
    # Top-decile lift = avg outcome rate in top 10% of scores vs base rate
    thr = np.quantile(sub[score_col], 0.90)
    top = sub[sub[score_col] >= thr]
    base = float(sub[y_col].mean())
    top_rate = float(top[y_col].mean()) if len(top) else base
    lift = (top_rate - base) / base if base > 0 else 0.0
    return {
        "n": int(len(sub)),
        "base_rate": base,
        "brier": brier,
        "log_loss": log_loss,
        "top10_rate": top_rate,
        "top10_lift": lift,
    }


def run_backtest(seasons: Iterable[int]) -> dict:
    """Run + aggregate across multiple seasons. Returns a dict with the
    pooled DataFrame and per-season summaries.
    """
    season_results = [run_season(s) for s in seasons]
    pooled = pd.concat([r.rows for r in season_results], ignore_index=True)

    # League base rates (used by score_to_prob)
    lg = {
        "hit": float(pooled["y_hit"].mean()),
        "hr":  float(pooled["y_hr"].mean()),
        "tb":  float(pooled["y_tb"].mean()),
        "k":   float(pooled["y_k"].mean()),
    }
    summary = {}
    for market, score_col, y_col in [
        ("hit", "hit_score", "y_hit"),
        ("hr",  "hr_score",  "y_hr"),
        ("tb",  "tb_score",  "y_tb"),
        ("k",   "k_score",   "y_k"),
    ]:
        summary[market] = {
            "headline": headline_metrics(pooled, score_col, y_col, lg[market]),
            "calibration": calibration_table(pooled, score_col, y_col),
        }
    return {"pooled": pooled, "lg_rates": lg, "summary": summary,
            "per_season": {r.season: r.rows for r in season_results}}
