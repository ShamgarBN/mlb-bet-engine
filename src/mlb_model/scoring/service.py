"""Shape-for-template service for the Matchups page.

Pulls today's slate from the warehouse, joins live lineups + season
aggregates + park factors, computes per-batter scores, and returns a
plain list of dicts the Jinja template can iterate over.

Cached per-date in a parquet alongside the existing prediction cache
so repeated visits don't re-hit the MLB Stats API.
"""

from __future__ import annotations

import concurrent.futures as _cf
from dataclasses import asdict
from datetime import date as date_cls
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger
from mlb_model.scoring.data import (
    batter_inputs_for,
    batter_season_stats,
    fetch_game_lineups,
    pitcher_inputs_for,
    pitcher_season_stats,
)
from mlb_model.scoring.hitter import score_matchup

log = get_logger("scoring.service")

CACHE_DIR = settings.cache_dir / "matchups"


# ---------------------------------------------------------------------------
# Park factors (lifted from features.park; small static table is fine here)
# ---------------------------------------------------------------------------

_PARK_FACTORS: dict[str, float] = {
    # Hitter-friendly
    "Coors Field": 1.30, "Great American Ball Park": 1.10,
    "Fenway Park": 1.06, "Yankee Stadium": 1.05, "Wrigley Field": 1.04,
    "Globe Life Field": 1.03, "Citizens Bank Park": 1.03,
    "Chase Field": 1.02, "Camden Yards": 1.01, "Rogers Centre": 1.01,
    # Neutral
    "PNC Park": 1.00, "Comerica Park": 1.00, "Busch Stadium": 1.00,
    "American Family Field": 1.00, "Truist Park": 0.99, "loanDepot park": 0.99,
    "Citi Field": 0.99, "Nationals Park": 0.98, "Target Field": 0.98,
    # Pitcher-friendly
    "Minute Maid Park": 0.97, "Daikin Park": 0.97,
    "Angel Stadium": 0.96, "Kauffman Stadium": 0.96,
    "Progressive Field": 0.95, "Sutter Health Park": 0.95,
    "Dodger Stadium": 0.94, "T-Mobile Park": 0.93,
    "Tropicana Field": 0.93, "George M. Steinbrenner Field": 0.93,
    "Oracle Park": 0.92, "Petco Park": 0.91,
}


def park_factor_for(venue_name: str | None) -> float:
    if not venue_name:
        return 1.0
    return _PARK_FACTORS.get(venue_name, 1.0)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def _cache_path(target: date_cls) -> Path:
    # JSON (not parquet) because each row contains nested dicts of dicts of
    # lists -- parquet flattens those to numpy arrays which then break
    # Jinja's truthiness checks downstream.
    return CACHE_DIR / f"{target.isoformat()}.json"


def _load_cached(target: date_cls) -> list[dict] | None:
    import json
    path = _cache_path(target)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_cache(target: date_cls, rows: list[dict]) -> None:
    import json
    if not rows:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _cache_path(target).write_text(json.dumps(rows, default=str), encoding="utf-8")
    except OSError:
        log.exception("matchups.cache.save_failed", target=target)


def _games_for(target: date_cls) -> pd.DataFrame:
    """Return one row per game_pk on ``target`` with venue + status."""
    return query(
        """
        SELECT g.game_pk, g.game_date, g.scheduled_start,
               g.away_team_abbr, g.home_team_abbr,
               g.venue_name, g.status
        FROM games g
        WHERE g.game_date = ?
          AND g.status NOT IN ('Postponed','Cancelled','Suspended')
        ORDER BY g.scheduled_start, g.game_pk
        """,
        (target,),
    )


def _safe_round(x, n=2) -> float | None:
    if x is None:
        return None
    try:
        if x != x:  # NaN
            return None
        return round(float(x), n)
    except (TypeError, ValueError):
        return None


def get_matchups_for_date(
    target: date_cls,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Return one row per game with both lineups scored.

    Cached per-date; pass ``refresh=True`` to bust it. Each returned row
    looks like::

        {
          "game_pk": 822812,
          "game_date": "2026-05-31",
          "scheduled_start": "2026-05-31T17:07:00",
          "venue_name": "Rogers Centre",
          "park_factor": 1.01,
          "away_team_abbr": "MIA", "home_team_abbr": "TOR",
          "away": {
            "starter": {"name": ..., "throws": "R", "fip": 3.4, "k_pct": .25, ...},
            "batters": [
              {"order": 1, "name": "Xavier Edwards", "bats": "L", "position": "SS",
               "season_avg": .280, "season_pa": 50, "split_pa": 22,
               "hit": 5.3, "hr": 4.1, "tb": 4.8, "k": 5.6},
              ...
            ],
          },
          "home": {...},
        }
    """
    if not refresh:
        cached = _load_cached(target)
        if cached is not None:
            return cached

    games = _games_for(target)
    if games.empty:
        return []

    season = target.year
    batters_df = batter_season_stats(season)
    pitchers_df = pitcher_season_stats(season)

    # Pull live lineups in parallel — each is a single HTTP call.
    game_pks = [int(p) for p in games["game_pk"].tolist()]
    with _cf.ThreadPoolExecutor(max_workers=8) as pool:
        lineups_by_pk: dict[int, Any] = dict(zip(
            game_pks, pool.map(fetch_game_lineups, game_pks)
        ))

    out: list[dict[str, Any]] = []
    for _, g in games.iterrows():
        pk = int(g["game_pk"])
        lu = lineups_by_pk.get(pk)
        pf = park_factor_for(g["venue_name"])

        def _team_block(side, opp_starter_id, opp_starter_throws):
            if side is None:
                return {"starter": None, "batters": []}
            # Starter inputs (using the OTHER team's SP isn't what we want here;
            # this block is about THIS team's batters facing the OPPOSING SP.
            # So opp_starter_* come in pre-resolved by the caller.)
            opp_row = (
                pitchers_df.loc[opp_starter_id].to_dict()
                if opp_starter_id and opp_starter_id in pitchers_df.index
                else None
            )
            opp_pi = pitcher_inputs_for(opp_row, opp_starter_throws)

            # Also build inputs for THIS side's own starter (for the strip)
            own_starter_id = side.starter_id
            own_row = (
                pitchers_df.loc[own_starter_id].to_dict()
                if own_starter_id and own_starter_id in pitchers_df.index
                else None
            )
            own_pi = pitcher_inputs_for(own_row, side.starter_throws)

            batter_rows = []
            for bat in side.batters:
                brow = (
                    batters_df.loc[bat.player_id].to_dict()
                    if bat.player_id in batters_df.index
                    else None
                )
                bi = batter_inputs_for(brow, bat.bats, opp_starter_throws)
                s = score_matchup(bi, opp_pi, park_factor=pf)
                batter_rows.append({
                    "order": bat.batting_order,
                    "name": bat.full_name,
                    "bats_raw": bat.bats_raw,
                    "bats": bat.bats,
                    "is_switch": bat.is_switch,
                    "position": bat.position,
                    "season_avg": _safe_round(bi.season_avg, 3),
                    "season_obp": _safe_round(bi.season_obp, 3),
                    "season_slg": _safe_round(bi.season_slg, 3),
                    "season_ops": _safe_round(bi.season_ops, 3),
                    "season_iso": _safe_round(bi.season_iso, 3),
                    "season_k_pct": _safe_round(bi.season_k_pct, 3),
                    "season_pa": bi.season_pa,
                    "split_avg": _safe_round(bi.split_avg, 3),
                    "split_slg": _safe_round(bi.split_slg, 3),
                    "split_pa": bi.split_pa,
                    "hit": s.hit, "hr": s.hr,
                    "tb": s.total_bases, "k": s.strikeout,
                })

            return {
                "team_id": side.team_id, "team_abbr": side.team_abbr,
                "starter": {
                    "id": side.starter_id,
                    "name": side.starter_name,
                    "throws": side.starter_throws,
                    "fip": _safe_round(own_pi.fip, 2),
                    "era": _safe_round(own_pi.era, 2),
                    "k_pct": _safe_round(own_pi.k_pct, 3),
                    "bb_pct": _safe_round(own_pi.bb_pct, 3),
                    "hr9": _safe_round(own_pi.hr9, 2),
                    "batters_faced": int(own_pi.batters_faced or 0),
                    "vs_lhb_ops": _safe_round(own_pi.vs_lhb_ops, 3),
                    "vs_rhb_ops": _safe_round(own_pi.vs_rhb_ops, 3),
                } if side.starter_id else None,
                "batters": batter_rows,
            }

        row: dict[str, Any] = {
            "game_pk": pk,
            "game_date": target.isoformat(),
            "scheduled_start": g["scheduled_start"].isoformat() if pd.notna(g["scheduled_start"]) else None,
            "venue_name": g["venue_name"],
            "park_factor": pf,
            "status": g["status"],
            "away_team_abbr": g["away_team_abbr"],
            "home_team_abbr": g["home_team_abbr"],
        }
        if lu is not None:
            row["away"] = _team_block(lu.away, lu.home.starter_id, lu.home.starter_throws)
            row["home"] = _team_block(lu.home, lu.away.starter_id, lu.away.starter_throws)
        else:
            row["away"] = {"starter": None, "batters": []}
            row["home"] = {"starter": None, "batters": []}
        out.append(row)

    _save_cache(target, out)
    return out
