"""Hitter-prop prediction journal + grading.

Lives next to the existing game-level prediction journal but uses its
own parquet because the schema is batter-level rather than game-level:

  * One row per (game_pk, batter_id, market) where ``market`` is one of
    ``prop_hit`` / ``prop_hr`` / ``prop_tb`` / ``prop_k``.
  * Records the raw 0-10 score, the calibrated probability, the
    league baseline, the resulting edge in percentage points, and a
    tier label derived from the edge.
  * Outcome columns (``is_settled``, ``actual_value``, ``y_win``) are
    filled in lazily by :func:`grade_props` once the game's
    ``batter_game_stats`` row lands.

Recording is idempotent: re-running on the same date just upserts the
existing rows.
"""

from __future__ import annotations

from datetime import datetime, date as date_cls
from pathlib import Path
from typing import Iterable

import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("journal.props")

JOURNAL_PATH = settings.data_dir / "journal" / "prop_predictions.parquet"

# League baselines from the 5-season backtest (logs/hitter_backtest_2021_2025_v2.parquet).
# Stable enough to hardcode -- updated when we re-calibrate.
LEAGUE_BASELINES = {
    "prop_hit": 0.563,
    "prop_hr":  0.112,
    "prop_tb":  0.321,
    "prop_k":   0.597,
}

# Tier thresholds in percentage points of probability above the league
# baseline. Designed to flag genuine edges for book-line comparison:
#   premium  -- big enough that even short prices look juicy
#   strong   -- comfortably above breakeven on -110ish props
#   edge     -- worth a small bet at fair price
#   lean     -- prefer over base rate
#   pass     -- skip
TIER_THRESHOLDS_PP = [
    ("premium",  8.0),
    ("strong",   5.0),
    ("edge",     3.0),
    ("lean",     1.0),
    ("pass",  -100.0),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def edge_to_tier(edge_pp: float | None) -> str:
    """Map an edge (probability − baseline, in pp) to a tier label."""
    if edge_pp is None or edge_pp != edge_pp:
        return "pass"
    for label, threshold in TIER_THRESHOLDS_PP:
        if edge_pp >= threshold:
            return label
    return "pass"


def _load() -> pd.DataFrame:
    if not JOURNAL_PATH.exists():
        return _empty()
    try:
        return pd.read_parquet(JOURNAL_PATH)
    except Exception:  # noqa: BLE001 -- corrupt journal must never crash
        log.exception("props.journal.read_failed", path=str(JOURNAL_PATH))
        return _empty()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "game_pk", "game_date", "batter_id", "batter_name", "team_abbr",
        "bats", "opposing_sp_name", "opposing_sp_throws",
        "market", "score", "model_prob", "league_baseline", "edge_pp", "tier",
        "recorded_at",
        # Outcome -- filled by grade_props()
        "is_settled", "pa_in_game", "actual_value", "y_win",
    ])


def _write(df: pd.DataFrame) -> None:
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(JOURNAL_PATH, index=False)


# ---------------------------------------------------------------------------
# Record: called from the Matchups service after scores are computed
# ---------------------------------------------------------------------------

def record_matchups(matchups: Iterable[dict]) -> int:
    """Append (or upsert) one row per (game, batter, market) for every
    scored matchup in ``matchups``. Returns the number of *new* rows
    written (existing pairs are silently updated in place).
    """
    new_rows: list[dict] = []
    now = pd.Timestamp.utcnow()
    for game in matchups:
        gpk = int(game.get("game_pk") or 0)
        if not gpk:
            continue
        try:
            gd = pd.to_datetime(game.get("game_date")).date()
        except Exception:  # noqa: BLE001
            continue
        for side_key in ("away", "home"):
            side = game.get(side_key) or {}
            opp_side = game.get("home" if side_key == "away" else "away") or {}
            opp_sp = (opp_side.get("starter") or {})
            team_abbr = side.get("team_abbr") or ""
            for b in (side.get("batters") or []):
                for market in ("prop_hit", "prop_hr", "prop_tb", "prop_k"):
                    score_key = market.replace("prop_", "")
                    prob_key = score_key + "_prob"
                    score = b.get(score_key)
                    prob = b.get(prob_key)
                    if score is None or prob is None:
                        continue
                    baseline = LEAGUE_BASELINES[market]
                    edge_pp = (float(prob) - baseline) * 100.0
                    new_rows.append({
                        "game_pk": gpk, "game_date": gd,
                        "batter_id": int(b.get("player_id") or b.get("batter_id") or 0),
                        "batter_name": b.get("name") or "",
                        "team_abbr": team_abbr,
                        "bats": b.get("bats") or "",
                        "opposing_sp_name": opp_sp.get("name") or "",
                        "opposing_sp_throws": opp_sp.get("throws") or "",
                        "market": market,
                        "score": float(score), "model_prob": float(prob),
                        "league_baseline": float(baseline),
                        "edge_pp": float(edge_pp),
                        "tier": edge_to_tier(edge_pp),
                        "recorded_at": now,
                        "is_settled": False, "pa_in_game": None,
                        "actual_value": None, "y_win": None,
                    })

    if not new_rows:
        return 0
    new_df = pd.DataFrame(new_rows)
    # Backfill batter_id from the lineup probe if it came in as 0
    new_df = new_df[new_df["batter_id"] != 0]
    if new_df.empty:
        return 0

    existing = _load()
    key = ["game_pk", "batter_id", "market"]
    if existing.empty:
        merged = new_df
        added = len(new_df)
    else:
        # Upsert: drop existing rows that match the new key set, then concat.
        existing_keys = set(map(tuple, existing[key].itertuples(index=False, name=None)))
        new_keys = set(map(tuple, new_df[key].itertuples(index=False, name=None)))
        existing = existing[
            ~existing.set_index(key).index.isin(new_keys)
        ].reset_index(drop=True)
        merged = pd.concat([existing, new_df], ignore_index=True)
        added = len(new_keys - existing_keys)
    _write(merged)
    log.info("props.journal.recorded", rows_in=len(new_rows),
             added=added, total=len(merged))
    return added


# ---------------------------------------------------------------------------
# Grade: join unsettled rows to actual outcomes
# ---------------------------------------------------------------------------

def grade_props() -> int:
    """Settle any journal rows whose game has finalized.

    Joins to ``batter_game_stats`` for the actual hit/HR/TB/K and writes
    ``actual_value`` + ``y_win`` per market. Returns the number of rows
    newly settled.
    """
    from mlb_model.data.warehouse import query

    df = _load()
    if df.empty:
        return 0
    unsettled = df[~df["is_settled"].fillna(False)]
    if unsettled.empty:
        return 0

    pks = list(set(int(p) for p in unsettled["game_pk"].tolist()))
    if not pks:
        return 0
    placeholders = ",".join("?" for _ in pks)
    outcomes = query(
        f"""
        SELECT bgs.game_pk, bgs.batter_id,
               bgs.plate_appearances AS pa,
               bgs.hits, bgs.home_runs, bgs.total_bases, bgs.strikeouts
        FROM batter_game_stats bgs
        JOIN games g USING (game_pk)
        WHERE g.status = 'Final' AND bgs.game_pk IN ({placeholders})
        """,
        tuple(pks),
    )
    if outcomes.empty:
        return 0
    # Aggregate in case of pinch-hit substitutions (sum across the batter's
    # appearances). The journal row reflects one matchup decision per
    # batter -- if they were lifted before the SP came out, they still
    # get credit for what they did vs that SP and any subsequent PAs.
    outcomes = outcomes.groupby(["game_pk", "batter_id"]).sum().reset_index()

    o_map = {
        (int(r.game_pk), int(r.batter_id)): r
        for r in outcomes.itertuples()
    }

    settled = 0
    for idx, row in unsettled.iterrows():
        key = (int(row["game_pk"]), int(row["batter_id"]))
        out = o_map.get(key)
        if out is None:
            continue
        pa = int(out.pa or 0)
        market = row["market"]
        if market == "prop_hit":
            actual = int(out.hits or 0); win = actual > 0
        elif market == "prop_hr":
            actual = int(out.home_runs or 0); win = actual > 0
        elif market == "prop_tb":
            actual = int(out.total_bases or 0); win = actual >= 2
        elif market == "prop_k":
            actual = int(out.strikeouts or 0); win = actual > 0
        else:
            continue
        df.at[idx, "is_settled"] = True
        df.at[idx, "pa_in_game"] = pa
        df.at[idx, "actual_value"] = actual
        df.at[idx, "y_win"] = int(bool(win))
        settled += 1
    if settled:
        _write(df)
        log.info("props.journal.graded", settled=settled, total=len(df))
    return settled


# ---------------------------------------------------------------------------
# Read: per-market × per-tier summary for the UI
# ---------------------------------------------------------------------------

TIER_ORDER = ["premium", "strong", "edge", "lean", "pass"]
MARKET_LABELS = {
    "prop_hit": "Hit (1+ H)",
    "prop_hr":  "Home run",
    "prop_tb":  "Total bases (2+)",
    "prop_k":   "Strikeout (1+ K)",
}


# ---------------------------------------------------------------------------
# Backfill from the as-of backtest output
# ---------------------------------------------------------------------------

def backfill_from_backtest(backtest_path: str | Path) -> int:
    """Bulk-import settled rows from a backtest parquet.

    The backtest writes one row per (game, batter) with score columns +
    binary outcomes. We apply the saved isotonic calibrators to recover
    a probability, derive an edge + tier, and append-or-merge into the
    main prop journal.

    Returns the number of new rows added.
    """
    import joblib
    p = Path(backtest_path)
    if not p.exists():
        log.warning("props.backfill.missing", path=str(p))
        return 0
    bt = pd.read_parquet(p)
    if bt.empty:
        return 0

    # Load calibrators once
    cals: dict[str, object] = {}
    for short in ("hit", "hr", "tb", "k"):
        cal_path = settings.model_dir / f"hitter_calibrator_{short}.joblib"
        if cal_path.exists():
            try:
                cals[short] = joblib.load(cal_path)["model"]
            except Exception:  # noqa: BLE001
                log.exception("props.backfill.calibrator_load", path=str(cal_path))

    out_rows: list[dict] = []
    now = pd.Timestamp.utcnow()
    market_map = [
        ("prop_hit", "hit_score", "y_hit", "hit"),
        ("prop_hr",  "hr_score",  "y_hr",  "hr"),
        ("prop_tb",  "tb_score",  "y_tb",  "tb"),
        ("prop_k",   "k_score",   "y_k",   "k"),
    ]
    for market, sc_col, y_col, cal_key in market_map:
        if sc_col not in bt.columns or y_col not in bt.columns:
            continue
        sub = bt[[
            "game_pk", "batter_id", "game_date", sc_col, y_col, "bats", "throws",
        ]].dropna(subset=[sc_col, y_col])
        if sub.empty:
            continue
        scores = sub[sc_col].to_numpy(dtype=float)
        cal = cals.get(cal_key)
        if cal is not None:
            probs = cal.predict(scores)
        else:
            # No calibrator -> fall back to a flat baseline so tier
            # derivation still works (just always "pass").
            probs = [LEAGUE_BASELINES[market]] * len(scores)
        baseline = LEAGUE_BASELINES[market]
        for sc, pr, y, (_, row) in zip(scores, probs, sub[y_col].to_numpy(), sub.iterrows()):
            edge_pp = (float(pr) - baseline) * 100.0
            out_rows.append({
                "game_pk": int(row["game_pk"]),
                "game_date": pd.to_datetime(row["game_date"]).date(),
                "batter_id": int(row["batter_id"]),
                "batter_name": "",  # backtest doesn't carry name
                "team_abbr": "",
                "bats": row.get("bats") or "",
                "opposing_sp_name": "",
                "opposing_sp_throws": row.get("throws") or "",
                "market": market,
                "score": float(sc),
                "model_prob": float(pr),
                "league_baseline": baseline,
                "edge_pp": float(edge_pp),
                "tier": edge_to_tier(edge_pp),
                "recorded_at": now,
                "is_settled": True,
                "pa_in_game": None,
                "actual_value": None,
                "y_win": int(y),
            })

    if not out_rows:
        return 0
    new_df = pd.DataFrame(out_rows)
    existing = _load()
    key = ["game_pk", "batter_id", "market"]
    if existing.empty:
        merged = new_df
        added = len(new_df)
    else:
        new_keys = set(map(tuple, new_df[key].itertuples(index=False, name=None)))
        existing = existing[~existing.set_index(key).index.isin(new_keys)].reset_index(drop=True)
        merged = pd.concat([existing, new_df], ignore_index=True)
        added = len(new_keys) - (len(merged) - len(existing) - len(new_df) + len(new_df))
        added = max(0, added) or len(new_df)
    _write(merged)
    log.info("props.backfill.done", path=str(p), rows=len(new_df), added=added)
    return added


def tier_breakdown(season: int | None = None) -> dict[str, list[dict]]:
    """Per-market tier table for the UI.

    Returns {market_label: [{tier, n, settled, wins, win_rate, baseline,
    edge_pp_mean}, ...]} sorted by tier importance.
    """
    df = _load()
    if df.empty:
        return {}
    if season is not None:
        df = df[pd.to_datetime(df["game_date"]).dt.year == int(season)]
    if df.empty:
        return {}

    out: dict[str, list[dict]] = {}
    for market, label in MARKET_LABELS.items():
        sub = df[df["market"] == market]
        if sub.empty:
            continue
        rows: list[dict] = []
        for tier in TIER_ORDER:
            t = sub[sub["tier"] == tier]
            settled = t[t["is_settled"] == True]  # noqa: E712 -- parquet bool
            wins = int((settled["y_win"] == 1).sum()) if not settled.empty else 0
            n_settled = int(len(settled))
            win_rate = (wins / n_settled) if n_settled > 0 else None
            rows.append({
                "tier": tier, "n": int(len(t)),
                "settled": n_settled, "wins": wins,
                "win_rate": win_rate,
                "edge_pp_mean": float(t["edge_pp"].mean()) if len(t) else None,
                "baseline": LEAGUE_BASELINES[market],
            })
        out[label] = rows
    return out
