"""Append-only recorder for the model's predictions.

Every time ``predict_for_date`` runs we capture three rows per game (one
per market: moneyline / runline / total) into ``data/journal/predictions.parquet``.

Design choices:

* **Append-only.** Rows are never updated. If you re-predict the same
  day with a refined model, a NEW snapshot row is added with a fresh
  ``recorded_at`` timestamp. When grading we use the latest snapshot
  before the game started -- the "what the model thought going into
  the game" prediction.
* **Idempotent at the timestamp level.** Calling ``record_predictions``
  twice within the same second collapses to one row (deduped by
  ``(game_pk, market, pick, recorded_at)``).
* **No PII / no secrets** -- this is just probabilities and team codes.
* **Schema-stable.** The columns here become the contract for the
  ``/season`` page and the end-of-season report. Adding new columns is
  fine; renaming/removing requires a migration.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, UTC
from pathlib import Path
from typing import Iterable

import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("journal.record")

JOURNAL_PATH: Path = settings.data_dir / "journal" / "predictions.parquet"


# These columns must stay stable; downstream code (metrics, /season
# page, end-of-season report) joins on these names.
JOURNAL_COLUMNS = [
    "recorded_at",            # ISO timestamp -- when did the model produce this row?
    "game_pk",
    "game_date",
    "season",
    "away_team_abbr",
    "home_team_abbr",
    "market",                 # 'moneyline' | 'runline' | 'total'
    "pick",                   # 'HOME' | 'AWAY' | 'OVER' | 'UNDER'
    "pick_long",
    "model_prob",
    "market_prob",            # may be NaN
    "edge_pp",                # model_prob - market_prob (percentage points), NaN if no market
    "total_line",             # the line we were predicting against (NaN if not a totals row)
    "total_line_source",      # 'market' | 'baseline'
    "confidence",             # |p - 0.5| * 2  in [0, 1]
    "tier",                   # confidence tier label
]


def _empty_journal() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in JOURNAL_COLUMNS})


def _load_or_empty() -> pd.DataFrame:
    if not JOURNAL_PATH.exists():
        return _empty_journal()
    try:
        return pd.read_parquet(JOURNAL_PATH)
    except Exception:  # noqa: BLE001 -- corrupt journal shouldn't kill predictions
        log.exception("journal.read.failed", path=str(JOURNAL_PATH))
        return _empty_journal()


def _normalize_new_rows(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df

    # Ensure every required column exists; missing → NaN.
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    # Cast game_date to date_cls so the on-disk type stays consistent.
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    df["recorded_at"] = pd.to_datetime(df["recorded_at"], errors="coerce")
    # Derive season from game_date when not provided.
    if df["season"].isna().any():
        derived = pd.to_datetime(df["game_date"]).dt.year
        df["season"] = df["season"].fillna(derived)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")

    return df[JOURNAL_COLUMNS]


def record_predictions(rows: Iterable[dict]) -> int:
    """Append ``rows`` (one dict per game/market) to the journal.

    Each ``row`` MUST include at minimum ``game_pk``, ``game_date``,
    ``market``, ``pick``, ``pick_long``, and ``model_prob``. Other
    fields are optional and default to NaN.

    Returns the number of rows added (post-dedup).
    """
    new_df = _normalize_new_rows(rows)
    if new_df.empty:
        return 0

    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_or_empty()
    if existing.empty:
        out = new_df
    else:
        # Align dtypes before concat -- empty DataFrames upcast everything
        # to ``object`` which then fights pandas on read-back.
        for col in JOURNAL_COLUMNS:
            if col not in existing.columns:
                existing[col] = pd.NA
        out = pd.concat([existing, new_df], ignore_index=True)
        # Drop exact duplicates (same prediction recorded twice in the same second).
        out = out.drop_duplicates(
            subset=["game_pk", "market", "pick", "recorded_at"],
            keep="last",
        )

    out.to_parquet(JOURNAL_PATH, index=False)
    n_added = int(len(out) - len(existing))
    log.info(
        "journal.recorded",
        rows_added=n_added,
        total_rows=int(len(out)),
        path=str(JOURNAL_PATH),
    )
    return n_added


def record_predictions_from_df(predictions: pd.DataFrame) -> int:
    """Convenience wrapper that flattens a ``predict_for_date`` output.

    The DataFrame produced by :func:`mlb_model.predict.daily.predict_for_date`
    has one row per game with probabilities for all three markets in
    parallel columns (``p_home_win``, ``p_home_runline_cover``,
    ``p_total_over``). This helper turns that wide DataFrame into the
    three-rows-per-game format the journal stores.
    """
    if predictions is None or predictions.empty:
        return 0

    now = datetime.now(UTC).isoformat(timespec="seconds")
    rows: list[dict] = []

    for _, r in predictions.iterrows():
        # We rely on the same field names the dashboard uses; if the
        # predict output ever evolves, only this method needs updating.
        game_pk = int(r["game_pk"])
        game_date = r["game_date"]
        away = str(r["away_team_abbr"])
        home = str(r["home_team_abbr"])
        market_ml = (
            float(r["market_ml_home_close_prob"])
            if pd.notna(r.get("market_ml_home_close_prob"))
            else None
        )

        # Moneyline.
        p_home = float(r["p_home_win"])
        ml_side = "HOME" if p_home >= 0.5 else "AWAY"
        ml_prob = max(p_home, 1 - p_home)
        ml_long = f"{home} ML" if ml_side == "HOME" else f"{away} ML"
        ml_market_prob = (
            (market_ml if ml_side == "HOME" else 1 - market_ml)
            if market_ml is not None
            else None
        )
        ml_edge = (ml_prob - ml_market_prob) * 100 if ml_market_prob is not None else None
        rows.append(_journal_row(
            now, game_pk, game_date, away, home,
            market="moneyline", pick=ml_side, pick_long=ml_long,
            model_prob=ml_prob, market_prob=ml_market_prob, edge_pp=ml_edge,
            total_line=None, total_line_source=None,
            confidence=abs(p_home - 0.5) * 2.0,
        ))

        # Run line.
        p_rl = float(r["p_home_runline_cover"])
        rl_side = "HOME" if p_rl >= 0.5 else "AWAY"
        rl_prob = max(p_rl, 1 - p_rl)
        rl_long = f"{home} -1.5" if rl_side == "HOME" else f"{away} +1.5"
        rows.append(_journal_row(
            now, game_pk, game_date, away, home,
            market="runline", pick=rl_side, pick_long=rl_long,
            model_prob=rl_prob, market_prob=None, edge_pp=None,
            total_line=None, total_line_source=None,
            confidence=abs(p_rl - 0.5) * 2.0,
        ))

        # Total.
        p_over = r.get("p_total_over")
        if pd.notna(p_over):
            p_over = float(p_over)
            total_line = None
            line_source = "baseline"
            if pd.notna(r.get("effective_total_line")):
                total_line = float(r["effective_total_line"])
            elif pd.notna(r.get("market_total_close")):
                total_line = float(r["market_total_close"])
                line_source = "market"
            elif pd.notna(r.get("total_line")):
                total_line = float(r["total_line"])
            if pd.notna(r.get("total_line_source")):
                line_source = str(r["total_line_source"])
            ou_side = "OVER" if p_over >= 0.5 else "UNDER"
            ou_prob = max(p_over, 1 - p_over)
            ou_long = f"{ou_side} {total_line:g}" if total_line is not None else f"{ou_side}"
            rows.append(_journal_row(
                now, game_pk, game_date, away, home,
                market="total", pick=ou_side, pick_long=ou_long,
                model_prob=ou_prob, market_prob=None, edge_pp=None,
                total_line=total_line, total_line_source=line_source,
                confidence=abs(p_over - 0.5) * 2.0,
            ))

    return record_predictions(rows)


def _journal_row(
    now: str,
    game_pk: int,
    game_date: date_cls,
    away: str,
    home: str,
    *,
    market: str,
    pick: str,
    pick_long: str,
    model_prob: float,
    market_prob: float | None,
    edge_pp: float | None,
    total_line: float | None,
    total_line_source: str | None,
    confidence: float,
) -> dict:
    tier = _confidence_tier(confidence)
    return {
        "recorded_at": now,
        "game_pk": int(game_pk),
        "game_date": game_date,
        "season": int(pd.to_datetime(game_date).year) if game_date is not None else None,
        "away_team_abbr": away,
        "home_team_abbr": home,
        "market": market,
        "pick": pick,
        "pick_long": pick_long,
        "model_prob": float(model_prob),
        "market_prob": float(market_prob) if market_prob is not None else None,
        "edge_pp": float(edge_pp) if edge_pp is not None else None,
        "total_line": float(total_line) if total_line is not None else None,
        "total_line_source": total_line_source,
        "confidence": float(confidence),
        "tier": tier,
    }


def _confidence_tier(confidence: float) -> str:
    """Bucket confidence into a tier label.

    Delegates to :func:`mlb_model.app.services._confidence_tier` so the
    journal and the dashboard ALWAYS agree on which probability counts
    as "premium" / "strong" / etc. Earlier drafts of this file inlined
    its own thresholds and they diverged from the dashboard's -- the
    journal would then label things one way and the /season tier
    breakdown would disagree with the / dashboard pills. Importing
    the canonical function eliminates that drift.
    """
    # Late import: this module is loaded during predict, and the app
    # layer in turn imports the journal. Doing the import inside the
    # function avoids the cycle while still keeping a single source
    # of truth.
    from mlb_model.app.services import _confidence_tier as canonical

    return canonical(confidence)
