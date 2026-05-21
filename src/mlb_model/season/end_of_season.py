"""End-of-season "full sweep".

What this does, in order, for a season that has just finished:

1.  **Verify completeness.** Make sure the warehouse has the whole
    season's worth of finalized data. If not, refuse to run -- the
    sweep is meaningless if half the games are missing.

2.  **Snapshot the journal.** Copy ``predictions.parquet`` into the
    archive folder so the "this is what the model said in real time
    during the season" record is preserved even if the journal is
    later edited or migrated.

3.  **Run the full walk-forward backtest through this season.**
    Same code path as ``mlb-model backtest`` -- the single source of
    truth -- so the report is directly comparable to prior seasons.

4.  **Grade the journal.** Combine the live-prediction journal with
    backtest output to produce per-market summaries, calibration
    bins, and tier breakdowns.

5.  **Identify worst-performing slices.** Surface the segments where
    the model bled the most (e.g. "interleague road favorites in
    August" or "totals at Coors") so we know where to invest next
    off-season.

6.  **Write a structured report** to ``reports/end_of_season_<YEAR>/``:

       * ``report.md``       -- human-readable summary with verdict + recommendations
       * ``summary.json``    -- machine-readable metrics for diffing
       * ``backtest.csv``    -- per-season metrics from the walk-forward run
       * ``slices.csv``      -- worst-slice analysis

7.  **Archive the model snapshot** that produced the season's live
    predictions so we always have a "model-of-record" we can compare
    next year's iteration against.

This module deliberately reuses existing code (backtest, journal
metrics, model archive) rather than introducing parallel logic. New
behavior lives in :func:`_render_report` (the writer) and
:func:`_identify_worst_slices` (the analyzer).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import date as date_cls, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import query
from mlb_model.journal import grade_journal, season_summary
from mlb_model.journal.record import JOURNAL_PATH
from mlb_model.logging import get_logger
from mlb_model.season.rollover import is_regular_season_over

log = get_logger("season.end_of_season")

REPORTS_ROOT: Path = settings.project_root / "reports"


# A "worst slice" is one we have at least this many graded picks in.
# Tiny slices (n=3) generate noise, not signal.
_MIN_SLICE_N = 20
_TOP_K_WORST = 8


@dataclass
class EndOfSeasonReport:
    """In-memory representation of the report we just wrote to disk."""

    season: int
    completed_at: str
    output_dir: str
    journal_snapshot_path: str | None
    backtest_csv_path: str | None
    summary_json_path: str
    report_md_path: str
    slices_csv_path: str | None
    summary_by_market: list[dict[str, Any]]
    worst_slices: list[dict[str, Any]]
    recommendations: list[str]
    model_snapshot_dir: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_end_of_season_sweep(
    season: int,
    *,
    train_start: int | None = None,
    force: bool = False,
    skip_backtest: bool = False,
) -> EndOfSeasonReport:
    """Execute the full sweep for ``season``.

    Parameters
    ----------
    season
        The 4-digit year to evaluate (e.g. ``2026``).
    train_start
        Earliest training season for the walk-forward backtest. Defaults
        to ``season - 6`` so we always use a rolling 6-year window.
    force
        Skip the "is the regular season actually over?" check. Useful
        for re-running a previous season's sweep, or for testing.
    skip_backtest
        Skip the (slow) walk-forward backtest. The sweep then relies
        solely on the live prediction journal. Mostly for fast tests.
    """
    if not force and not is_regular_season_over(season):
        raise RuntimeError(
            f"Refusing to run end-of-season sweep for {season}: "
            f"regular season does not appear to be complete. "
            f"Pass force=True if you really mean it."
        )

    train_start = train_start or max(2010, season - 6)
    output_dir = REPORTS_ROOT / f"end_of_season_{season}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("end_of_season.start", season=season, output_dir=str(output_dir))

    completed_at = datetime.now().isoformat(timespec="seconds")

    # ----- 1. Journal snapshot. Cheap so do it first.
    journal_snapshot_path: str | None = None
    if JOURNAL_PATH.exists():
        snap = output_dir / "predictions_journal_snapshot.parquet"
        shutil.copy2(JOURNAL_PATH, snap)
        journal_snapshot_path = str(snap)
        log.info("end_of_season.journal.snapshot", path=str(snap))

    # ----- 2. Grade the live journal for this season.
    graded = grade_journal(season=season)
    summaries = [s.as_dict() for s in season_summary(graded)]
    log.info("end_of_season.journal.graded", n_summaries=len(summaries))

    # ----- 3. Walk-forward backtest (optional/slow).
    backtest_csv_path: str | None = None
    backtest_df: pd.DataFrame | None = None
    if not skip_backtest:
        try:
            from mlb_model.backtest.walkforward import run_walkforward

            log.info(
                "end_of_season.backtest.start",
                start=train_start + 1, end=season,
            )
            backtest_df = run_walkforward(train_start + 1, season)
            if not backtest_df.empty:
                backtest_csv_path = str(output_dir / "backtest.csv")
                backtest_df.to_csv(backtest_csv_path, index=False)
        except Exception:  # noqa: BLE001 -- a failed backtest shouldn't block report
            log.exception("end_of_season.backtest.failed")

    # ----- 4. Worst-slice analysis (drawing only from the journal grading,
    # which is the closest thing to "what would a bettor have actually
    # experienced this year").
    slices_df = _identify_worst_slices(graded)
    slices_csv_path: str | None = None
    if not slices_df.empty:
        slices_csv_path = str(output_dir / "slices.csv")
        slices_df.to_csv(slices_csv_path, index=False)

    # ----- 5. Archive the model that produced these predictions.
    model_snapshot_dir = _archive_model_of_record(season, output_dir)

    # ----- 6. Recommendations: data-driven prose.
    recommendations = _generate_recommendations(
        season=season, summaries=summaries,
        slices=slices_df, backtest=backtest_df,
    )

    # ----- 7. Write all output artefacts.
    summary_json = {
        "season": season,
        "completed_at": completed_at,
        "train_start": train_start,
        "summary_by_market": summaries,
        "worst_slices": slices_df.to_dict("records") if not slices_df.empty else [],
        "recommendations": recommendations,
        "backtest_csv": backtest_csv_path,
        "journal_snapshot": journal_snapshot_path,
        "model_snapshot_dir": str(model_snapshot_dir) if model_snapshot_dir else None,
    }
    summary_json_path = output_dir / "summary.json"
    summary_json_path.write_text(
        json.dumps(summary_json, indent=2, default=str),
        encoding="utf-8",
    )

    report_md_path = output_dir / "report.md"
    report_md_path.write_text(
        _render_report(
            season=season,
            completed_at=completed_at,
            summaries=summaries,
            slices=slices_df,
            backtest=backtest_df,
            recommendations=recommendations,
        ),
        encoding="utf-8",
    )

    report = EndOfSeasonReport(
        season=season,
        completed_at=completed_at,
        output_dir=str(output_dir),
        journal_snapshot_path=journal_snapshot_path,
        backtest_csv_path=backtest_csv_path,
        summary_json_path=str(summary_json_path),
        report_md_path=str(report_md_path),
        slices_csv_path=slices_csv_path,
        summary_by_market=summaries,
        worst_slices=slices_df.to_dict("records") if not slices_df.empty else [],
        recommendations=recommendations,
        model_snapshot_dir=str(model_snapshot_dir) if model_snapshot_dir else None,
        metadata={"train_start": train_start},
    )
    log.info(
        "end_of_season.done", season=season,
        report_md=str(report_md_path),
        backtest_csv=backtest_csv_path,
    )
    return report


# --------------------------------------------------------------------------- #
# Internals                                                                    #
# --------------------------------------------------------------------------- #


def _identify_worst_slices(graded: pd.DataFrame) -> pd.DataFrame:
    """Find the segments of the slate where the model bled the most.

    We score by ROI (units lost) rather than win rate alone, so a small
    slice with a brutal 30% win rate doesn't get prioritized over a
    larger slice with a steady 49% drag.
    """
    if graded is None or graded.empty:
        return pd.DataFrame()

    finalized = graded[graded["result"].isin({"win", "loss", "push"})].copy()
    if finalized.empty:
        return pd.DataFrame()

    finalized["is_win"] = (finalized["result"] == "win").astype(float)
    finalized["is_loss"] = (finalized["result"] == "loss").astype(float)
    finalized["roi_value"] = pd.to_numeric(finalized["roi_units"], errors="coerce").fillna(0.0)

    # Enrich with team-vs-team / venue / month dimensions for slicing.
    pks = finalized["game_pk"].astype(int).tolist()
    if pks:
        placeholders = ", ".join(["?"] * len(pks))
        # MLB Stats API stamps ``venue_name`` directly on the game row,
        # which is the source of truth. Our ``venues.venue_id`` is a
        # synthetic CRC32 hash of the home team abbr (MLB's venue_id is
        # not stable across eras), so joining on it would be wrong --
        # fall back to looking up by abbr if the game row's venue_name
        # is somehow blank.
        ctx = query(
            f"""
            SELECT g.game_pk, g.venue_id, g.day_night, g.scheduled_start,
                   COALESCE(NULLIF(g.venue_name, ''), v.name) AS venue_name
            FROM games g
            LEFT JOIN venues v ON v.team_abbr = g.home_team_abbr
            WHERE g.game_pk IN ({placeholders})
            """,
            tuple(pks),
        )
        finalized = finalized.merge(ctx, on="game_pk", how="left")
        finalized["month"] = pd.to_datetime(finalized["game_date"]).dt.month_name()

    dimensions = [
        ("market", "market"),
        ("tier", "tier"),
        ("home_team_abbr", "home_team"),
        ("away_team_abbr", "away_team"),
        ("venue_name", "venue") if "venue_name" in finalized.columns else (None, None),
        ("month", "month") if "month" in finalized.columns else (None, None),
        ("day_night", "day_night") if "day_night" in finalized.columns else (None, None),
        ("total_line_source", "total_line_source"),
    ]

    rows: list[dict[str, Any]] = []
    for col, label in dimensions:
        if col is None or col not in finalized.columns:
            continue
        # ``dropna=True`` drops the "this column doesn't apply" buckets
        # (e.g. total_line_source for moneyline rows) instead of grouping
        # them under a stringified ``'nan'`` that would clutter the report.
        grouped = (
            finalized.dropna(subset=[col])
            .groupby(col, dropna=True)
            .agg(
                n=("is_win", "size"),
                wins=("is_win", "sum"),
                losses=("is_loss", "sum"),
                roi=("roi_value", "sum"),
            )
            .reset_index()
        )
        grouped = grouped[grouped["n"] >= _MIN_SLICE_N]
        if grouped.empty:
            continue
        denom = (grouped["wins"] + grouped["losses"]).replace(0, pd.NA)
        grouped["win_rate"] = (grouped["wins"] / denom).astype(float)
        grouped["dimension"] = label
        grouped["value"] = grouped[col].astype(str)
        grouped = grouped.drop(columns=[col])
        rows.append(grouped[["dimension", "value", "n", "wins", "losses", "win_rate", "roi"]])

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values("roi").head(_TOP_K_WORST).reset_index(drop=True)
    return out


def _archive_model_of_record(season: int, output_dir: Path) -> Path | None:
    """Copy the currently-trained model files into the report folder.

    Naming: ``model_of_record_<SEASON>/``. We don't move them -- the
    live model keeps running. We just want a frozen copy for reference.
    """
    model_dir = settings.model_dir
    if not model_dir.exists():
        return None
    artefacts = list(model_dir.glob("*.joblib"))
    if not artefacts:
        return None

    dest = output_dir / f"model_of_record_{season}"
    dest.mkdir(parents=True, exist_ok=True)
    for p in artefacts:
        shutil.copy2(p, dest / p.name)
    log.info(
        "end_of_season.model_archive",
        season=season,
        dest=str(dest), n_files=len(artefacts),
    )
    return dest


def _generate_recommendations(
    *,
    season: int,
    summaries: list[dict[str, Any]],
    slices: pd.DataFrame,
    backtest: pd.DataFrame | None,
) -> list[str]:
    """Produce a short list of "here's what to look at next off-season" notes.

    Pure data-driven heuristics -- nothing fancy. The goal is to point
    the human at the worst-performing segments, not to auto-fix anything.
    """
    recs: list[str] = []

    # Per-market sanity:
    by_market = {row["market"]: row for row in summaries}
    if "moneyline" in by_market:
        ml = by_market["moneyline"]
        wr = ml.get("win_rate")
        if wr is not None and wr == wr and wr < 0.520:
            recs.append(
                f"Moneyline win rate this season was {wr*100:.1f}% on {ml['n']} graded picks "
                f"(target: 54-58%). Re-examine the feature set -- closing-line value and "
                f"bullpen freshness usually pay biggest dividends here."
            )
        elif wr is not None and wr == wr and wr >= 0.560:
            recs.append(
                f"Moneyline win rate of {wr*100:.1f}% beat the 56% target -- the high-tier "
                f"calibration is working. Document the feature set as a baseline before any "
                f"off-season changes."
            )

    if "total" in by_market:
        ou = by_market["total"]
        wr = ou.get("win_rate")
        if wr is not None and wr == wr and wr < 0.500:
            recs.append(
                f"Totals (O/U) bled at {wr*100:.1f}% over {ou['n']} graded picks. The "
                f"league-average baseline likely needs supplementing with park-adjusted "
                f"matchup totals -- especially when no closing line is available."
            )

    # Slices analysis:
    if not slices.empty:
        worst = slices.iloc[0]
        recs.append(
            f"Biggest single drag was {worst['dimension']} = {worst['value']!r} "
            f"({int(worst['n'])} picks, {worst['win_rate']*100:.1f}% win rate, "
            f"{worst['roi']:+.2f}u P/L). Investigate whether feature coverage is "
            f"thin in this segment, or whether the line/model disagrees systematically."
        )
        # Surface up to two more if they're meaningful losses.
        for _, row in slices.iloc[1:3].iterrows():
            if row["roi"] < -2.0:
                recs.append(
                    f"Other notable drag: {row['dimension']} = {row['value']!r} "
                    f"({int(row['n'])} picks, {row['win_rate']*100:.1f}%, {row['roi']:+.2f}u)."
                )

    # Backtest drift:
    if backtest is not None and not backtest.empty and len(backtest) >= 2:
        recent = backtest.tail(2)["ml_accuracy"].astype(float).tolist()
        if len(recent) == 2 and (recent[-1] - recent[-2]) < -0.015:
            recs.append(
                f"Walk-forward moneyline accuracy fell from {recent[-2]*100:.1f}% to "
                f"{recent[-1]*100:.1f}% year-over-year -- a >1.5pp drop. Consider whether "
                f"a rule change (pitch clock, ball composition) shifted the run environment "
                f"in a way the rolling features can't yet capture."
            )

    if not recs:
        recs.append(
            "No structural concerns detected. Continue weekly retraining on a "
            "6-season rolling window."
        )
    return recs


def _render_report(
    *,
    season: int,
    completed_at: str,
    summaries: list[dict[str, Any]],
    slices: pd.DataFrame,
    backtest: pd.DataFrame | None,
    recommendations: list[str],
) -> str:
    """Render the human-readable markdown report."""
    lines: list[str] = []
    lines.append(f"# {season} season post-mortem")
    lines.append("")
    lines.append(f"_Generated {completed_at}._")
    lines.append("")
    lines.append("This report is the model's annual self-assessment. It pulls together")
    lines.append("everything we said in real time during the season (from the prediction")
    lines.append("journal), grades it against actual outcomes, and tries to surface the")
    lines.append("places where next year's iteration should focus.")
    lines.append("")

    lines.append("## Per-market performance")
    lines.append("")
    if summaries:
        lines.append("| Market | N | W-L-P | Win % | P/L (u) | Brier | Log-loss |")
        lines.append("|---|---:|:---:|---:|---:|---:|---:|")
        for row in summaries:
            wr = row.get("win_rate")
            wr_s = f"{wr*100:.1f}%" if wr is not None and wr == wr else "—"
            brier = row.get("brier")
            brier_s = f"{brier:.3f}" if brier is not None and brier == brier else "—"
            ll = row.get("log_loss")
            ll_s = f"{ll:.3f}" if ll is not None and ll == ll else "—"
            wlp = f"{row['wins']}-{row['losses']}"
            if row.get("pushes"):
                wlp += f"-{row['pushes']}"
            lines.append(
                f"| {row['market']} | {row['n']} | {wlp} | {wr_s} | "
                f"{row['roi_units']:+.2f} | {brier_s} | {ll_s} |"
            )
    else:
        lines.append("_No graded predictions in the journal -- did the model run live this season?_")
    lines.append("")

    lines.append("## Worst-performing slices")
    lines.append("")
    if slices is not None and not slices.empty:
        lines.append("Sorted by P/L drag. Slices with fewer than "
                     f"{_MIN_SLICE_N} graded picks are excluded as noise.")
        lines.append("")
        lines.append("| Dimension | Value | N | Win % | P/L (u) |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in slices.iterrows():
            wr = row.get("win_rate")
            wr_s = f"{wr*100:.1f}%" if pd.notna(wr) else "—"
            lines.append(
                f"| {row['dimension']} | {row['value']} | {int(row['n'])} | "
                f"{wr_s} | {row['roi']:+.2f} |"
            )
    else:
        lines.append("_Not enough data to identify slices -- need at least "
                     f"{_MIN_SLICE_N} graded picks per segment._")
    lines.append("")

    lines.append("## Walk-forward backtest (cross-season comparison)")
    lines.append("")
    if backtest is not None and not backtest.empty:
        lines.append("| Season | N | ML acc | ML top-10% | ML top-3% | RL acc | OU acc | CLV |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in backtest.iterrows():
            lines.append(
                f"| {int(r['season'])} | {int(r['n_games'])} | "
                f"{r['ml_accuracy']*100:.1f}% | {r['ml_top10_acc']*100:.1f}% | "
                f"{r['ml_top3_acc']*100:.1f}% | {r['rl_accuracy']*100:.1f}% | "
                f"{r['ou_accuracy']*100:.1f}% | {r['clv_ml']:+.4f} |"
            )
    else:
        lines.append("_Walk-forward backtest was skipped or failed -- see logs._")
    lines.append("")

    lines.append("## Off-season recommendations")
    lines.append("")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("_Artefacts in this folder:_")
    lines.append("")
    lines.append("* ``summary.json`` -- machine-readable metrics (diffable year-over-year)")
    lines.append("* ``backtest.csv`` -- per-season walk-forward numbers")
    lines.append("* ``slices.csv`` -- worst-slice breakdown")
    lines.append("* ``predictions_journal_snapshot.parquet`` -- frozen copy of every live prediction")
    lines.append("* ``model_of_record_<SEASON>/`` -- copies of the trained .joblib artefacts")
    lines.append("")
    return "\n".join(lines)
