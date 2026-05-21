"""Tests for the prediction journal: recording, grading, calibration.

These tests are hermetic -- every fixture builds its own DuckDB warehouse
and journal Parquet file in a tempdir. No network, no real models, no
state shared with the live ``data/`` directory.
"""

from __future__ import annotations

from datetime import datetime, UTC

import numpy as np
import pandas as pd
import pytest

from mlb_model.journal.metrics import (
    _grade_row,
    calibration_bins,
    grade_journal,
    rolling_accuracy,
    season_summary,
    slice_breakdown,
)


# --------------------------------------------------------------------------- #
# Synthetic fixtures                                                          #
# --------------------------------------------------------------------------- #


def _make_journal(rows: list[dict]) -> pd.DataFrame:
    """Build a journal-shaped DataFrame from a list of partial dicts."""
    defaults = dict(
        recorded_at=datetime.now(UTC).isoformat(),
        season=2026,
        away_team_abbr="ATL",
        home_team_abbr="NYM",
        market="moneyline",
        pick="HOME",
        pick_long="NYM ML",
        market_prob=None,
        edge_pp=None,
        total_line=None,
        total_line_source=None,
        confidence=0.3,
        tier="strong",
    )
    enriched = []
    for r in rows:
        merged = {**defaults, **r}
        enriched.append(merged)
    df = pd.DataFrame(enriched)
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


# --------------------------------------------------------------------------- #
# _grade_row                                                                  #
# --------------------------------------------------------------------------- #


class TestGradeRow:
    def test_moneyline_home_wins_pick_home(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 3,
            "market": "moneyline", "pick": "HOME",
        })
        assert _grade_row(row) == "win"

    def test_moneyline_home_wins_pick_away(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 3,
            "market": "moneyline", "pick": "AWAY",
        })
        assert _grade_row(row) == "loss"

    def test_moneyline_tie_is_push(self) -> None:
        """MLB doesn't tie in regular season, but the grader still handles it."""
        row = pd.Series({
            "status": "Final", "home_score": 4, "away_score": 4,
            "market": "moneyline", "pick": "HOME",
        })
        assert _grade_row(row) == "push"

    def test_unfinalized_is_pending(self) -> None:
        row = pd.Series({
            "status": "In Progress", "home_score": 3, "away_score": 2,
            "market": "moneyline", "pick": "HOME",
        })
        assert _grade_row(row) == "pending"

    def test_runline_home_covers(self) -> None:
        # Home wins by 2 → HOME -1.5 covers.
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 3,
            "market": "runline", "pick": "HOME",
        })
        assert _grade_row(row) == "win"

    def test_runline_home_misses(self) -> None:
        # Home wins by exactly 1 → HOME -1.5 misses, AWAY +1.5 covers.
        row = pd.Series({
            "status": "Final", "home_score": 4, "away_score": 3,
            "market": "runline", "pick": "HOME",
        })
        assert _grade_row(row) == "loss"

    def test_runline_away_covers(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 4, "away_score": 3,
            "market": "runline", "pick": "AWAY",
        })
        assert _grade_row(row) == "win"

    def test_total_over_clears(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 4,
            "market": "total", "pick": "OVER", "total_line": 8.5,
        })
        assert _grade_row(row) == "win"

    def test_total_under_clears(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 3, "away_score": 2,
            "market": "total", "pick": "UNDER", "total_line": 8.5,
        })
        assert _grade_row(row) == "win"

    def test_total_exact_line_is_push(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 4,
            "market": "total", "pick": "OVER", "total_line": 9.0,
        })
        assert _grade_row(row) == "push"

    def test_total_missing_line_pending(self) -> None:
        row = pd.Series({
            "status": "Final", "home_score": 5, "away_score": 4,
            "market": "total", "pick": "OVER", "total_line": float("nan"),
        })
        assert _grade_row(row) == "pending"


# --------------------------------------------------------------------------- #
# season_summary                                                              #
# --------------------------------------------------------------------------- #


class TestSeasonSummary:
    def test_empty_returns_empty_list(self) -> None:
        assert season_summary(pd.DataFrame()) == []

    def test_basic_win_rate(self) -> None:
        graded = pd.DataFrame([
            # 3 wins, 2 losses, 0 pushes  → 60% win rate
            {"season": 2026, "market": "moneyline", "result": "win", "roi_units": 0.909, "model_prob": 0.6},
            {"season": 2026, "market": "moneyline", "result": "win", "roi_units": 0.909, "model_prob": 0.7},
            {"season": 2026, "market": "moneyline", "result": "win", "roi_units": 0.909, "model_prob": 0.55},
            {"season": 2026, "market": "moneyline", "result": "loss", "roi_units": -1.0, "model_prob": 0.65},
            {"season": 2026, "market": "moneyline", "result": "loss", "roi_units": -1.0, "model_prob": 0.58},
        ])
        out = season_summary(graded)
        assert len(out) == 1
        s = out[0]
        assert s.season == 2026
        assert s.market == "moneyline"
        assert s.wins == 3
        assert s.losses == 2
        assert s.pushes == 0
        assert s.n == 5
        assert s.win_rate == pytest.approx(0.6)
        # ROI: 3 wins * 0.909 - 2 losses * 1.0 = 0.727
        assert s.roi_units == pytest.approx(2.727 - 2.0, abs=0.01)

    def test_pushes_excluded_from_win_rate(self) -> None:
        graded = pd.DataFrame([
            {"season": 2026, "market": "moneyline", "result": "win", "roi_units": 0.909, "model_prob": 0.6},
            {"season": 2026, "market": "moneyline", "result": "loss", "roi_units": -1.0, "model_prob": 0.55},
            {"season": 2026, "market": "moneyline", "result": "push", "roi_units": 0.0, "model_prob": 0.5},
        ])
        out = season_summary(graded)
        assert out[0].win_rate == 0.5  # 1/2, push excluded
        assert out[0].pushes == 1

    def test_brier_perfect_predictions(self) -> None:
        """Brier score is 0 when probabilities perfectly match outcomes."""
        graded = pd.DataFrame([
            {"season": 2026, "market": "moneyline", "result": "win", "roi_units": 0.909, "model_prob": 1.0},
            {"season": 2026, "market": "moneyline", "result": "loss", "roi_units": -1.0, "model_prob": 0.0},
        ])
        out = season_summary(graded)
        # We clip prob to [1e-6, 1-1e-6] so it's not exactly zero.
        assert out[0].brier < 1e-6 * 10


# --------------------------------------------------------------------------- #
# calibration_bins                                                            #
# --------------------------------------------------------------------------- #


class TestCalibration:
    def test_empty(self) -> None:
        assert calibration_bins(pd.DataFrame()).empty

    def test_well_calibrated_model(self) -> None:
        """A well-calibrated 60%-confidence pick should win ~60% of the time."""
        rng = np.random.default_rng(seed=42)
        rows = []
        for _ in range(1000):
            p = rng.uniform(0.4, 0.7)
            won = rng.random() < p
            rows.append({
                "market": "moneyline",
                "result": "win" if won else "loss",
                "model_prob": p,
                "roi_units": 0.909 if won else -1.0,
            })
        df = pd.DataFrame(rows)
        bins = calibration_bins(df, n_bins=5)
        # Every bucket with reasonable n should have forecast ≈ realized.
        big_bins = bins[bins["n"] >= 50]
        for _, b in big_bins.iterrows():
            assert abs(b["forecast_mean"] - b["realized_rate"]) < 0.07

    def test_overconfident_model(self) -> None:
        """A model that says 90% but wins 50% should show large diff_pp."""
        df = pd.DataFrame([
            {"market": "moneyline", "result": "win" if i % 2 == 0 else "loss",
             "model_prob": 0.9, "roi_units": 0.909 if i % 2 == 0 else -1.0}
            for i in range(100)
        ])
        bins = calibration_bins(df, n_bins=10)
        last_bucket = bins[bins["bucket_lo"] >= 0.8]
        assert not last_bucket.empty
        assert last_bucket.iloc[0]["diff_pp"] > 30  # 40pp gap is the truth


# --------------------------------------------------------------------------- #
# rolling_accuracy                                                            #
# --------------------------------------------------------------------------- #


class TestRollingAccuracy:
    def test_empty(self) -> None:
        out = rolling_accuracy(pd.DataFrame())
        assert out.empty

    def test_basic_30day_window(self) -> None:
        rows = []
        for i in range(60):
            d = pd.Timestamp("2026-04-01") + pd.Timedelta(days=i)
            rows.append({
                "market": "moneyline",
                "game_date": d.date(),
                "result": "win" if i % 2 == 0 else "loss",
                "roi_units": 0.909 if i % 2 == 0 else -1.0,
            })
        df = pd.DataFrame(rows)
        out = rolling_accuracy(df, window_days=30)
        # Every day should be approximately 50% win rate.
        recent = out[out["n"] >= 20]
        assert (recent["win_rate"] - 0.5).abs().max() < 0.05


# --------------------------------------------------------------------------- #
# slice_breakdown                                                             #
# --------------------------------------------------------------------------- #


class TestSliceBreakdown:
    def test_by_tier(self) -> None:
        df = pd.DataFrame([
            {"tier": "premium", "result": "win", "roi_units": 0.909},
            {"tier": "premium", "result": "win", "roi_units": 0.909},
            {"tier": "premium", "result": "loss", "roi_units": -1.0},
            {"tier": "lean", "result": "loss", "roi_units": -1.0},
            {"tier": "lean", "result": "loss", "roi_units": -1.0},
            {"tier": "lean", "result": "win", "roi_units": 0.909},
        ])
        out = slice_breakdown(df, by="tier")
        prem = out[out["tier"] == "premium"].iloc[0]
        lean = out[out["tier"] == "lean"].iloc[0]
        assert prem["win_rate"] == pytest.approx(2/3)
        assert lean["win_rate"] == pytest.approx(1/3)
        # Premium ROI should be positive, lean negative.
        assert prem["roi"] > 0
        assert lean["roi"] < 0

    def test_unknown_dimension(self) -> None:
        """Asking for a column we don't have just returns empty, not crash."""
        df = pd.DataFrame([{"tier": "lean", "result": "win", "roi_units": 0.0}])
        out = slice_breakdown(df, by="nonexistent_column")
        assert out.empty


# --------------------------------------------------------------------------- #
# grade_journal (integration with warehouse)                                  #
# --------------------------------------------------------------------------- #


class TestGradeJournalEndToEnd:
    def test_no_journal_file_returns_empty(self, tmp_path, monkeypatch) -> None:
        """grade_journal() must not crash when no journal exists yet."""
        from mlb_model.journal import record as record_mod
        monkeypatch.setattr(record_mod, "JOURNAL_PATH", tmp_path / "missing.parquet")
        # Re-import the metrics module so it picks up the patched path.
        import importlib
        from mlb_model.journal import metrics as metrics_mod
        importlib.reload(metrics_mod)
        out = metrics_mod.grade_journal()
        assert out.empty
