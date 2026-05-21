"""Tests for season lifecycle predicates + end-of-season sweep.

Lifecycle tests use the live warehouse (they're inherently coupled to
real game counts) but ONLY make read-only queries. The end-of-season
sweep test stubs out the slow walk-forward backtest so it runs in a
second or two against a temp directory.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from mlb_model.season.rollover import (
    detect_season_state,
    is_regular_season_over,
    last_completed_season,
)


class TestIsRegularSeasonOver:
    def test_future_date_returns_false(self) -> None:
        # April of 2026 cannot have a "season over" verdict for 2026.
        assert is_regular_season_over(2026, today=date(2026, 4, 15)) is False

    def test_no_data_returns_false(self) -> None:
        # Year far before warehouse coverage -> empty count -> False.
        assert is_regular_season_over(1900, today=date(2050, 12, 31)) is False


class TestSeasonState:
    def test_returns_known_state(self) -> None:
        status = detect_season_state()
        assert status.state in {"in_progress", "ended", "off_season", "pre_season"}
        assert status.season == date.today().year

    def test_state_consistent_with_finalized_count(self) -> None:
        status = detect_season_state()
        if status.state == "pre_season":
            assert status.finalized_games == 0
        elif status.state == "ended":
            # Heuristic threshold inside the module
            assert status.finalized_games >= 2200


class TestLastCompletedSeason:
    def test_returns_int_or_none(self) -> None:
        s = last_completed_season()
        assert s is None or isinstance(s, int)


# --------------------------------------------------------------------------- #
# End-of-season sweep                                                         #
# --------------------------------------------------------------------------- #


class TestEndOfSeasonSweep:
    def test_refuses_when_season_not_over(self, monkeypatch) -> None:
        """The sweep must refuse to run for an in-progress season unless force=True."""
        from mlb_model.season import end_of_season as eos

        monkeypatch.setattr(eos, "is_regular_season_over", lambda s, today=None: False)
        with pytest.raises(RuntimeError, match="Refusing"):
            eos.run_end_of_season_sweep(season=2026)

    def test_force_bypasses_completeness_check(self, tmp_path, monkeypatch) -> None:
        """force=True must skip the completeness gate AND produce a report,
        even if the journal is empty (smoke test for the artefact writer)."""
        from mlb_model.season import end_of_season as eos

        # Redirect every output path into the tempdir.
        monkeypatch.setattr(eos, "REPORTS_ROOT", tmp_path / "reports")
        # Stub the slow backtest -- we trust it via its own tests.
        monkeypatch.setattr(eos, "is_regular_season_over", lambda s, today=None: True)
        # Pretend grading found nothing.
        monkeypatch.setattr(eos, "grade_journal", lambda **kw: pd.DataFrame())
        monkeypatch.setattr(eos, "season_summary", lambda graded: [])

        report = eos.run_end_of_season_sweep(
            season=1999, force=True, skip_backtest=True,
        )
        assert report.season == 1999
        report_md = tmp_path / "reports" / "end_of_season_1999" / "report.md"
        assert report_md.exists()
        content = report_md.read_text()
        assert "1999 season post-mortem" in content
        assert "Off-season recommendations" in content
        # Empty-journal recommendation should appear (no concerns detected).
        assert "No structural concerns" in content

    def test_summary_json_is_diffable(self, tmp_path, monkeypatch) -> None:
        """summary.json must be valid JSON with stable top-level keys."""
        import json

        from mlb_model.season import end_of_season as eos

        monkeypatch.setattr(eos, "REPORTS_ROOT", tmp_path / "reports")
        monkeypatch.setattr(eos, "is_regular_season_over", lambda s, today=None: True)
        monkeypatch.setattr(eos, "grade_journal", lambda **kw: pd.DataFrame())
        monkeypatch.setattr(eos, "season_summary", lambda graded: [])

        report = eos.run_end_of_season_sweep(
            season=2050, force=True, skip_backtest=True,
        )
        payload = json.loads(open(report.summary_json_path).read())
        for key in [
            "season", "completed_at", "summary_by_market", "worst_slices",
            "recommendations",
        ]:
            assert key in payload
        assert payload["season"] == 2050


# --------------------------------------------------------------------------- #
# weekly_train auto-trigger                                                   #
# --------------------------------------------------------------------------- #


class TestWeeklyTrainAutoTrigger:
    def test_skips_when_report_already_exists(self, tmp_path, monkeypatch) -> None:
        """If the season's end-of-season report already exists, the
        auto-trigger must NOT re-run it on every Sunday for the rest of
        the off-season."""
        import mlb_model.season as season_pkg
        from mlb_model.automation import weekly_train as wt
        from mlb_model.config import settings

        # The marker file the auto-trigger looks for.
        report_dir = tmp_path / "reports" / "end_of_season_2026"
        report_dir.mkdir(parents=True)
        (report_dir / "summary.json").write_text('{"season": 2026}')

        monkeypatch.setattr(settings, "project_root", tmp_path)
        # ``weekly_train`` imports ``is_regular_season_over`` via the
        # ``mlb_model.season`` package re-export, so we must patch that
        # binding (not the one in ``rollover`` -- they're separate names).
        monkeypatch.setattr(
            season_pkg, "is_regular_season_over",
            lambda s, today=None: True,
        )

        result = wt._maybe_trigger_end_of_season(2026)
        assert result is not None
        assert result.get("skipped") is True
        assert result.get("reason") == "already_complete"

    def test_skips_when_season_not_over(self, monkeypatch) -> None:
        import mlb_model.season as season_pkg
        from mlb_model.automation import weekly_train as wt

        monkeypatch.setattr(
            season_pkg, "is_regular_season_over",
            lambda s, today=None: False,
        )
        result = wt._maybe_trigger_end_of_season(2026)
        assert result is None
