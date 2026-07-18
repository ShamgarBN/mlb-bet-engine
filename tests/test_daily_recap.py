"""Morning scorecard: summaries, message format, dedup (offline)."""

from __future__ import annotations

from datetime import date

from mlb_model.automation import daily_recap

_PROPS = {"premium": {"wins": 4, "losses": 2}, "strong": {"wins": 6, "losses": 6}}
_GAMES = {"strong": {"wins": 1, "losses": 0}}
_PITCHERS = [
    {"pitcher": "Dylan Cease", "team": "SD", "est_k": 8.8, "actual_k": 9, "hit": True},
    {"pitcher": "Logan Gilbert", "team": "SEA", "est_k": 7.1, "actual_k": 4, "hit": False},
    {"pitcher": "Rained Out", "team": "COL", "est_k": 6.0, "actual_k": None, "hit": False},
]


def test_message_none_when_nothing_settled():
    assert daily_recap.build_recap_message(date(2026, 7, 17), {}, {}, []) is None


def test_message_shows_tier_records_and_pcts():
    msg = daily_recap.build_recap_message(date(2026, 7, 17), _PROPS, _GAMES, [])
    assert "yesterday's scorecard" in msg
    assert "🟢 Premium: 4-2 (67%)" in msg
    assert "🟡 Strong: 6-6 (50%)" in msg
    assert "🟡 Strong: 1-0 (100%)" in msg


def test_message_pitcher_section_counts_only_graded():
    msg = daily_recap.build_recap_message(date(2026, 7, 17), {}, {}, _PITCHERS)
    assert "Pitcher K spots__ (1/2 reached estimate)" in msg
    assert "Dylan Cease (SD): est 8.8 → 9 ✓" in msg
    assert "Logan Gilbert (SEA): est 7.1 → 4" in msg
    assert "Rained Out (COL): est 6 → —" in msg


def test_message_empty_sections_labeled():
    msg = daily_recap.build_recap_message(date(2026, 7, 17), _PROPS, {}, [])
    assert "_none alerted_" in msg


def test_pct_zero_denominator():
    assert daily_recap._pct(0, 0) == "—"


def _wire(monkeypatch, tmp_path):
    from mlb_model.automation import alerts

    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(daily_recap, "prop_summary", lambda t: _PROPS)
    monkeypatch.setattr(daily_recap, "game_summary", lambda t: _GAMES)
    monkeypatch.setattr(daily_recap, "pitcher_summary", lambda t: [])
    import mlb_model.automation.morning_sync as ms

    monkeypatch.setattr(ms, "_refresh_finals_for_date", lambda t: {})
    monkeypatch.setattr("mlb_model.journal.props.grade_props", lambda: 0)


def test_run_recap_dry_run_and_dedup(monkeypatch, tmp_path):
    from mlb_model.automation import alerts

    _wire(monkeypatch, tmp_path)
    res = daily_recap.run_daily_recap(date(2026, 7, 18), dry_run=True)
    assert res["sent"] is False and res["reason"] == "dry-run"
    assert res["target"] == "2026-07-17" and res["message"]

    monkeypatch.setattr(alerts, "post_to_discord", lambda content, **k: True)
    monkeypatch.setattr(daily_recap, "post_to_discord", alerts.post_to_discord, raising=False)
    first = daily_recap.run_daily_recap(date(2026, 7, 18))
    assert first["sent"] is True
    second = daily_recap.run_daily_recap(date(2026, 7, 18))
    assert second["sent"] is False and second["reason"] == "already-sent-today"
