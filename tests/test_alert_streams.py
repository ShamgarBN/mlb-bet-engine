"""Tests for the staggered Discord alert streams (offline, no network).

Covers the deterministic core: early/late window split, cold-bat colour +
ordering, per-stream dedup (pitchers, game markets "if changed"), the per-day
marker gate, and dispatch. Network-touching selection helpers are monkeypatched.
"""

from __future__ import annotations

from datetime import date

import pytest

from mlb_model.automation import alert_streams as S


@pytest.fixture(autouse=True)
def _tmp_data_dir(tmp_path, monkeypatch):
    """Point every on-disk log/marker at a tmp dir so tests don't collide."""
    monkeypatch.setattr(S.settings, "data_dir", tmp_path)
    monkeypatch.setattr(S.settings, "cache_dir", tmp_path / "cache")
    (tmp_path / "cache").mkdir(parents=True, exist_ok=True)
    # alerts.py resolves its own paths off settings too.
    from mlb_model.automation import alerts
    monkeypatch.setattr(alerts.settings, "data_dir", tmp_path)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path / "cache")
    yield


# --- window split ---------------------------------------------------------- #
def test_unknown_start_is_early():
    assert S._is_early(None, 17) is True
    assert S._is_early("not-a-date", 17) is True


def test_filter_window_partitions_by_local_hour(monkeypatch):
    # Make the split deterministic regardless of the host timezone by stubbing
    # the per-game early/late decision.
    early = {"scheduled_start": "E", "away_team_abbr": "A", "home_team_abbr": "B"}
    late = {"scheduled_start": "L", "away_team_abbr": "C", "home_team_abbr": "D"}
    monkeypatch.setattr(S, "_is_early", lambda s, c: s == "E")
    assert S._filter_window([early, late], "early") == [early]
    assert S._filter_window([early, late], "late") == [late]


# --- cold-bat colour + ordering -------------------------------------------- #
def test_cold_rows_filter_and_sort():
    snap = {"bats": [
        {"name": "A", "matchup_label": "TOUGH", "drought_games": 4, "active_today": True},
        {"name": "B", "matchup_label": "FAVORABLE", "drought_games": 3, "active_today": True},
        {"name": "C", "matchup_label": "NEUTRAL", "drought_games": 6, "active_today": False},
    ]}
    rows = S._cold_rows(snap, "bats")
    assert [r["name"] for r in rows] == ["B", "A"]  # favorable first; inactive dropped


def test_cold_message_colours():
    rows = [
        {"name": "Fav", "team": "X", "matchup_label": "FAVORABLE", "drought_games": 3, "matchup_pitcher": "Joe Cold"},
        {"name": "Tough", "team": "Y", "matchup_label": "TOUGH", "drought_games": 4, "matchup_pitcher": "Ace Good"},
        {"name": "TBD", "team": "Z", "matchup_label": "NONE", "drought_games": 5, "matchup_pitcher": ""},
    ]
    msg = S.build_cold_message(date(2026, 9, 1), rows, title="cold bats",
                               stat_fn=lambda r: "stat", drought_word="hitless")
    assert "🟢 Fav" in msg and "🔴 Tough" in msg and "⚪ TBD" in msg
    assert "vs Cold" in msg and "no probable yet" in msg


# --- game-market dedup ("if changed") -------------------------------------- #
def test_game_keys_roundtrip_and_dedup():
    t = date(2026, 9, 1)
    picks = [
        {"matchup": "A @ B", "market": "ML", "pick": "B ML", "tier": "premium"},
        {"matchup": "C @ D", "market": "O/U", "pick": "Over 8.5", "tier": "strong"},
    ]
    assert S._alerted_game_keys(t) == set()
    S._log_alerted_games(t, picks)
    keys = S._alerted_game_keys(t)
    assert ("A @ B", "ML", "B ML") in keys
    # A changed pick for the same game/market is NOT deduped (it's "changed").
    changed = [{"matchup": "A @ B", "market": "ML", "pick": "A ML", "tier": "strong"}]
    assert (changed[0]["matchup"], changed[0]["market"], changed[0]["pick"]) not in keys


def test_pitcher_dedup_reads_k_log():
    t = date(2026, 9, 1)
    from mlb_model.automation import alerts
    assert S._alerted_pitcher_ids(t) == set()
    alerts._log_pitcher_ks(t, [{"pitcher_id": 11, "pitcher": "P", "team": "X", "vs_team": "Y", "est_k": 6}])
    assert S._alerted_pitcher_ids(t) == {11}


# --- marker gate ----------------------------------------------------------- #
def test_guard_blocks_second_send_same_day():
    t = date(2026, 9, 1)
    m = S._guard(t, "early-pitchers", force=False, dry_run=False)
    assert m is not None
    m.parent.mkdir(parents=True, exist_ok=True)
    m.write_text("sent")
    # Second call same day → gated.
    assert S._guard(t, "early-pitchers", force=False, dry_run=False) is None
    # force overrides.
    assert S._guard(t, "early-pitchers", force=True, dry_run=False) is not None


# --- dispatch + dry-run plumbing ------------------------------------------- #
def test_run_stream_unknown_raises():
    with pytest.raises(KeyError):
        S.run_stream("nope", date(2026, 9, 1))


def test_send_pitchers_dry_run_filters_and_dedupes(monkeypatch):
    t = date(2026, 9, 1)
    matchups = [
        {"scheduled_start": "E", "away_team_abbr": "A", "home_team_abbr": "B"},
        {"scheduled_start": "L", "away_team_abbr": "C", "home_team_abbr": "D"},
    ]
    monkeypatch.setattr(S.alerts, "todays_matchups", lambda *a, **k: matchups)
    monkeypatch.setattr(S, "_is_early", lambda s, c: s == "E")
    # Both games have a high-K starter; only the early one should survive.
    monkeypatch.setattr(S.alerts, "select_pitcher_strikeouts", lambda ms: [
        {"pitcher_id": g["away_team_abbr"], "pitcher": f"SP {g['away_team_abbr']}",
         "team": g["away_team_abbr"], "vs_team": g["home_team_abbr"], "est_k": 7}
        for g in ms
    ])
    res = S.send_pitchers(t, window="early", dry_run=True)
    assert res["n_new"] == 1
    assert "SP A" in res["message"] and "SP C" not in res["message"]
