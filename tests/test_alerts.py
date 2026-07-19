"""Tests for the Discord high-confidence pick alerts (offline, no network)."""

from __future__ import annotations

from datetime import date

from mlb_model.automation import alerts


_GAME = [
    {"tier": "premium", "matchup": "NYY @ BOS", "market": "ML", "pick": "NYY",
     "confidence": 0.74, "edge_pp": 10.2},
    {"tier": "strong", "matchup": "LAD @ MIN", "market": "RL", "pick": "LAD -1.5",
     "confidence": 0.58, "edge_pp": 6.0},
]
_PROP = [
    {"tier": "premium", "player": "Aaron Judge", "team": "NYY", "market": "HR",
     "model_prob": 0.24, "edge_pp": 12.8, "opp_sp": "Crochet", "opp_throws": "L"},
]


def test_build_message_none_when_empty():
    assert alerts.build_message(date(2026, 6, 25), [], []) is None


def test_build_message_includes_both_sections():
    msg = alerts.build_message(date(2026, 6, 25), _GAME, _PROP)
    assert "Game markets" in msg and "Hitter props" in msg
    assert "🟢" in msg and "🟡" in msg
    assert "NYY @ BOS" in msg
    assert "NYY vs Crochet (L): 🟢 Judge HR 24%" in msg   # grouped prop line
    assert "74%" in msg                                    # game confidence rendered


def test_build_message_caps_long_lists(monkeypatch):
    monkeypatch.setattr(alerts.settings, "alert_max_prop_picks", 2)
    props = [dict(_PROP[0], player=f"P{i}") for i in range(10)]
    msg = alerts.build_message(date(2026, 6, 25), [], props)
    # 2 shown + an "…and 8 more" summary line.
    assert "and 8 more" in msg
    assert msg.count("🟢") == 2


def test_build_message_uncapped_shows_everything(monkeypatch):
    monkeypatch.setattr(alerts.settings, "alert_max_prop_picks", 2)
    props = [dict(_PROP[0], player=f"P{i}") for i in range(10)]
    msg = alerts.build_message(date(2026, 6, 25), [], props, uncapped=True)
    assert "more" not in msg        # no "…and N more" truncation
    assert msg.count("🟢") == 10    # all 10 listed


def test_chunk_respects_limit():
    long = "\n".join([f"line {i} " + "x" * 80 for i in range(100)])
    chunks = alerts._chunk(long, limit=500)
    assert len(chunks) > 1
    assert all(len(c) <= 500 for c in chunks)


def test_post_to_discord_noop_without_webhook(monkeypatch):
    monkeypatch.setattr(alerts.settings, "discord_webhook_url", None)
    assert alerts.post_to_discord("hi") is False


def _wire(monkeypatch, *, game=_GAME, prop=_PROP, pk=None):
    monkeypatch.setattr(alerts, "select_game_picks", lambda t: game)
    monkeypatch.setattr(alerts, "todays_matchups", lambda t, **k: [])
    monkeypatch.setattr(alerts, "select_prop_picks", lambda m: prop)
    monkeypatch.setattr(alerts, "select_pitcher_strikeouts", lambda m: pk or [])


def test_send_daily_alert_dry_run(monkeypatch, tmp_path):
    _wire(monkeypatch)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    res = alerts.send_daily_alert(date(2026, 6, 25), dry_run=True)
    assert res["sent"] is False and res["reason"] == "dry-run"
    assert res["n_game"] == 2 and res["n_prop"] == 1 and res["message"]


def test_send_daily_alert_dedup(monkeypatch, tmp_path):
    _wire(monkeypatch)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(alerts, "post_to_discord", lambda content, **k: True)
    first = alerts.send_daily_alert(date(2026, 6, 25))
    assert first["sent"] is True
    second = alerts.send_daily_alert(date(2026, 6, 25))   # marker now exists
    assert second["sent"] is False and second["reason"] == "already-sent-today"


def test_send_daily_alert_no_picks(monkeypatch, tmp_path):
    _wire(monkeypatch, game=[], prop=[], pk=[])
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    res = alerts.send_daily_alert(date(2026, 6, 25))
    assert res["sent"] is False and res["reason"] == "no-picks"


# --- per-pitcher strikeout projection (replaces the 1+K flood) -------------- #
def _matchup_high_k():
    # SEA's Gilbert (k_pct .30) faces a CLE lineup of high-K bats.
    cle = {"team_abbr": "CLE", "batters": [
        {"name": f"B{i}", "k_prob": 0.74, "hit_prob": 0.50, "hr_prob": 0.05, "tb_prob": 0.20}
        for i in range(4)
    ]}
    sea = {"team_abbr": "SEA", "starter": {"name": "Logan Gilbert", "throws": "R", "k_pct": 0.30},
           "batters": []}
    return {"game_pk": 1, "away": cle, "home": sea}


def test_pitcher_strikeouts_collapses_to_one_line():
    ks = alerts.select_pitcher_strikeouts([_matchup_high_k()])
    assert len(ks) == 1                      # one line, not four batter props
    k = ks[0]
    assert k["pitcher"] == "Logan Gilbert" and k["vs_team"] == "CLE"
    assert k["est_k"] == round(0.30 * alerts.EXPECTED_BATTERS_FACED, 1)


def test_prop_picks_exclude_strikeouts():
    # Even with a 1+K-heavy lineup, no "1+ K" prop rows are emitted.
    picks = alerts.select_prop_picks([_matchup_high_k()])
    assert all(p["market"] != "1+ K" for p in picks)


def test_build_message_renders_pitcher_k_section():
    pk = [{"pitcher": "Logan Gilbert", "team": "SEA", "throws": "R", "vs_team": "CLE", "est_k": 6.6}]
    msg = alerts.build_message(date(2026, 6, 25), [], [], pk)
    assert "Pitcher Ks (est)" in msg and "Gilbert (SEA) 6.6" in msg


# --- afternoon lineup-props alert ------------------------------------------ #
def _wire_afternoon(monkeypatch, *, prop=_PROP, seen=None):
    monkeypatch.setattr(alerts, "todays_matchups", lambda t, **k: [{"game_pk": 1}])
    monkeypatch.setattr(alerts, "select_prop_picks", lambda m: prop)
    monkeypatch.setattr(alerts, "_alerted_prop_keys", lambda t: seen or set())


def test_afternoon_message_none_when_empty():
    assert alerts.build_afternoon_message(date(2026, 7, 18), []) is None


def test_afternoon_message_lists_props():
    msg = alerts.build_afternoon_message(date(2026, 7, 18), _PROP)
    assert "afternoon lineup props" in msg
    assert "🟢 Judge HR 24%" in msg


def test_afternoon_sends_only_new_props(monkeypatch, tmp_path):
    # Judge's HR prop was already journaled by the morning run -> excluded.
    _wire_afternoon(monkeypatch, seen={("Aaron Judge", "HR")})
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    res = alerts.send_afternoon_props(date(2026, 7, 18), dry_run=True)
    assert res["n_prop"] == 1 and res["n_prop_new"] == 0
    assert res["sent"] is False and res["reason"] == "no-new-props"


def test_afternoon_sends_unseen_props(monkeypatch, tmp_path):
    _wire_afternoon(monkeypatch, seen={("Someone Else", "HR")})
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(alerts, "post_to_discord", lambda content, **k: True)
    res = alerts.send_afternoon_props(date(2026, 7, 18))
    assert res["n_prop_new"] == 1 and res["sent"] is True
    second = alerts.send_afternoon_props(date(2026, 7, 18))
    assert second["sent"] is False and second["reason"] == "already-sent-today"


def test_afternoon_marker_separate_from_morning(tmp_path, monkeypatch):
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    d = date(2026, 7, 18)
    assert alerts._marker_path(d) != alerts._marker_path(d, kind="afternoon")


def test_afternoon_dry_run_does_not_log_alerted(monkeypatch, tmp_path):
    calls = []
    _wire_afternoon(monkeypatch)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(alerts, "_log_alerted_props", lambda t, ps: calls.append(ps))
    res = alerts.send_afternoon_props(date(2026, 7, 18), dry_run=True)
    assert res["n_prop_new"] == 1 and calls == []
    live = alerts.send_afternoon_props(date(2026, 7, 18), dry_run=True)
    assert live["n_prop_new"] == 1  # dry run didn't poison the dedup


def test_afternoon_failed_post_does_not_log_alerted(monkeypatch, tmp_path):
    calls = []
    _wire_afternoon(monkeypatch)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(alerts, "_log_alerted_props", lambda t, ps: calls.append(ps))
    monkeypatch.setattr(alerts, "post_to_discord", lambda content, **k: False)
    res = alerts.send_afternoon_props(date(2026, 7, 18))
    assert res["sent"] is False and calls == []


def test_afternoon_successful_post_logs_alerted(monkeypatch, tmp_path):
    calls = []
    _wire_afternoon(monkeypatch)
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    monkeypatch.setattr(alerts, "_log_alerted_props", lambda t, ps: calls.append(ps))
    monkeypatch.setattr(alerts, "post_to_discord", lambda content, **k: True)
    res = alerts.send_afternoon_props(date(2026, 7, 18))
    assert res["sent"] is True and len(calls) == 1


def test_alerted_props_log_round_trip(monkeypatch, tmp_path):
    monkeypatch.setattr(alerts.settings, "data_dir", tmp_path)
    d = date(2026, 7, 19)
    assert alerts._alerted_prop_keys(d) == set()
    alerts._log_alerted_props(d, _PROP)
    assert alerts._alerted_prop_keys(d) == {("Aaron Judge", "HR")}
    alerts._log_alerted_props(d, _PROP)  # idempotent
    import pandas as pd
    assert len(pd.read_parquet(alerts._alerted_props_log_path())) == 1


def test_not_started_filter():
    from datetime import UTC, datetime

    now = datetime(2026, 7, 19, 20, 30, tzinfo=UTC)
    assert alerts._not_started("2026-07-19 23:10:00", now=now) is True
    assert alerts._not_started("2026-07-19 17:10:00", now=now) is False
    assert alerts._not_started(None, now=now) is True
    assert alerts._not_started("garbage", now=now) is True


def test_afternoon_skips_started_games(monkeypatch, tmp_path):
    from datetime import UTC, datetime, timedelta

    now = datetime.now(UTC)
    started = {"scheduled_start": (now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S")}
    upcoming = {"scheduled_start": (now + timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S")}
    monkeypatch.setattr(alerts, "todays_matchups", lambda t, **k: [started, upcoming])
    captured = []
    monkeypatch.setattr(
        alerts, "select_prop_picks", lambda m: captured.append(m) or []
    )
    monkeypatch.setattr(alerts, "_alerted_prop_keys", lambda t: set())
    monkeypatch.setattr(alerts.settings, "cache_dir", tmp_path)
    alerts.send_afternoon_props(date(2026, 7, 19), dry_run=True)
    assert captured[0] == [upcoming]
