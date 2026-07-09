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
    assert "PREMIUM" in msg and "STRONG" in msg
    assert "NYY @ BOS" in msg and "Aaron Judge" in msg
    assert "74%" in msg                       # game confidence rendered
    assert "+13pp" in msg                     # prop edge rounded


def test_build_message_caps_long_lists(monkeypatch):
    monkeypatch.setattr(alerts.settings, "alert_max_prop_picks", 2)
    props = [dict(_PROP[0], player=f"P{i}") for i in range(10)]
    msg = alerts.build_message(date(2026, 6, 25), [], props)
    # 2 shown + an "…and 8 more" summary line.
    assert "and 8 more" in msg
    assert msg.count("PREMIUM") == 2


def test_build_message_uncapped_shows_everything(monkeypatch):
    monkeypatch.setattr(alerts.settings, "alert_max_prop_picks", 2)
    props = [dict(_PROP[0], player=f"P{i}") for i in range(10)]
    msg = alerts.build_message(date(2026, 6, 25), [], props, uncapped=True)
    assert "more" not in msg           # no "…and N more" truncation
    assert msg.count("PREMIUM") == 10  # all 10 listed


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
    assert "Pitcher strikeouts" in msg and "estimated K's: 6.6" in msg
