"""Tests for the slumping-slugger analysis (offline, no network).

Covers the deterministic core:
  * HR-drought counts trailing *appeared* games with no HR.
  * Threshold breakdown computes the right share across denominators.
  * History CSV is append-only but idempotent for a given day.
  * News classification only fires when an injury/external term sits near
    the player's name and isn't negated (the false-positive guard).
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from mlb_model.analysis import matchups, news, transactions
from mlb_model.analysis import slugger_slump as ss


def _glog(pairs: list[tuple[str, int]]) -> list[ss.GameLogEntry]:
    return [
        ss.GameLogEntry(game_date=date.fromisoformat(d), home_runs=hr) for d, hr in pairs
    ]


# --- drought --------------------------------------------------------------- #
def test_drought_counts_trailing_hrless_games():
    log = _glog([("2026-06-01", 1), ("2026-06-02", 0), ("2026-06-03", 0)])
    assert ss.hr_drought(log) == 2


def test_drought_zero_when_last_game_has_hr():
    log = _glog([("2026-06-01", 0), ("2026-06-02", 1)])
    assert ss.hr_drought(log) == 0


def test_drought_full_log_when_no_hr_at_all():
    log = _glog([("2026-06-01", 0), ("2026-06-02", 0)])
    assert ss.hr_drought(log) == 2
    assert ss.last_hr_date(log) is None


def test_last_hr_date_picks_most_recent():
    log = _glog([("2026-06-01", 1), ("2026-06-05", 1), ("2026-06-06", 0)])
    assert ss.last_hr_date(log) == date(2026, 6, 5)


# --- threshold breakdown --------------------------------------------------- #
def _hitter(hr, pa):
    return ss.HitterSeason(
        player_id=hr * 1000 + pa, name=f"p{hr}-{pa}", team="ABC", team_id=1,
        home_runs=hr, plate_appearances=pa, at_bats=pa, games=pa // 4,
    )


def test_threshold_breakdown_denominators():
    hitters = [
        _hitter(20, 300),  # qualified, 15+
        _hitter(15, 250),  # qualified, 15+
        _hitter(5, 220),   # qualified, has HR, not 15+
        _hitter(1, 10),    # has HR, not qualified
        _hitter(0, 5),     # has PA only
    ]
    bd = ss.threshold_breakdown(hitters, season=2026, threshold=15)
    assert bd.n_at_threshold == 2
    assert bd.shares["all_pa"].denom_count == 5
    assert bd.shares["has_hr"].denom_count == 4
    assert bd.shares["qualified"].denom_count == 3
    assert round(bd.shares["qualified"].pct, 1) == round(200 / 3, 1)


# --- history persistence --------------------------------------------------- #
def test_append_history_is_idempotent_per_day(tmp_path):
    path = tmp_path / "hist.csv"
    hitters = [_hitter(20, 300), _hitter(5, 220)]
    bd = ss.threshold_breakdown(hitters, season=2026, threshold=15, as_of=date(2026, 6, 22))

    ss.append_history(bd, path=path)
    ss.append_history(bd, path=path)  # same day again -> overwrite, not append
    import csv
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    # exactly 3 denominator rows for the one date, no duplicates
    assert len(rows) == 3
    assert {r["as_of"] for r in rows} == {"2026-06-22"}

    # a later date adds rows without clobbering the first
    bd2 = ss.threshold_breakdown(hitters, season=2026, threshold=15, as_of=date(2026, 6, 23))
    ss.append_history(bd2, path=path)
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6


# --- news classification --------------------------------------------------- #
def test_surname_drops_suffix():
    assert news._surname("Bobby Witt Jr.") == "witt"
    assert news._surname("Munetaka Murakami") == "murakami"
    assert news._surname("Pete Crow-Armstrong") == "crow-armstrong"


def test_injury_term_near_name_fires():
    title = "angels' mike trout placed on il with right hamstring strain"
    assert news._term_near_name(title, "trout", news.INJURY_TERMS) is True


def test_injury_term_about_teammate_does_not_fire():
    # The common real-world case: a teammate's injury headline that doesn't
    # even name our player. Surname-absent => never fires.
    title = "royals' witt jr. out of lineup as he deals with knee injury"
    assert news._term_near_name(title, "walker", news.INJURY_TERMS) is False


def test_injury_term_far_from_name_does_not_fire():
    # Player named, but the injury belongs to someone else far away in the
    # headline => proximity window keeps it from firing.
    title = "matt olson stays productive as braves address spencer strider injury concerns"
    assert news._term_near_name(title, "olson", news.INJURY_TERMS) is False


def test_negation_suppresses_injury():
    title = "braves' matt olson shuts down injury concerns after breaking out of slump"
    assert news._term_near_name(title, "olson", news.INJURY_TERMS) is False


def test_external_term_near_name_fires():
    title = "miguel vargas receives first off day since may 20"
    assert news._term_near_name(title, "vargas", news.EXTERNAL_TERMS) is True


# --- advisory wording (verdict driven by verified transactions) ----------- #
def test_advisory_injury_verified_shows_il_and_note():
    s = ss.SluggerStatus(
        player_id=1, name="X", team="NYY", home_runs=20, games=60,
        drought_games=6, last_hr_date=date(2026, 5, 24), last_game_date=date(2026, 5, 31),
        days_since_last_game=22, is_absent=True,
        verified_cause="injury-verified", il_type="10-day", il_start=date(2026, 6, 2),
        injury_note="Right rib stress fracture",
    )
    assert s.status_label == "INJURY (verified)"
    assert "10-day IL" in s.advisory
    assert "2026-06-02" in s.advisory
    assert "Right rib stress fracture" in s.advisory


def test_advisory_unclear_when_unverified_even_if_absent():
    s = ss.SluggerStatus(
        player_id=2, name="Y", team="CHC", home_runs=16, games=70,
        drought_games=6, last_hr_date=date(2026, 6, 13), last_game_date=date(2026, 6, 14),
        days_since_last_game=8, is_absent=True, verified_cause="unverified",
    )
    assert s.status_label == "UNCLEAR"
    assert "UNCLEAR" in s.advisory
    assert "no IL stint" in s.advisory


def test_advisory_external_verified():
    s = ss.SluggerStatus(
        player_id=3, name="Z", team="LAD", home_runs=15, games=65,
        drought_games=7, last_hr_date=date(2026, 6, 1), last_game_date=date(2026, 6, 10),
        days_since_last_game=12, is_absent=True,
        verified_cause="external-verified", external_move="optioned/sent to minors",
        move_date=date(2026, 6, 11),
    )
    assert s.status_label == "EXTERNAL (verified)"
    assert "optioned" in s.advisory


# --- transactions / IL verification ---------------------------------------- #
def _tx(date_str, pid, desc):
    return {"date": date_str, "person": {"id": pid}, "description": desc}


def test_verify_on_il_with_injury_note():
    txs = [
        _tx("2026-04-15", 99, "RF Aaron Judge changed number to 42."),
        _tx("2026-06-02", 99,
            "New York Yankees placed RF Aaron Judge on the 10-day injured list "
            "retroactive to June 2, 2026. Right rib stress fracture."),
    ]
    v = transactions.verify_status(txs, 99, as_of=date(2026, 6, 22))
    assert v.cause == "injury-verified"
    assert v.il_type == "10-day"
    assert v.il_start == date(2026, 6, 2)
    assert v.injury_note == "Right rib stress fracture"


def test_verify_activated_clears_il():
    txs = [
        _tx("2026-05-01", 7, "Team placed C Foo on the 10-day injured list. Left wrist."),
        _tx("2026-05-20", 7, "Team activated C Foo from the 10-day injured list."),
    ]
    v = transactions.verify_status(txs, 7, as_of=date(2026, 6, 22))
    assert v.cause == "unverified"


def test_verify_transfer_keeps_original_start_updates_type():
    txs = [
        _tx("2026-05-01", 7, "Team placed P Bar on the 15-day injured list "
                             "retroactive to April 28, 2026. Elbow."),
        _tx("2026-05-22", 7, "Team transferred P Bar from the 15-day injured list "
                             "to the 60-day injured list."),
    ]
    v = transactions.verify_status(txs, 7, as_of=date(2026, 6, 22))
    assert v.cause == "injury-verified"
    assert v.il_type == "60-day"
    assert v.il_start == date(2026, 4, 28)


def test_verify_external_optioned():
    txs = [_tx("2026-06-11", 5, "Team optioned 1B Baz to Triple-A Somewhere.")]
    v = transactions.verify_status(txs, 5, as_of=date(2026, 6, 22))
    assert v.cause == "external-verified"
    assert "optioned" in v.external_move


def test_verify_future_dated_move_ignored():
    txs = [_tx("2026-07-01", 5, "Team placed 1B Baz on the 10-day injured list.")]
    v = transactions.verify_status(txs, 5, as_of=date(2026, 6, 22))
    assert v.cause == "unverified"


# --- morning-sync integration (offline) ------------------------------------ #
def test_morning_sync_records_snapshot(tmp_path, monkeypatch):
    from mlb_model.automation import morning_sync as msync

    monkeypatch.setattr(ss, "HISTORY_PATH", tmp_path / "hist.csv")
    monkeypatch.setattr(ss, "fetch_season_hitting", lambda season: [_hitter(20, 300), _hitter(5, 220)])

    result = msync._record_slugger_snapshot(date(2026, 6, 22))
    assert result["recorded"] is True
    assert result["n_at_threshold"] == 1
    assert (tmp_path / "hist.csv").exists()


def test_morning_sync_skips_offseason(tmp_path, monkeypatch):
    from mlb_model.automation import morning_sync as msync

    monkeypatch.setattr(ss, "HISTORY_PATH", tmp_path / "hist.csv")
    # No player has a PA yet -> off-season / pre-season, don't write junk.
    monkeypatch.setattr(ss, "fetch_season_hitting", lambda season: [_hitter(0, 0)])

    result = msync._record_slugger_snapshot(date(2026, 1, 15))
    assert result["recorded"] is False
    assert not (tmp_path / "hist.csv").exists()


# --- dynamic HR threshold --------------------------------------------------- #
def test_dynamic_threshold_tracks_top_pct():
    # 100 qualified hitters with HR = 0..99. Top 10% -> the 10th-highest is 90.
    hitters = [
        ss.HitterSeason(player_id=i, name=f"p{i}", team="X", team_id=1,
                        home_runs=i, plate_appearances=300, at_bats=300, games=70)
        for i in range(100)
    ]
    assert ss.dynamic_threshold(hitters, target_pct=10, floor=0) == 90


def test_dynamic_threshold_respects_floor():
    # Early-season-ish: top 10% is only 6 HR, but the floor holds at 15.
    hitters = [
        ss.HitterSeason(player_id=i, name=f"p{i}", team="X", team_id=1,
                        home_runs=i % 8, plate_appearances=300, at_bats=300, games=70)
        for i in range(100)
    ]
    assert ss.dynamic_threshold(hitters, target_pct=10, floor=15) == 15


def test_dynamic_threshold_ignores_non_qualified():
    # Big HR totals but under the PA floor shouldn't define the bar.
    hitters = [
        ss.HitterSeason(player_id=1, name="reg", team="X", team_id=1,
                        home_runs=12, plate_appearances=400, at_bats=400, games=90),
        ss.HitterSeason(player_id=2, name="parttime", team="X", team_id=1,
                        home_runs=40, plate_appearances=50, at_bats=50, games=15),
    ]
    # Only the 400-PA hitter qualifies -> bar is his 12, floored at 10.
    assert ss.dynamic_threshold(hitters, target_pct=10, floor=10) == 12


# --- active-today gating ---------------------------------------------------- #
def _status(**kw):
    base = dict(
        player_id=1, name="X", team="NYY", home_runs=20, games=70, drought_games=6,
        last_hr_date=date(2026, 6, 10), last_game_date=date(2026, 6, 24),
        days_since_last_game=1, is_absent=False,
    )
    base.update(kw)
    return ss.SluggerStatus(**base)


def test_active_today_requires_playing_unverified_present():
    assert _status(verified_cause="unverified", plays_today=True, is_absent=False).active_today
    assert not _status(verified_cause="unverified", plays_today=False).active_today  # off today
    assert not _status(verified_cause="unverified", plays_today=True, is_absent=True).active_today
    assert not _status(verified_cause="injury-verified", plays_today=True).active_today


# --- pitching matchup grading (offline) ------------------------------------ #
def _profile(throws="R", ip=80.0, hr9=None, ops=None, vl=None, vr=None):
    return matchups.PitcherProfile(
        pitcher_id=1, name="Test Pitcher", throws=throws, innings=ip,
        home_runs=10, hr9=hr9, ops_allowed=ops, vs_l_ops=vl, vs_r_ops=vr,
    )


def _tm(profile):
    return matchups.TeamMatchup(game_date=date(2026, 6, 24), opp_pitcher=profile)


def test_matchup_none_when_no_game():
    assert matchups.grade_for_batter("R", None).label == "NONE"


def test_matchup_favorable_homer_prone_pitcher():
    # RHB vs a pitcher who is homer-prone and soft to righties.
    v = matchups.grade_for_batter("R", _tm(_profile(throws="R", hr9=1.8, ops=.820, vr=.910)))
    assert v.label == "FAVORABLE"
    assert v.ops_allowed == .910  # used the vs-RHB split
    assert v.edge > 0.8


def test_matchup_tough_stingy_pitcher():
    v = matchups.grade_for_batter("R", _tm(_profile(throws="R", hr9=0.6, ops=.600, vr=.610)))
    assert v.label == "TOUGH"
    assert v.edge < -0.8


def test_matchup_switch_hitter_takes_platoon_side():
    # Switch hitter vs LHP -> bats R -> should use the vs-RHB (vr) split.
    prof = _profile(throws="L", hr9=1.2, vl=.500, vr=.900)
    v = matchups.grade_for_batter("S", _tm(prof))
    assert v.ops_allowed == .900


def test_matchup_falls_back_to_overall_ops_without_splits():
    v = matchups.grade_for_batter("L", _tm(_profile(throws="R", hr9=1.2, ops=.780, vl=None, vr=None)))
    assert v.ops_allowed == .780


def test_ip_parser_handles_thirds():
    assert matchups._ip_to_float("16.1") == 16 + 1 / 3
    assert matchups._ip_to_float("16.2") == 16 + 2 / 3
    assert matchups._ip_to_float("16.0") == 16.0


def test_headline_age_days():
    h = news.Headline(
        title="t", url="u", source="s",
        published=datetime(2026, 6, 12, tzinfo=timezone.utc),
    )
    age = h.age_days(now=datetime(2026, 6, 22, tzinfo=timezone.utc))
    assert age == 10.0
