"""Tests for the cold-.300-hitter analysis (offline, no network).

Covers the deterministic core:
  * Hit-drought counts trailing *appeared* games with no hit.
  * Batting average is derived from H/AB and the .300 threshold selects the
    right qualified hitters.
  * The threshold breakdown reports the qualified-hitter share.
"""

from __future__ import annotations

from datetime import date

from mlb_model.analysis import hitter_slump as hs


def _glog(pairs: list[tuple[str, int]]) -> list[hs.GameLogEntry]:
    return [hs.GameLogEntry(game_date=date.fromisoformat(d), hits=h) for d, h in pairs]


# --- drought --------------------------------------------------------------- #
def test_drought_counts_trailing_hitless_games():
    log = _glog([("2026-06-01", 2), ("2026-06-02", 0), ("2026-06-03", 0)])
    assert hs.hit_drought(log) == 2


def test_drought_zero_when_last_game_has_hit():
    log = _glog([("2026-06-01", 0), ("2026-06-02", 1)])
    assert hs.hit_drought(log) == 0


def test_drought_full_log_when_no_hit_at_all():
    log = _glog([("2026-06-01", 0), ("2026-06-02", 0)])
    assert hs.hit_drought(log) == 2
    assert hs.last_hit_date(log) is None


def test_last_hit_date_picks_most_recent():
    log = _glog([("2026-06-01", 1), ("2026-06-05", 3), ("2026-06-06", 0)])
    assert hs.last_hit_date(log) == date(2026, 6, 5)


# --- average + threshold --------------------------------------------------- #
def _hitter(hits, ab, pa):
    return hs.HitterSeason(
        player_id=hits * 1000 + ab, name=f"p{hits}-{ab}", team="ABC", team_id=1,
        hits=hits, at_bats=ab, plate_appearances=pa, games=pa // 4,
    )


def test_avg_derived_from_hits_and_at_bats():
    assert _hitter(90, 300, 330).avg == 0.30
    assert _hitter(0, 0, 0).avg == 0.0  # no divide-by-zero for a PA-less player


def test_threshold_breakdown_qualified_only():
    hitters = [
        _hitter(93, 300, 330),  # .310, qualified — in the club
        _hitter(90, 300, 330),  # .300, qualified — in the club (>=)
        _hitter(60, 220, 240),  # .273, qualified — not in the club
        _hitter(5, 10, 12),     # .500 but only 12 PA — not qualified
    ]
    bd = hs.threshold_breakdown(hitters, season=2026, threshold=0.300)
    assert bd.n_at_threshold == 2
    assert bd.shares["qualified"].denom_count == 3
    assert round(bd.shares["qualified"].pct, 1) == round(200 / 3, 1)


# --- selection via find_slumping_hitters (no network) ---------------------- #
def test_find_selects_qualified_300_bats_only(monkeypatch):
    hot = _hitter(93, 300, 330)      # .310, qualified
    below = _hitter(60, 220, 240)    # .273, qualified — excluded by average
    tiny = _hitter(5, 10, 12)        # .500 but not qualified — excluded

    # Everyone is in a 3-game hitless skid.
    monkeypatch.setattr(
        hs, "fetch_gamelog",
        lambda pid, season: _glog([("2026-08-30", 0), ("2026-08-31", 0), ("2026-09-01", 0)]),
    )
    out = hs.find_slumping_hitters(
        2026, as_of=date(2026, 9, 1), with_news=False, with_matchups=False,
        hitters=[hot, below, tiny],
    )
    assert {s.player_id for s in out} == {hot.player_id}
    assert out[0].drought_games == 3
    assert out[0].batting_average == 0.31
