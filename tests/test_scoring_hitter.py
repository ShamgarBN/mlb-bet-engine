"""Tests for the per-hitter prop scoring functions.

Spec, in plain language:
  * A league-average batter vs a league-average pitcher must score 5.0
    (anchor verification — biggest source of regressions on the old design).
  * Elite batter vs poor pitcher should score > 7.0.
  * Poor batter vs elite pitcher should score < 3.0.
  * Pitcher OPS allowed vs hand should DOMINATE batter form for hit-score.
  * Missing inputs should never crash; only ``None`` when nothing usable.
"""

from __future__ import annotations

import pytest

from mlb_model.scoring.hitter import (
    BatterInputs,
    PitcherInputs,
    LG_AVG, LG_OPS_ALLOWED, LG_K_PCT_PIT, LG_K_PCT, LG_ISO, LG_HR9,
    LG_XBA, LG_BARREL, LG_XSLG, LG_HARDHIT, LG_EV,
    score_hit, score_hr, score_tb, score_k, score_matchup,
)


def _avg_batter(bats="R") -> BatterInputs:
    return BatterInputs(
        bats=bats,
        season_avg=LG_AVG, season_pa=400, season_iso=LG_ISO, season_k_pct=LG_K_PCT,
        split_avg=LG_AVG, split_pa=150, split_iso=LG_ISO, split_k_pct=LG_K_PCT,
        xba=LG_XBA, xslg=LG_XSLG, barrel_pct=LG_BARREL, hardhit_pct=LG_HARDHIT,
        avg_ev=LG_EV, bip=250,
    )


def _avg_pitcher(throws="R") -> PitcherInputs:
    return PitcherInputs(
        throws=throws,
        vs_lhb_ops=LG_OPS_ALLOWED, vs_rhb_ops=LG_OPS_ALLOWED,
        vs_lhb_hr9=LG_HR9, vs_rhb_hr9=LG_HR9,
        vs_lhb_k_pct=LG_K_PCT_PIT, vs_rhb_k_pct=LG_K_PCT_PIT,
        k_pct=LG_K_PCT_PIT, hr9=LG_HR9, batters_faced=400,
    )


# ─── Anchor: league avg vs league avg → 5.0 ────────────────────────────────

@pytest.mark.parametrize("fn", [score_hit, score_k])
def test_league_avg_matchup_scores_5(fn) -> None:
    assert fn(_avg_batter(), _avg_pitcher()) == pytest.approx(5.0, abs=0.1)


def test_league_avg_hr_lands_mid_scale() -> None:
    # HR score anchors a bit lower (HR is rare; avg power → ~4.5)
    s = score_hr(_avg_batter(), _avg_pitcher())
    assert 3.5 <= s <= 5.5, f"avg/avg HR score should be middle of scale, got {s}"


# ─── Elite vs poor → high score ────────────────────────────────────────────

def test_elite_batter_vs_poor_pitcher_hit_high() -> None:
    b = _avg_batter("R")
    b.season_avg = 0.330
    b.split_avg = 0.340
    b.xba = 0.310
    b.hardhit_pct = 0.50
    p = _avg_pitcher("R")
    p.vs_rhb_ops = 0.880  # very hittable
    p.fip = 5.20
    s = score_hit(b, p)
    assert s is not None and s >= 7.0, f"elite/poor hit should be high, got {s}"


def test_poor_batter_vs_elite_pitcher_hit_low() -> None:
    b = _avg_batter("R")
    b.season_avg = 0.180
    b.split_avg = 0.180
    b.xba = 0.190
    b.hardhit_pct = 0.22
    p = _avg_pitcher("R")
    p.vs_rhb_ops = 0.560  # ace
    p.fip = 2.70
    s = score_hit(b, p)
    assert s is not None and s <= 3.0, f"poor/elite hit should be low, got {s}"


# ─── HR composite-power anti-stacking ──────────────────────────────────────

def test_elite_power_does_not_auto_stack_to_10() -> None:
    """A genuine slugger vs avg pitcher in a neutral park should top out
    around 7.5-8.5 -- never 10. Composite power index prevents stacking.
    """
    b = _avg_batter("R")
    b.barrel_pct = 0.18; b.xslg = 0.60; b.hardhit_pct = 0.52; b.avg_ev = 93.0
    b.bip = 200
    s = score_hr(b, _avg_pitcher(), park_factor=1.0)
    assert s is not None and 7.0 <= s <= 9.0, f"elite power neutral matchup should be 7-9, got {s}"


def test_hr_score_responds_to_park() -> None:
    b = _avg_batter("R")
    p = _avg_pitcher("R")
    coors = score_hr(b, p, park_factor=1.30)
    petco = score_hr(b, p, park_factor=0.85)
    assert coors is not None and petco is not None
    assert coors > petco + 0.5, f"Coors should beat Petco by >=0.5, got coors={coors} petco={petco}"


# ─── Pitcher OPS-vs-hand is the dominant hit signal ────────────────────────

def test_pitcher_ops_swings_hit_score_more_than_batter_form() -> None:
    """The pitcher should move the score more than the batter does, within
    a realistic range. Sharpens that platoon matchup quality dominates.
    """
    b = _avg_batter("R")
    p_easy = _avg_pitcher("R"); p_easy.vs_rhb_ops = 0.840
    p_hard = _avg_pitcher("R"); p_hard.vs_rhb_ops = 0.600
    swing_from_pitcher = score_hit(b, p_easy) - score_hit(b, p_hard)

    b_hot = _avg_batter("R"); b_hot.split_avg = 0.300; b_hot.season_avg = 0.300
    b_cold = _avg_batter("R"); b_cold.split_avg = 0.200; b_cold.season_avg = 0.200
    p = _avg_pitcher("R")
    swing_from_batter = score_hit(b_hot, p) - score_hit(b_cold, p)

    assert swing_from_pitcher >= swing_from_batter, (
        f"pitcher should swing more than batter; got pitcher={swing_from_pitcher} batter={swing_from_batter}"
    )


# ─── Strikeout: direction sanity ───────────────────────────────────────────

def test_k_score_high_when_pitcher_strikes_out_many() -> None:
    b = _avg_batter("R")
    p = _avg_pitcher("R")
    p.vs_rhb_k_pct = 0.32  # very high
    p.k_pct = 0.30
    s = score_k(b, p)
    assert s is not None and s >= 6.5, f"high-K pitcher should yield high K-score, got {s}"


def test_k_score_low_when_contact_batter_meets_contact_pitcher() -> None:
    b = _avg_batter("R"); b.split_k_pct = 0.13; b.season_k_pct = 0.13
    p = _avg_pitcher("R"); p.vs_rhb_k_pct = 0.16; p.k_pct = 0.16
    s = score_k(b, p)
    assert s is not None and s <= 3.5, f"contact/contact should suppress K-score, got {s}"


# ─── Graceful degradation ──────────────────────────────────────────────────

def test_empty_inputs_return_none_not_crash() -> None:
    s = score_matchup(BatterInputs(), PitcherInputs())
    assert s.hit is None and s.hr is None and s.total_bases is None and s.strikeout is None


def test_partial_inputs_still_scores() -> None:
    # Only season AVG + pitcher FIP — minimal viable hit-score inputs
    b = BatterInputs(bats="R", season_avg=0.270, season_pa=200)
    p = PitcherInputs(throws="R", fip=3.50, batters_faced=300)
    s = score_hit(b, p)
    assert s is not None and 4.0 <= s <= 6.0, f"minimal-input hit score should be near 5, got {s}"


def test_tb_is_blend_of_hit_and_hr() -> None:
    b = _avg_batter("R"); p = _avg_pitcher("R")
    h = score_hit(b, p); hr = score_hr(b, p); tb = score_tb(b, p)
    assert tb is not None
    expected = round(0.6 * h + 0.4 * hr, 1)
    assert tb == expected, f"TB should be 0.6*hit + 0.4*hr; got tb={tb}, expected={expected}"


# ─── Shrinkage: tiny sample regresses toward league mean ───────────────────

def test_tiny_sample_regresses_hit_score() -> None:
    """A batter with a .500 AVG in 10 PA should not score elite -- shrinkage
    pulls them back toward league mean."""
    small = BatterInputs(bats="R", season_avg=0.500, season_pa=10, split_avg=0.500, split_pa=10)
    big = BatterInputs(bats="R", season_avg=0.330, season_pa=500, split_avg=0.340, split_pa=200)
    p = _avg_pitcher("R")
    assert score_hit(small, p) < score_hit(big, p), (
        "10-PA .500 batter should regress below 200-PA .340 batter after shrinkage"
    )
