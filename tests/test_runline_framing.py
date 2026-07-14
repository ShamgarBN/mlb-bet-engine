"""Market-anchored runline framing: sim lines, grading, pick labels, book priority."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlb_model.data.sources.odds_api import _best_market
from mlb_model.journal.metrics import _grade_row
from mlb_model.model.simulate import simulate_games


def _sim(home_rl_lines=None):
    """Two lopsided games: home crushes in game 0, away crushes in game 1."""
    return simulate_games(
        game_pks=np.array([1, 2]),
        pred_home=(np.array([7.0, 2.5]), np.array([2.0, 1.6])),
        pred_away=(np.array([2.5, 7.0]), np.array([1.6, 2.0])),
        home_rl_lines=home_rl_lines,
        n_sims=20_000,
        seed=7,
    )


def test_sim_default_frame_is_home_minus_15():
    preds = _sim()
    # Game 0: heavy home favorite -> P(home -1.5) well above 0.5.
    assert preds[0].p_home_runline_cover > 0.6
    # Game 1: heavy home dog -> P(home -1.5) well below 0.5.
    assert preds[1].p_home_runline_cover < 0.4


def test_sim_market_lines_flip_the_frame():
    # Game 1's market line is home +1.5 (away favorite). P(home +1.5)
    # must be much higher than P(home -1.5) for the same distributions.
    minus = _sim()[1].p_home_runline_cover
    plus = _sim(home_rl_lines=np.array([-1.5, 1.5]))[1].p_home_runline_cover
    assert plus > minus + 0.1
    # The pick derived from the +1.5 frame is AWAY, i.e. away -1.5 --
    # the side the market actually posts for an away favorite.
    assert plus < 0.5 and (1 - plus) > 0.7


def test_sim_nan_lines_fall_back_to_default():
    with_nan = _sim(home_rl_lines=np.array([np.nan, np.nan]))
    default = _sim()
    for a, b in zip(with_nan, default):
        assert abs(a.p_home_runline_cover - b.p_home_runline_cover) < 1e-9


def _row(pick, hs, as_, rl_line=None):
    return pd.Series({
        "status": "Final", "home_score": hs, "away_score": as_,
        "market": "runline", "pick": pick, "total_line": None,
        "rl_line": rl_line,
    })


def test_grade_legacy_rows_keep_old_frame():
    # Legacy (no rl_line): HOME means -1.5, AWAY means +1.5.
    assert _grade_row(_row("HOME", 5, 3)) == "win"     # won by 2
    assert _grade_row(_row("HOME", 4, 3)) == "loss"    # won by 1
    assert _grade_row(_row("AWAY", 4, 3)) == "win"     # away lost by 1
    assert _grade_row(_row("AWAY", 5, 3)) == "loss"    # away lost by 2


def test_grade_market_framed_rows():
    # Away is the market favorite: pick AWAY carries rl_line=-1.5.
    assert _grade_row(_row("AWAY", 3, 5, rl_line=-1.5)) == "win"   # away won by 2
    assert _grade_row(_row("AWAY", 3, 4, rl_line=-1.5)) == "loss"  # away won by 1
    # Home dog at +1.5.
    assert _grade_row(_row("HOME", 4, 5, rl_line=1.5)) == "win"    # lost by 1
    assert _grade_row(_row("HOME", 3, 5, rl_line=1.5)) == "loss"   # lost by 2
    # Integer line pushes.
    assert _grade_row(_row("HOME", 3, 5, rl_line=2.0)) == "push"


def test_best_market_prefers_fanduel():
    bms = [
        {"key": "draftkings", "markets": [{"key": "spreads", "outcomes": [{"name": "A"}]}]},
        {"key": "fanduel", "markets": [{"key": "spreads", "outcomes": [{"name": "B"}]}]},
    ]
    got = _best_market(bms, "spreads")
    assert got["bookmaker"] == "fanduel"


def test_best_market_falls_back_when_fanduel_silent():
    bms = [
        {"key": "pinnacle", "markets": [{"key": "spreads", "outcomes": [{"name": "A"}]}]},
        {"key": "fanduel", "markets": [{"key": "h2h", "outcomes": [{"name": "B"}]}]},
    ]
    got = _best_market(bms, "spreads")
    assert got["bookmaker"] == "pinnacle"
