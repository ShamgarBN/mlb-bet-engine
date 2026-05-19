"""Stage B: Monte Carlo game simulation -> ML / RL / O/U probabilities.

Given each team's expected runs distribution (mean and std), we draw N
simulated final scores per game and read off the empirical probabilities:

  P(home wins)              -- moneyline pick
  P(home covers -1.5 RL)    -- run line pick
  P(home + away > total)    -- over/under pick

Why simulation rather than a direct classifier? Three reasons:
  1) The ML, RL, and O/U markets are *coherent* -- they're all derived
     from the same joint run distribution. Simulation enforces that
     coherence; three independent classifiers would not.
  2) We get full uncertainty estimates "for free" (interval widths).
  3) Ties (regulation) are handled correctly by the negative-binomial
     draw shape -- baseball doesn't have ties in box scores, but the
     distribution has to handle the bunching at integer counts.

We use a **negative binomial** rather than Gaussian for the per-team draws
because runs are non-negative integer counts with overdispersion. The
parameters are derived from the (mean, std) predicted by the run models.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mlb_model.config import settings


@dataclass
class GamePrediction:
    """Per-game probabilities derived from Monte Carlo simulation."""

    game_pk: int
    pred_home_runs: float
    pred_away_runs: float
    p_home_win: float
    p_home_runline_cover: float   # home -1.5
    p_total_over: float
    total_line: float | None
    runs_std_home: float
    runs_std_away: float

    def confidence(self, *, market: str) -> float:
        """Distance from 0.5 -- larger means more conviction."""
        match market:
            case "moneyline":
                return abs(self.p_home_win - 0.5) * 2.0
            case "runline":
                return abs(self.p_home_runline_cover - 0.5) * 2.0
            case "total":
                return abs(self.p_total_over - 0.5) * 2.0
            case _:
                raise ValueError(market)


def _negbin_draw(
    mean: np.ndarray,
    std: np.ndarray,
    n_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw from a negative binomial parameterized by (mean, std).

    NegBin variance = mean + mean^2 / r, where r is the dispersion.
    Solve for r given target variance: r = mean^2 / (variance - mean).
    """
    mean = np.clip(mean, 0.1, None)
    var = np.clip(std**2, mean + 0.1, None)  # ensure overdispersion
    r = mean**2 / (var - mean)
    p = r / (r + mean)
    # numpy's negative_binomial takes (n_successes_until_stop, prob_success).
    # The mean of NB(n, p) is n*(1-p)/p; substituting our r and p above
    # yields the desired mean, std.
    draws = rng.negative_binomial(n=np.broadcast_to(r, (n_sims, len(mean))), p=np.broadcast_to(p, (n_sims, len(mean))))
    return draws


def simulate_games(
    game_pks: np.ndarray,
    pred_home: tuple[np.ndarray, np.ndarray],
    pred_away: tuple[np.ndarray, np.ndarray],
    *,
    total_lines: np.ndarray | None = None,
    runline: float = 1.5,
    n_sims: int = 50_000,
    seed: int | None = None,
) -> list[GamePrediction]:
    """Run Monte Carlo simulation for a batch of games.

    Args:
        game_pks: Game id per row.
        pred_home: (means, stds) for home runs.
        pred_away: (means, stds) for away runs.
        total_lines: O/U total per game (for P(over)). NaN-safe.
        runline: Spread magnitude (default 1.5).
        n_sims: Monte Carlo draws per game.
        seed: RNG seed (defaults to settings.random_seed).
    """
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)
    home_mean, home_std = pred_home
    away_mean, away_std = pred_away

    n_games = len(game_pks)
    home_draws = _negbin_draw(home_mean, home_std, n_sims, rng)   # shape (n_sims, n_games)
    away_draws = _negbin_draw(away_mean, away_std, n_sims, rng)

    # Tie-break: MLB games can't end in a tie -- in extra-innings the
    # team that scores first in the next half-inning wins. We approximate
    # this with a 50/50 coin flip on simulated ties.
    ties = home_draws == away_draws
    if ties.any():
        coin = rng.integers(0, 2, size=ties.shape, dtype=np.int8)
        home_draws = home_draws + ties * coin
        # Mark the coin-flipped ties as home wins now if coin=1, else away
        # by adding 1 to home runs; this is equivalent to "home wins half".

    home_win = (home_draws > away_draws).mean(axis=0)
    home_rl_cover = (home_draws - away_draws > runline).mean(axis=0)
    totals_sim = home_draws + away_draws

    predictions: list[GamePrediction] = []
    for i in range(n_games):
        total_line = float(total_lines[i]) if total_lines is not None and not np.isnan(total_lines[i]) else None
        p_over = (totals_sim[:, i] > total_line).mean() if total_line is not None else float("nan")
        predictions.append(
            GamePrediction(
                game_pk=int(game_pks[i]),
                pred_home_runs=float(home_mean[i]),
                pred_away_runs=float(away_mean[i]),
                p_home_win=float(home_win[i]),
                p_home_runline_cover=float(home_rl_cover[i]),
                p_total_over=float(p_over),
                total_line=total_line,
                runs_std_home=float(home_std[i]),
                runs_std_away=float(away_std[i]),
            )
        )
    return predictions
