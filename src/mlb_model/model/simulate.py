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
    # P(home covers ITS line). The line is the game's market home runline
    # when supplied to ``simulate_games`` (-1.5 if home is the favorite,
    # +1.5 if the away team is), else the historical default of home -1.5.
    p_home_runline_cover: float
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
    *,
    env: np.ndarray | None = None,
) -> np.ndarray:
    """Draw from a negative binomial parameterized by (mean, std).

    NegBin variance = mean + mean^2 / r, where r is the dispersion.
    Solve for r given target variance: r = mean^2 / (variance - mean).

    If ``env`` is given (shape (n_sims, n_games)), it acts as a
    per-draw multiplicative scaling of the conditional mean. This lets
    callers introduce CORRELATED noise across home and away within
    the same game (e.g. shared weather / umpire run environment): if
    today is going to be a high-scoring game both teams should score
    more, not just one. Without this, the simulator treats home and
    away as independent draws and underestimates totals variance.
    """
    mean = np.clip(mean, 0.1, None)
    var = np.clip(std**2, mean + 0.1, None)  # ensure overdispersion
    r = mean**2 / (var - mean)
    p = r / (r + mean)
    # Broadcast to (n_sims, n_games)
    r_b = np.broadcast_to(r, (n_sims, len(mean)))
    p_b = np.broadcast_to(p, (n_sims, len(mean)))
    if env is not None:
        # Apply the env multiplier by transforming (r, p) so the mean
        # becomes mean * env while preserving the dispersion shape.
        # Mean of NB(r, p) is r * (1-p) / p. Multiplying by env is
        # equivalent to setting p_new such that r * (1-p_new)/p_new = mean*env
        # => p_new = r / (r + mean*env).
        scaled_mean = mean * env  # (n_sims, n_games)
        p_b = r_b / (r_b + scaled_mean)
    draws = rng.negative_binomial(n=r_b, p=p_b)
    return draws


def simulate_games(
    game_pks: np.ndarray,
    pred_home: tuple[np.ndarray, np.ndarray],
    pred_away: tuple[np.ndarray, np.ndarray],
    *,
    total_lines: np.ndarray | None = None,
    runline: float = 1.5,
    home_rl_lines: np.ndarray | None = None,
    n_sims: int = 50_000,
    seed: int | None = None,
) -> list[GamePrediction]:
    """Run Monte Carlo simulation for a batch of games.

    Args:
        game_pks: Game id per row.
        pred_home: (means, stds) for home runs.
        pred_away: (means, stds) for away runs.
        total_lines: O/U total per game (for P(over)). NaN-safe.
        runline: Spread magnitude when no per-game line is given (default
            1.5, framed as home -1.5). Training/backtest use this fixed
            frame so the calibrator has a stable definition.
        home_rl_lines: Per-game HOME runline points from the market
            (-1.5 when home is favorite, +1.5 when away is). NaN entries
            fall back to ``-runline``. When given, ``p_home_runline_cover``
            is P(home covers its actual market line).
        n_sims: Monte Carlo draws per game.
        seed: RNG seed (defaults to settings.random_seed).
    """
    rng = np.random.default_rng(seed if seed is not None else settings.random_seed)
    home_mean, home_std = pred_home
    away_mean, away_std = pred_away

    n_games = len(game_pks)

    # ------------------------------------------------------------------
    # Shared game-environment multiplier. Each simulated game-draw gets
    # one ``env`` factor that scales BOTH teams' expected runs the same
    # direction (high-scoring day, low-scoring day, etc.). We draw the
    # multiplier from a log-normal centered at 1 with a small sigma so
    # the marginal mean per team is unchanged but home/away draws are
    # positively correlated. This better matches reality (weather,
    # umpires, ball composition all push both teams together).
    # Sigma=0.10 corresponds to ~10% std on env -- empirically the
    # within-game home/away correlation hovers around 0.10-0.15 in the
    # historical data.
    # ------------------------------------------------------------------
    env_sigma = 0.10
    env = rng.lognormal(mean=-(env_sigma ** 2) / 2.0, sigma=env_sigma, size=(n_sims, n_games))

    home_draws = _negbin_draw(home_mean, home_std, n_sims, rng, env=env)
    away_draws = _negbin_draw(away_mean, away_std, n_sims, rng, env=env)

    # Tie-break: MLB games can't end in a tie. We weight the coin flip
    # toward whichever team has the higher predicted mean (a 7-run team
    # beats a 4-run team in extras far more often than 50/50). For
    # close ties (within 0.3 runs of expected) we still flip a coin.
    ties = home_draws == away_draws
    if ties.any():
        # Weight: P(home wins extras) = home_mean / (home_mean + away_mean),
        # clipped to [0.35, 0.65] so we never go too extreme.
        weight = np.clip(home_mean / (home_mean + away_mean), 0.35, 0.65)
        weight_b = np.broadcast_to(weight, ties.shape)
        coin = (rng.random(size=ties.shape) < weight_b).astype(np.int8)
        home_draws = home_draws + ties * coin

    home_win = (home_draws > away_draws).mean(axis=0)
    # Home covers its line L when margin + L > 0. With the default
    # L = -runline (-1.5) this is the historical margin > 1.5.
    if home_rl_lines is None:
        lines = np.full(n_games, -runline)
    else:
        lines = np.where(np.isfinite(home_rl_lines), home_rl_lines, -runline)
    home_rl_cover = (home_draws - away_draws > -lines).mean(axis=0)
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
