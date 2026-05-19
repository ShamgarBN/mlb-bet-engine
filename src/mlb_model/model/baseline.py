"""Bayesian baseline: prior over team strength and league run environment.

Use case: even with sparse features (early-season games, missing Statcast,
unfamiliar starters), we want the model to fall back to a *sensible* prior
rather than a garbage extrapolation. This module provides:

  * League-average runs per team-game (typically 4.3-4.7 in modern era)
  * Per-team prior strength derived from prior-season Pythagorean record
  * Pitcher prior FIP (with a population mean fallback)

Used as a feature alongside the learned model output -- not as a replacement.
The LightGBM model can then learn the residual on top of the prior.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("model.baseline")


def league_avg_runs(season: int, lookback: int = 3) -> float:
    """Return the league average runs per team-game from the prior seasons."""
    start = max(2010, season - lookback)
    end = season - 1
    df = query(
        """
        SELECT AVG(b.runs::DOUBLE) AS avg_runs
        FROM team_boxscores b
        JOIN games g ON g.game_pk = b.game_pk
        WHERE g.season BETWEEN ? AND ? AND g.status = 'Final'
        """,
        params=(start, end),
    )
    if df.empty or df.iloc[0]["avg_runs"] is None:
        return 4.5
    return float(df.iloc[0]["avg_runs"])


def team_priors(season: int, lookback: int = 1) -> pd.DataFrame:
    """Per-team prior runs-scored and runs-allowed from prior seasons.

    Returns DataFrame with columns: team_id, prior_rs, prior_ra.
    """
    start = max(2010, season - lookback)
    end = season - 1
    df = query(
        """
        SELECT
            b.team_id,
            AVG(b.runs::DOUBLE)   AS prior_rs,
            AVG(opp.runs::DOUBLE) AS prior_ra
        FROM team_boxscores b
        JOIN team_boxscores opp ON opp.game_pk = b.game_pk AND opp.team_id != b.team_id
        JOIN games g ON g.game_pk = b.game_pk
        WHERE g.season BETWEEN ? AND ? AND g.status = 'Final'
        GROUP BY b.team_id
        """,
        params=(start, end),
    )
    log.info("baseline.team_priors", season=season, n_teams=len(df))
    return df


def shrunk_team_form(
    season: int,
    team_form_features: pd.DataFrame,
    *,
    prior_weight: float = 30.0,
) -> pd.DataFrame:
    """Shrink rolling team-form stats toward prior-season averages.

    For each team-game row, blend the in-season rolling mean with the prior
    using a precision-weighted average:

        shrunk = (n * in_season + prior_weight * prior) / (n + prior_weight)

    Where n approximates the games in the rolling window. ``prior_weight``
    of 30 means it takes about 30 actual games before the in-season signal
    dominates -- a reasonable starting point.

    This is the cold-start fix that lets the model produce sensible
    predictions in April.
    """
    priors = team_priors(season).set_index("team_id")
    out = team_form_features.copy()
    if "runs_scored_r30d" in out.columns and not priors.empty:
        out = out.merge(
            priors.rename(columns={"prior_rs": "prior_rs_team"}),
            left_on="team_id", right_index=True, how="left",
        )
        out["runs_scored_r30d_shrunk"] = (
            (30.0 * out["runs_scored_r30d"].fillna(out["prior_rs_team"]) + prior_weight * out["prior_rs_team"])
            / (30.0 + prior_weight)
        )
    return out
