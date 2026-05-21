"""Market-derived features from historical odds.

Two ideas:

  1) The closing line is itself the single best forecast feature available.
     Including it lets the model learn to deviate from market consensus only
     when it has reason to. Models that ignore the line under-perform on
     out-of-sample data.

  2) Line *movement* (open to close) carries sharp-money signal. When the
     line moves significantly against the public side, that's typically
     informed money.

This module computes:
  * market_ml_home_close_prob -- implied probability after de-vigging.
  * market_ml_movement       -- (close - open) in cents.
  * market_total_close       -- closing total line.
  * market_runline_home_price -- price on -1.5/+1.5 home runline.
"""

from __future__ import annotations

import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("features.market")


def american_to_implied(price: float) -> float | None:
    """Convert American odds to implied probability (no vig removed)."""
    if pd.isna(price) or price == 0:
        return None
    p = float(price)
    if p > 0:
        return 100.0 / (p + 100.0)
    return -p / (-p + 100.0)


def remove_vig(p_home: float, p_away: float) -> tuple[float, float] | tuple[None, None]:
    """Power-method-free de-vig: renormalize to sum to 1.

    Simple but works fine for moneyline. More rigorous methods exist
    (Shin, log-odds, power) but offer marginal improvements for in-sample
    games.
    """
    if pd.isna(p_home) or pd.isna(p_away):
        return None, None
    s = p_home + p_away
    if s <= 0:
        return None, None
    return p_home / s, p_away / s


def build_market_features(season_start: int, season_end: int) -> pd.DataFrame:
    """One row per game with closing-line-derived features.

    Joins odds_history (date + team_abbr key) to games.
    """
    # Prefer the newer per-book SBR dataset (book='sbr_consensus'); fall
    # back to the older xlsx archive (book='sbro_consensus') when no
    # newer row exists. The window function picks one row per game.
    sql = """
        WITH ranked_odds AS (
            SELECT o.*,
                   CASE WHEN o.book = 'sbr_consensus' THEN 0
                        WHEN o.book = 'sbro_consensus' THEN 1
                        ELSE 2 END AS book_priority,
                   ROW_NUMBER() OVER (
                       PARTITION BY o.game_date, o.home_team_abbr, o.away_team_abbr
                       ORDER BY CASE WHEN o.book = 'sbr_consensus' THEN 0
                                     WHEN o.book = 'sbro_consensus' THEN 1
                                     ELSE 2 END
                   ) AS rn
            FROM odds_history o
        )
        SELECT
            g.game_pk,
            g.game_date,
            g.season,
            o.ml_open_home,
            o.ml_open_away,
            o.ml_close_home,
            o.ml_close_away,
            o.rl_close_home,
            o.rl_close_home_price,
            o.rl_close_away_price,
            o.total_close,
            o.total_close_over,
            o.total_close_under
        FROM games g
        LEFT JOIN ranked_odds o
          ON o.game_date = g.game_date
         AND o.home_team_abbr = g.home_team_abbr
         AND o.away_team_abbr = g.away_team_abbr
         AND o.rn = 1
        WHERE g.season BETWEEN ? AND ?
    """
    df = query(sql, params=(season_start, season_end))
    if df.empty:
        return df

    df["ml_close_home_implied_raw"] = df["ml_close_home"].apply(american_to_implied)
    df["ml_close_away_implied_raw"] = df["ml_close_away"].apply(american_to_implied)

    devig = df.apply(
        lambda r: remove_vig(r["ml_close_home_implied_raw"], r["ml_close_away_implied_raw"]),
        axis=1,
        result_type="expand",
    )
    df["market_ml_home_close_prob"] = devig[0]
    df["market_ml_away_close_prob"] = devig[1]

    # ------------------------------------------------------------------
    # Runline implied probabilities (de-vigged). The UI uses these to
    # show "model vs market" edge on runline picks; the features
    # themselves also help the model when a sharp runline price
    # disagrees with the moneyline.
    # ------------------------------------------------------------------
    df["rl_close_home_implied_raw"] = df["rl_close_home_price"].apply(american_to_implied)
    df["rl_close_away_implied_raw"] = df["rl_close_away_price"].apply(american_to_implied)
    rl_devig = df.apply(
        lambda r: remove_vig(r["rl_close_home_implied_raw"], r["rl_close_away_implied_raw"]),
        axis=1,
        result_type="expand",
    )
    df["market_rl_home_close_prob"] = rl_devig[0]
    df["market_rl_away_close_prob"] = rl_devig[1]

    # ------------------------------------------------------------------
    # Total over/under implied probabilities (de-vigged).
    # ------------------------------------------------------------------
    df["ou_close_over_implied_raw"]  = df["total_close_over"].apply(american_to_implied)
    df["ou_close_under_implied_raw"] = df["total_close_under"].apply(american_to_implied)
    ou_devig = df.apply(
        lambda r: remove_vig(r["ou_close_over_implied_raw"], r["ou_close_under_implied_raw"]),
        axis=1,
        result_type="expand",
    )
    df["market_total_over_close_prob"]  = ou_devig[0]
    df["market_total_under_close_prob"] = ou_devig[1]

    df["market_ml_movement_home"] = df["ml_close_home"].fillna(0) - df["ml_open_home"].fillna(0)
    df["market_rl_home_close_price"] = df["rl_close_home_price"]
    df["market_total_close"] = df["total_close"]

    keep = [
        "game_pk", "game_date", "season",
        "market_ml_home_close_prob", "market_ml_away_close_prob",
        "market_rl_home_close_prob", "market_rl_away_close_prob",
        "market_total_over_close_prob", "market_total_under_close_prob",
        "market_ml_movement_home", "market_rl_home_close_price",
        "market_total_close",
    ]
    out = df[keep].copy()
    log.info("market.features.built", rows=len(out))
    return out
