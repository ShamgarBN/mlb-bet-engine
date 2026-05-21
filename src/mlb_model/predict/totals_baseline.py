"""League-average runs baseline used when no closing total is posted.

Today's slate frequently has no closing total in the warehouse -- our
historical odds dataset (SBR JSON) only runs through 2025 and we don't
yet scrape live odds. Without a baseline, the over/under market becomes
invisible to the user. That's a bad outcome: the run-distribution
prediction is still meaningful and the user can still compare to a
sensible reference line.

This module computes a *backwards-looking* league-average total runs per
game from completed finals so the baseline reflects the current
scoring environment (the 2026 ball is not the 2019 ball). The number
falls back to a hard 8.5 if the warehouse has no finals for the season.
"""

from __future__ import annotations

from functools import lru_cache

from mlb_model.data.warehouse import query
from mlb_model.logging import get_logger

log = get_logger("predict.totals_baseline")

# League-wide fallback when we have no warehouse data at all.
HARD_FALLBACK = 8.5


@lru_cache(maxsize=64)
def league_avg_total(season: int) -> float:
    """Mean (home + away) score across finalized games in ``season``.

    Cached because the value is stable for the day and we may query it
    once per game in a slate. Cache is cleared at process restart.
    """
    df = query(
        """
        SELECT AVG(home_score + away_score) AS avg_total,
               COUNT(*) AS n
        FROM games
        WHERE season = ? AND status = 'Final'
          AND home_score IS NOT NULL AND away_score IS NOT NULL
        """,
        (int(season),),
    )
    if df.empty:
        return HARD_FALLBACK
    n = int(df.iloc[0]["n"] or 0)
    if n < 50:
        # Early in the season we don't trust the sample. Blend toward the
        # prior season once we have one finalized year of data.
        prior_df = query(
            """
            SELECT AVG(home_score + away_score) AS avg_total, COUNT(*) AS n
            FROM games WHERE season = ? AND status = 'Final'
              AND home_score IS NOT NULL AND away_score IS NOT NULL
            """,
            (int(season) - 1,),
        )
        prior_n = int(prior_df.iloc[0]["n"] or 0) if not prior_df.empty else 0
        if prior_n > 0:
            cur = float(df.iloc[0]["avg_total"] or HARD_FALLBACK)
            prev = float(prior_df.iloc[0]["avg_total"] or HARD_FALLBACK)
            # Weighted average: prior season carries until we have ~500 games.
            w_prior = max(0.0, 1.0 - n / 500.0)
            blended = w_prior * prev + (1 - w_prior) * cur
            return round(blended * 2.0) / 2.0  # round to nearest 0.5
        return HARD_FALLBACK
    avg = float(df.iloc[0]["avg_total"] or HARD_FALLBACK)
    # Round to nearest 0.5 so it looks like a real posted total.
    return round(avg * 2.0) / 2.0
