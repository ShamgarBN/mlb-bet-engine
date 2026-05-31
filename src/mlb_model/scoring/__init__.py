"""Per-hitter prop scoring (hit / HR / total bases / strikeout).

Anchored 0-10 delta models that read like the public ``mlb-scout`` algorithms
but run server-side on our warehouse and degrade gracefully when any input
is missing. See :mod:`mlb_model.scoring.hitter` for the implementations.
"""

from mlb_model.scoring.hitter import (
    BatterInputs,
    PitcherInputs,
    Scores,
    score_matchup,
    score_hit,
    score_hr,
    score_tb,
    score_k,
)

__all__ = [
    "BatterInputs",
    "PitcherInputs",
    "Scores",
    "score_matchup",
    "score_hit",
    "score_hr",
    "score_tb",
    "score_k",
]
