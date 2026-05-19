"""MLB prediction and betting model.

A free-data, walk-forward backtested model covering moneyline, run line, and
over/under markets. See README.md for performance targets and disclaimers.
"""

from __future__ import annotations

__version__ = "0.1.0"

from mlb_model.config import settings

__all__ = ["__version__", "settings"]
