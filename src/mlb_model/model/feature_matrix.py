"""Convert the wide feature table into numeric matrices for modeling.

This module isolates the boring-but-critical work of:
  * Selecting numeric columns
  * Encoding categoricals (handedness, dome flag, etc.)
  * Building (X_home, X_away, X_game) matrices for the two run models
  * Handling missing values with column medians (computed train-side only)

Crucially, the *training-data-derived imputation values* are returned as
part of the fitted preprocessor so backtesting and prediction time use the
same values -- no leakage from test data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column conventions
# ---------------------------------------------------------------------------

ID_COLS = [
    "game_pk", "game_date", "season",
    "home_team_id", "away_team_id",
    "home_team_abbr", "away_team_abbr",
    "venue_id", "home_sp_id", "away_sp_id",
]

TARGET_COLS = [
    "target_home_win",
    "target_home_score",
    "target_away_score",
    "target_total_runs",
]

CATEGORICAL_COLS = [
    "home_sp_throws",
    "away_sp_throws",
    "is_dome",
]

# Columns that look game-level but should never be features (textual labels).
NON_FEATURE_COLS = [
    "ump_name",
]


@dataclass
class FeatureSpec:
    """Schema-aware view of the assembled feature table."""

    home_feature_cols: list[str] = field(default_factory=list)
    away_feature_cols: list[str] = field(default_factory=list)
    game_feature_cols: list[str] = field(default_factory=list)
    medians: dict[str, float] = field(default_factory=dict)

    def all_cols(self) -> list[str]:
        return self.home_feature_cols + self.away_feature_cols + self.game_feature_cols


def _classify_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Partition columns into home / away / game-level feature lists.

    Excludes:
      * ID and target columns
      * Explicitly non-feature text columns (NON_FEATURE_COLS)
      * Object-dtype columns *except* the known categoricals
    """
    drop = set(ID_COLS + TARGET_COLS + NON_FEATURE_COLS)
    home_cols, away_cols, game_cols = [], [], []
    for c in df.columns:
        if c in drop:
            continue
        col = df[c]
        is_numeric = pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)
        is_known_cat = c in CATEGORICAL_COLS
        # Some columns end up object-dtype because they hold mixed
        # None/float (e.g. market columns for seasons without odds). Treat
        # those as numeric features so they don't silently drop out.
        is_numeric_object = (
            not is_numeric
            and not is_known_cat
            and col.dtype == object
            and pd.to_numeric(col, errors="coerce").notna().any()
        )
        if not (is_numeric or is_known_cat or is_numeric_object):
            continue
        if c.startswith("home_"):
            home_cols.append(c)
        elif c.startswith("away_"):
            away_cols.append(c)
        else:
            game_cols.append(c)
    return home_cols, away_cols, game_cols


def fit_spec(df: pd.DataFrame) -> FeatureSpec:
    """Fit a FeatureSpec (medians for imputation) on training data."""
    home_cols, away_cols, game_cols = _classify_columns(df)
    all_numeric = [
        c for c in home_cols + away_cols + game_cols
        if c not in CATEGORICAL_COLS  # categoricals are one-hot'd, not imputed
    ]
    medians: dict[str, float] = {}
    for c in all_numeric:
        col = df[c]
        # Coerce safely; non-numeric values become NaN and are then ignored
        # by median(), so object-dtype columns of mixed None/float work.
        col_f = pd.to_numeric(col, errors="coerce")
        if not col_f.notna().any():
            medians[c] = 0.0
            continue
        m = float(col_f.median())
        if not (m == m) or m in (float("inf"), float("-inf")):
            m = 0.0
        medians[c] = m
    return FeatureSpec(
        home_feature_cols=home_cols,
        away_feature_cols=away_cols,
        game_feature_cols=game_cols,
        medians=medians,
    )


def transform(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Apply the spec's imputation + one-hot encoding for categoricals.

    Object-dtype columns that hold mixed numeric/None values (e.g. odds
    columns that are all None for a season without market data) are
    coerced to float64 before imputation. After this call every column
    listed in ``spec`` is finite float64.
    """
    out = df.copy()
    for c in spec.all_cols():
        if c in CATEGORICAL_COLS:
            continue
        if c not in out.columns:
            out[c] = spec.medians.get(c, 0.0)
            continue
        fill = spec.medians.get(c, 0.0)
        col = out[c]
        if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
            out[c] = col.astype("float64").fillna(fill)
        else:
            # object / mixed -- coerce safely and fill any non-numeric.
            out[c] = pd.to_numeric(col, errors="coerce").astype("float64").fillna(fill)

    # One-hot the small set of known categoricals.
    for c in CATEGORICAL_COLS:
        if c in out.columns:
            dummies = pd.get_dummies(out[c].astype(str), prefix=c, dtype=float)
            out = pd.concat([out.drop(columns=[c]), dummies], axis=1)
    return out


def build_runs_matrix(features: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build the X matrix for predicting *home runs*. Returns (X, mask, columns).

    The matrix concatenates:
      - all home_* feature cols (offense + SP + form + schedule)
      - all away_* feature cols (acting as opponent defense / SP)
      - game-level features (park, weather, ump, market)

    For the *away runs* model we just swap home/away by transposition --
    see ``build_runs_matrix_away``.
    """
    transformed = transform(features, spec)

    feature_cols = (
        spec.home_feature_cols + spec.away_feature_cols + spec.game_feature_cols
    )
    # After one-hot, some categorical columns turn into multiple dummy columns;
    # include any new dummy columns whose prefix matches a categorical.
    extra_dummy_cols = [
        c for c in transformed.columns
        if any(c.startswith(prefix + "_") for prefix in CATEGORICAL_COLS)
        and c not in feature_cols
    ]
    feature_cols = feature_cols + extra_dummy_cols

    feature_cols = [c for c in feature_cols if c in transformed.columns]
    # Drop the raw categorical columns themselves if they remain
    feature_cols = [c for c in feature_cols if c not in CATEGORICAL_COLS]

    X = transformed[feature_cols].to_numpy(dtype=np.float64)
    mask = np.isfinite(X).all(axis=1)
    return X, mask, feature_cols


def build_runs_matrix_away(features: pd.DataFrame, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Matrix for predicting *away runs* (mirror of home).

    We physically swap the home_* and away_* prefixed columns so the model
    that learned "predict runs scored by team-on-offense given their stats
    plus opposing pitcher stats" applies symmetrically.
    """
    mirrored = features.copy()
    rename: dict[str, str] = {}
    for c in features.columns:
        if c.startswith("home_"):
            rename[c] = "away_" + c[5:]
        elif c.startswith("away_"):
            rename[c] = "home_" + c[5:]
    mirrored = mirrored.rename(columns=rename)
    return build_runs_matrix(mirrored, spec)
