"""Read the prediction journal and compute model-performance metrics.

The functions here all take a *graded* journal as input -- a DataFrame
produced by :func:`grade_journal` that has joined each prediction row
to the finalized outcome of the corresponding game. From there we
compute:

* :func:`season_summary` -- one-line headline per market (W-L-P, win
  rate, ROI proxy, Brier score, log-loss).
* :func:`rolling_accuracy` -- weekly/daily moving win rate so we can
  spot drift mid-season.
* :func:`calibration_bins` -- forecast vs realized win rate across
  probability buckets. The single most honest "is the model lying to
  itself?" check.
* :func:`slice_breakdown` -- accuracy by tier, by month, by venue, by
  team, etc.

All numbers ignore unfinalized games (``result == 'pending'``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date as date_cls
from typing import Any

import numpy as np
import pandas as pd

from mlb_model.data.warehouse import query
from mlb_model.journal.record import JOURNAL_PATH
from mlb_model.logging import get_logger

log = get_logger("journal.metrics")


# ROI assuming a flat -110 vig. Profit on a 1-unit winning bet = 0.909
# (you lay 1.10 to win 1.00). Loss on a 1-unit losing bet = -1.0. Push = 0.
_ROI_BY_RESULT = {"win": 0.909, "loss": -1.0, "push": 0.0}


def grade_journal(
    *,
    journal: pd.DataFrame | None = None,
    season: int | None = None,
    through: date_cls | None = None,
) -> pd.DataFrame:
    """Load the journal, keep the *latest* snapshot per (game, market, pick),
    and join finalized game outcomes.

    Parameters
    ----------
    journal
        Optional pre-loaded journal DataFrame. If None we read from
        :data:`JOURNAL_PATH`.
    season
        Optional season filter (4-digit year).
    through
        Optional upper-bound on ``game_date`` (inclusive). Useful for
        backtesting "what did we know as of date X?".
    """
    if journal is None:
        if not JOURNAL_PATH.exists():
            return pd.DataFrame()
        journal = pd.read_parquet(JOURNAL_PATH)
    if journal is None or journal.empty:
        return pd.DataFrame()

    df = journal.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    if season is not None:
        df = df[df["season"] == int(season)]
    if through is not None:
        df = df[df["game_date"] <= through]
    if df.empty:
        return df

    # For each (game, market, pick) we keep only the latest snapshot. That
    # is the model's final word before the game started.
    df["recorded_at"] = pd.to_datetime(df["recorded_at"])
    df = df.sort_values("recorded_at").drop_duplicates(
        subset=["game_pk", "market", "pick"],
        keep="last",
    )

    # Pull finalized scores from the warehouse.
    game_pks = df["game_pk"].astype(int).tolist()
    placeholders = ", ".join(["?"] * len(game_pks)) if game_pks else "NULL"
    outcomes = query(
        f"""
        SELECT game_pk, home_score, away_score, status
        FROM games
        WHERE game_pk IN ({placeholders})
        """,
        tuple(game_pks),
    )

    merged = df.merge(outcomes, on="game_pk", how="left")
    merged["result"] = merged.apply(_grade_row, axis=1)
    merged["roi_units"] = merged["result"].map(
        lambda r: _ROI_BY_RESULT.get(r, np.nan)
    )
    return merged


def _grade_row(row: pd.Series) -> str:
    """Return 'win' / 'loss' / 'push' / 'pending' for a single journal row."""
    status = row.get("status")
    if pd.isna(status) or status != "Final":
        return "pending"
    hs = row.get("home_score")
    as_ = row.get("away_score")
    if pd.isna(hs) or pd.isna(as_):
        return "pending"
    hs = float(hs)
    as_ = float(as_)
    market = str(row.get("market", "")).lower()
    pick = str(row.get("pick", "")).upper()
    total_line = row.get("total_line")

    if market == "moneyline":
        if hs == as_:
            return "push"
        home_won = hs > as_
        return "win" if (home_won and pick == "HOME") or (not home_won and pick == "AWAY") else "loss"

    if market == "runline":
        diff_home = hs - as_  # positive = home won by N
        # Grade against the line recorded with the pick. Legacy rows
        # (no rl_line) predate market framing and were always home -1.5 /
        # away +1.5, so default to that.
        line = row.get("rl_line")
        if line is None or pd.isna(line):
            line = -1.5 if pick == "HOME" else 1.5
        margin = diff_home if pick == "HOME" else -diff_home
        adj = margin + float(line)
        if adj == 0:
            return "push"
        return "win" if adj > 0 else "loss"

    if market == "total":
        if total_line is None or pd.isna(total_line):
            return "pending"
        total = hs + as_
        if total == float(total_line):
            return "push"
        is_over = total > float(total_line)
        return "win" if (is_over and pick == "OVER") or (not is_over and pick == "UNDER") else "loss"

    return "pending"


# --------------------------------------------------------------------------- #
# Summaries                                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class SeasonSummary:
    """One row in the headline table on the /season page."""

    season: int
    market: str
    n: int             # graded picks
    pending: int       # picks awaiting finals
    wins: int
    losses: int
    pushes: int
    win_rate: float    # wins / (wins + losses); pushes excluded
    roi_units: float   # cumulative profit/loss across all picks at -110
    brier: float       # mean squared error vs realized outcome (lower is better)
    log_loss: float    # mean log-loss vs realized outcome (lower is better)
    # CLV proxy. We don't track our actual ticket entry price, so the
    # best signal we have is the model-vs-market edge at the time the
    # prediction was recorded. If we consistently posted with positive
    # edge_pp on picks the model later won, that's strong signal we are
    # in fact beating closing lines. ``mean_edge_pp`` is averaged over
    # all graded rows with a known market_prob; ``clv_n`` is the
    # sample size that average is built from.
    mean_edge_pp: float
    clv_n: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def season_summary(graded: pd.DataFrame) -> list[SeasonSummary]:
    """One :class:`SeasonSummary` per (season, market) in the graded journal."""
    if graded is None or graded.empty:
        return []

    out: list[SeasonSummary] = []
    for (season, market), sub in graded.groupby(["season", "market"], dropna=True):
        wins = int((sub["result"] == "win").sum())
        losses = int((sub["result"] == "loss").sum())
        pushes = int((sub["result"] == "push").sum())
        pending = int((sub["result"] == "pending").sum())
        graded_n = wins + losses + pushes

        # Win rate excludes pushes (industry standard).
        denom = wins + losses
        win_rate = wins / denom if denom > 0 else float("nan")

        roi = float(pd.to_numeric(sub["roi_units"], errors="coerce").fillna(0.0).sum())

        # Brier + log-loss on the graded subset only.
        finalized = sub[sub["result"].isin({"win", "loss"})]
        if not finalized.empty:
            y = (finalized["result"] == "win").astype(float).to_numpy()
            p = pd.to_numeric(finalized["model_prob"], errors="coerce").fillna(0.5).to_numpy()
            p = np.clip(p, 1e-6, 1 - 1e-6)
            brier = float(np.mean((p - y) ** 2))
            log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            brier = float("nan")
            log_loss = float("nan")

        # CLV proxy -- mean posting-time edge across all graded picks
        # in this season+market that had a known market line. Pending
        # rows are excluded so this only reflects "edge on bets we
        # actually got an outcome for".
        # ``DataFrame.get('col')`` returns a scalar default when the
        # column is missing entirely (not a Series), so guard against
        # that before calling Series APIs.
        if "edge_pp" in finalized.columns:
            edge_series = pd.to_numeric(finalized["edge_pp"], errors="coerce").dropna()
        else:
            edge_series = pd.Series([], dtype=float)
        if not edge_series.empty:
            mean_edge_pp = float(edge_series.mean())
            clv_n = int(edge_series.shape[0])
        else:
            mean_edge_pp = float("nan")
            clv_n = 0

        out.append(
            SeasonSummary(
                season=int(season),
                market=str(market),
                n=graded_n,
                pending=pending,
                wins=wins,
                losses=losses,
                pushes=pushes,
                win_rate=win_rate,
                roi_units=roi,
                brier=brier,
                log_loss=log_loss,
                mean_edge_pp=mean_edge_pp,
                clv_n=clv_n,
            )
        )
    out.sort(key=lambda s: (s.season, s.market))
    return out


def rolling_accuracy(
    graded: pd.DataFrame,
    *,
    window_days: int = 30,
    market: str | None = None,
) -> pd.DataFrame:
    """Daily series of rolling win rate over the last ``window_days``.

    Returns DataFrame with columns ``[game_date, n, win_rate, roi]``.
    """
    if graded is None or graded.empty:
        return pd.DataFrame(columns=["game_date", "n", "win_rate", "roi"])

    sub = graded
    if market is not None:
        sub = sub[sub["market"] == market]
    sub = sub[sub["result"].isin({"win", "loss", "push"})].copy()
    if sub.empty:
        return pd.DataFrame(columns=["game_date", "n", "win_rate", "roi"])

    sub["game_date"] = pd.to_datetime(sub["game_date"])
    sub["is_win"] = (sub["result"] == "win").astype(float)
    sub["is_loss"] = (sub["result"] == "loss").astype(float)
    sub["is_push"] = (sub["result"] == "push").astype(float)
    sub["roi_value"] = pd.to_numeric(sub["roi_units"], errors="coerce").fillna(0.0)

    daily = (
        sub.groupby(sub["game_date"].dt.date)
        .agg(wins=("is_win", "sum"), losses=("is_loss", "sum"),
             pushes=("is_push", "sum"), roi=("roi_value", "sum"))
        .reset_index()
        .rename(columns={"game_date": "game_date"})
    )
    daily["game_date"] = pd.to_datetime(daily["game_date"])
    daily = daily.sort_values("game_date").set_index("game_date")

    rolled = daily.rolling(window=f"{window_days}D").sum()
    denom = (rolled["wins"] + rolled["losses"]).replace(0, np.nan)
    rolled["win_rate"] = rolled["wins"] / denom
    rolled["n"] = rolled["wins"] + rolled["losses"] + rolled["pushes"]
    rolled = rolled.reset_index()
    rolled["game_date"] = rolled["game_date"].dt.date
    return rolled[["game_date", "n", "win_rate", "roi"]]


def calibration_bins(
    graded: pd.DataFrame,
    *,
    n_bins: int = 10,
    market: str | None = None,
) -> pd.DataFrame:
    """Forecast vs realized win rate, bucketed by model probability.

    Returns columns ``[bucket_lo, bucket_hi, mid, n, forecast_mean, realized_rate, diff_pp]``.
    A well-calibrated model has ``forecast_mean ≈ realized_rate`` across
    every bucket.
    """
    if graded is None or graded.empty:
        return pd.DataFrame(columns=["bucket_lo", "bucket_hi", "mid", "n", "forecast_mean", "realized_rate", "diff_pp"])

    sub = graded
    if market is not None:
        sub = sub[sub["market"] == market]
    sub = sub[sub["result"].isin({"win", "loss"})].copy()
    if sub.empty:
        return pd.DataFrame(columns=["bucket_lo", "bucket_hi", "mid", "n", "forecast_mean", "realized_rate", "diff_pp"])

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    sub["bucket"] = pd.cut(sub["model_prob"].astype(float), bins=edges, include_lowest=True)
    sub["is_win"] = (sub["result"] == "win").astype(float)

    g = sub.groupby("bucket", observed=True).agg(
        n=("is_win", "size"),
        forecast_mean=("model_prob", "mean"),
        realized_rate=("is_win", "mean"),
    ).reset_index()
    # ``bucket`` after groupby+reset_index is a Categorical of pd.Interval
    # objects. Extract left/right via a python-level iteration to avoid
    # ``Categorical + Categorical`` arithmetic failures.
    intervals = list(g["bucket"])
    g["bucket_lo"] = [float(b.left) for b in intervals]
    g["bucket_hi"] = [float(b.right) for b in intervals]
    g["mid"] = (np.asarray(g["bucket_lo"], dtype=float) + np.asarray(g["bucket_hi"], dtype=float)) / 2.0
    g["diff_pp"] = (g["forecast_mean"].astype(float) - g["realized_rate"].astype(float)) * 100
    return g[["bucket_lo", "bucket_hi", "mid", "n", "forecast_mean", "realized_rate", "diff_pp"]]


def slice_breakdown(graded: pd.DataFrame, *, by: str = "tier") -> pd.DataFrame:
    """Win rate by an arbitrary categorical slice.

    Common dimensions: ``tier``, ``market``, ``home_team_abbr``,
    ``away_team_abbr``, or any column you stuff into the journal.
    """
    if graded is None or graded.empty or by not in graded.columns:
        return pd.DataFrame(columns=[by, "n", "win_rate", "roi"])

    sub = graded[graded["result"].isin({"win", "loss", "push"})].copy()
    if sub.empty:
        return pd.DataFrame(columns=[by, "n", "win_rate", "roi"])

    sub["is_win"] = (sub["result"] == "win").astype(float)
    sub["is_loss"] = (sub["result"] == "loss").astype(float)
    sub["is_push"] = (sub["result"] == "push").astype(float)
    sub["roi_value"] = pd.to_numeric(sub["roi_units"], errors="coerce").fillna(0.0)

    g = sub.groupby(by, dropna=False).agg(
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        pushes=("is_push", "sum"),
        roi=("roi_value", "sum"),
    ).reset_index()
    denom = g["wins"] + g["losses"]
    g["win_rate"] = np.where(denom > 0, g["wins"] / denom.replace(0, np.nan), np.nan)
    g["n"] = g["wins"] + g["losses"] + g["pushes"]
    g = g.sort_values("n", ascending=False)
    return g[[by, "n", "win_rate", "roi", "wins", "losses", "pushes"]]
