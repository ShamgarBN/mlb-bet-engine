"""Service layer for the desktop app.

Wraps the existing prediction pipeline with caching, lightweight result
shaping for templates, and a small picks-log persistence layer. Keep all
business logic here; the route module should be thin.

Caching strategy
----------------
The prediction pipeline takes ~3-10 seconds (feature build + boosted
trees + simulation). To keep the UI snappy we cache per-date prediction
results to disk under ``cache/predictions/<YYYY-MM-DD>.parquet``. The
refresh endpoint deletes the cache for the target date and recomputes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date as date_cls, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("app.services")


# ---------------------------------------------------------------------------
# Predictions cache
# ---------------------------------------------------------------------------

PRED_CACHE_DIR = settings.cache_dir / "predictions"


def _cache_path(target: date_cls) -> Path:
    PRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return PRED_CACHE_DIR / f"{target.isoformat()}.parquet"


def load_cached_predictions(target: date_cls) -> pd.DataFrame | None:
    """Return cached prediction DataFrame for ``target`` if any."""
    path = _cache_path(target)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df
    except Exception as exc:  # noqa: BLE001 -- defensive cache read
        log.warning("predictions.cache.read_failed", path=str(path), error=str(exc))
        return None


def save_predictions(target: date_cls, df: pd.DataFrame) -> Path:
    """Persist a freshly computed prediction DataFrame."""
    path = _cache_path(target)
    out = df.copy()
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.date
    out.to_parquet(path, index=False)
    return path


def compute_predictions(target: date_cls, *, refresh: bool) -> pd.DataFrame:
    """Run the model for ``target`` (optionally refreshing data first)."""
    from mlb_model.predict.daily import predict_for_date

    df = predict_for_date(target, refresh_data=refresh)
    if df.empty:
        return df
    save_predictions(target, df)
    return df


def get_predictions(
    target: date_cls,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """High-level entrypoint used by the routes."""
    if not refresh:
        cached = load_cached_predictions(target)
        if cached is not None and not cached.empty:
            return cached
    return compute_predictions(target, refresh=refresh)


# ---------------------------------------------------------------------------
# Pick shaping for the UI
# ---------------------------------------------------------------------------


@dataclass
class PickRow:
    """One row per game per market (ML/RL/OU) for the dashboard."""

    game_pk: int
    game_date: date_cls
    away_team_abbr: str
    home_team_abbr: str
    market: str
    pick: str
    pick_long: str
    model_prob: float
    market_prob: float | None
    edge_pp: float | None
    confidence: float
    tier: str
    pred_home_runs: float
    pred_away_runs: float
    p_home_win: float
    p_home_runline_cover: float
    p_total_over: float | None
    total_line: float | None
    # "market" if we used a real closing line, "baseline" if we fell
    # back to the league-average. The UI shows this so the user can
    # weight totals picks accordingly.
    total_line_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        d["game_date"] = self.game_date.isoformat()
        return d


def _confidence_tier(conf: float) -> str:
    """Bucket confidence into a tier label."""
    if conf >= 0.70:
        return "premium"
    if conf >= 0.50:
        return "strong"
    if conf >= 0.30:
        return "edge"
    if conf >= 0.15:
        return "lean"
    return "pass"


def shape_picks(df: pd.DataFrame) -> list[PickRow]:
    """Convert the raw prediction DataFrame to UI rows.

    Each game contributes three rows (ML / RL / OU) so the table can be
    filtered by market and sorted by confidence within a market.
    """
    if df.empty:
        return []

    rows: list[PickRow] = []
    for _, r in df.iterrows():
        p_ml = float(r["p_home_win"])
        p_rl = float(r["p_home_runline_cover"])
        p_ou = float(r["p_total_over"]) if pd.notna(r.get("p_total_over")) else None
        market_ml = (
            float(r["market_ml_home_close_prob"])
            if pd.notna(r.get("market_ml_home_close_prob"))
            else None
        )
        # Effective line = market line if available, else league-avg baseline.
        # ``effective_total_line`` is set by predict_for_date; older cached
        # parquet files predating that column will still have ``total_line``.
        if "effective_total_line" in r.index and pd.notna(r["effective_total_line"]):
            total_line = float(r["effective_total_line"])
        elif pd.notna(r.get("market_total_close")):
            total_line = float(r["market_total_close"])
        elif pd.notna(r.get("total_line")):
            total_line = float(r["total_line"])
        else:
            total_line = None
        total_line_source = (
            str(r["total_line_source"]) if pd.notna(r.get("total_line_source"))
            else ("market" if pd.notna(r.get("market_total_close")) else "baseline")
        )

        away, home = str(r["away_team_abbr"]), str(r["home_team_abbr"])

        game_date_val = (
            r["game_date"] if isinstance(r["game_date"], date_cls)
            else pd.to_datetime(r["game_date"]).date()
        )

        # Moneyline
        pick_side, pick_long, prob = _ml_pick(p_ml, away, home)
        edge = (prob - market_ml) * 100 if market_ml is not None else None
        rows.append(
            PickRow(
                game_pk=int(r["game_pk"]),
                game_date=game_date_val,
                away_team_abbr=away, home_team_abbr=home,
                market="moneyline", pick=pick_side, pick_long=pick_long,
                model_prob=prob,
                market_prob=market_ml if pick_side == "HOME" else (1 - market_ml if market_ml is not None else None),
                edge_pp=edge if pick_side == "HOME" else (-edge if edge is not None else None),
                confidence=abs(p_ml - 0.5) * 2.0,
                tier=_confidence_tier(abs(p_ml - 0.5) * 2.0),
                pred_home_runs=float(r["pred_home_runs"]),
                pred_away_runs=float(r["pred_away_runs"]),
                p_home_win=p_ml, p_home_runline_cover=p_rl,
                p_total_over=p_ou, total_line=total_line,
                total_line_source=total_line_source,
            )
        )

        # Run line (always home -1.5 / away +1.5). When the de-vigged
        # market RL probability exists, surface it + the edge.
        rl_side = "HOME" if p_rl >= 0.5 else "AWAY"
        rl_prob = max(p_rl, 1 - p_rl)
        rl_long = (
            f"{home} -1.5" if rl_side == "HOME" else f"{away} +1.5"
        )
        market_rl_home = (
            float(r["market_rl_home_close_prob"])
            if "market_rl_home_close_prob" in r.index and pd.notna(r.get("market_rl_home_close_prob"))
            else None
        )
        rl_market_prob = None
        rl_edge_pp = None
        if market_rl_home is not None:
            rl_market_prob = market_rl_home if rl_side == "HOME" else 1 - market_rl_home
            rl_edge_pp = (rl_prob - rl_market_prob) * 100
        rows.append(
            PickRow(
                game_pk=int(r["game_pk"]),
                game_date=game_date_val,
                away_team_abbr=away, home_team_abbr=home,
                market="runline", pick=rl_side, pick_long=rl_long,
                model_prob=rl_prob, market_prob=rl_market_prob, edge_pp=rl_edge_pp,
                confidence=abs(p_rl - 0.5) * 2.0,
                tier=_confidence_tier(abs(p_rl - 0.5) * 2.0),
                pred_home_runs=float(r["pred_home_runs"]),
                pred_away_runs=float(r["pred_away_runs"]),
                p_home_win=p_ml, p_home_runline_cover=p_rl,
                p_total_over=p_ou, total_line=total_line,
                total_line_source=total_line_source,
            )
        )

        # Over/under -- always emit when we have a probability + line
        # (which is now true for every game thanks to the baseline fallback).
        if p_ou is not None and total_line is not None and total_line == total_line:
            ou_side = "OVER" if p_ou >= 0.5 else "UNDER"
            ou_prob = max(p_ou, 1 - p_ou)
            line_label = (
                f"{total_line:g}" if total_line_source == "market"
                else f"{total_line:g}*"  # asterisk = league-avg fallback
            )
            ou_long = f"{ou_side} {line_label}"
            market_ou_over = (
                float(r["market_total_over_close_prob"])
                if "market_total_over_close_prob" in r.index and pd.notna(r.get("market_total_over_close_prob"))
                else None
            )
            ou_market_prob = None
            ou_edge_pp = None
            # Only surface a totals edge when we are actually playing
            # against a market line (not when total_line_source == "baseline").
            if market_ou_over is not None and total_line_source == "market":
                ou_market_prob = market_ou_over if ou_side == "OVER" else 1 - market_ou_over
                ou_edge_pp = (ou_prob - ou_market_prob) * 100
            rows.append(
                PickRow(
                    game_pk=int(r["game_pk"]),
                    game_date=game_date_val,
                    away_team_abbr=away, home_team_abbr=home,
                    market="total", pick=ou_side, pick_long=ou_long,
                    model_prob=ou_prob, market_prob=ou_market_prob, edge_pp=ou_edge_pp,
                    confidence=abs(p_ou - 0.5) * 2.0,
                    tier=_confidence_tier(abs(p_ou - 0.5) * 2.0),
                    pred_home_runs=float(r["pred_home_runs"]),
                    pred_away_runs=float(r["pred_away_runs"]),
                    p_home_win=p_ml, p_home_runline_cover=p_rl,
                    p_total_over=p_ou, total_line=total_line,
                    total_line_source=total_line_source,
                )
            )
    rows.sort(key=lambda r: r.confidence, reverse=True)
    return rows


def _ml_pick(p_home: float, away: str, home: str) -> tuple[str, str, float]:
    if p_home >= 0.5:
        return "HOME", home, p_home
    return "AWAY", away, 1.0 - p_home


# ---------------------------------------------------------------------------
# Game detail
# ---------------------------------------------------------------------------


@dataclass
class GameDetail:
    game_pk: int
    game_date: date_cls
    away_team_abbr: str
    home_team_abbr: str
    venue: str
    pred_home_runs: float
    pred_away_runs: float
    pred_home_std: float
    pred_away_std: float
    p_home_win: float
    p_home_runline_cover: float
    p_total_over: float | None
    total_line: float | None
    total_line_source: str
    market_ml_home_prob: float | None
    home_sp_id: int | None
    away_sp_id: int | None
    home_sp_name: str | None
    away_sp_name: str | None
    weather: dict[str, Any]
    score_distribution: dict[str, Any]
    feature_panel: list[dict[str, Any]]


def p_total_over_at_line(
    home_mean: float, home_std: float,
    away_mean: float, away_std: float,
    line: float, *, n_sims: int = 50_000, seed: int = 7,
) -> float:
    """Monte-Carlo P(home + away > line) for an arbitrary line.

    Lightweight wrapper around the same negative-binomial draw used by the
    main simulator. Used by the game-detail "try a line" feature.
    """
    import numpy as np

    rng = np.random.default_rng(seed)

    def draw(mean: float, std: float) -> "np.ndarray":
        mean = max(mean, 0.1)
        var = max(std**2, mean + 0.1)
        r = mean**2 / (var - mean)
        p = r / (r + mean)
        return rng.negative_binomial(n=r, p=p, size=n_sims)

    totals = draw(home_mean, home_std) + draw(away_mean, away_std)
    return float((totals > float(line)).mean())


def get_game_detail(target: date_cls, game_pk: int) -> GameDetail | None:
    """Fetch one game's prediction + the feature panel that drove it."""
    from mlb_model.data.warehouse import query

    df = get_predictions(target)
    if df.empty:
        return None
    game_row = df[df["game_pk"] == game_pk]
    if game_row.empty:
        return None
    r = game_row.iloc[0]

    # Prefer the venue name MLB stamped directly on the game row -- it
    # never goes stale. If for some reason that column is empty, look up
    # the home team's seed venue by abbr (the same join the rest of the
    # codebase now uses, because MLB venue_ids are not stable across
    # eras and don't line up with our synthetic seed IDs).
    if pd.notna(r.get("venue_name")) and str(r["venue_name"]).strip():
        venue = str(r["venue_name"]).strip()
    else:
        venue_row = query(
            "SELECT v.name FROM venues v WHERE v.team_abbr = ?",
            (str(r["home_team_abbr"]),),
        )
        venue = str(venue_row.iloc[0, 0]) if not venue_row.empty else "Unknown"

    # Probable pitcher names
    sp_names = query(
        """
        SELECT pp.pitcher_id, pp.pitcher_name, pp.is_home
        FROM probable_pitchers pp WHERE pp.game_pk = ?
        """,
        (int(game_pk),),
    )
    home_sp_name = away_sp_name = None
    if not sp_names.empty:
        for _, p in sp_names.iterrows():
            if bool(p["is_home"]):
                home_sp_name = p["pitcher_name"]
            else:
                away_sp_name = p["pitcher_name"]

    weather_row = query(
        "SELECT temp_f, humidity_pct, wind_speed_mph, wind_out_to_cf, is_dome FROM weather WHERE game_pk = ?",
        (int(game_pk),),
    )
    weather = (
        {
            "temp_f": float(weather_row.iloc[0]["temp_f"]) if pd.notna(weather_row.iloc[0]["temp_f"]) else None,
            "humidity_pct": float(weather_row.iloc[0]["humidity_pct"]) if pd.notna(weather_row.iloc[0]["humidity_pct"]) else None,
            "wind_speed_mph": float(weather_row.iloc[0]["wind_speed_mph"]) if pd.notna(weather_row.iloc[0]["wind_speed_mph"]) else None,
            "wind_out_to_cf": float(weather_row.iloc[0]["wind_out_to_cf"]) if pd.notna(weather_row.iloc[0]["wind_out_to_cf"]) else None,
            "is_dome": bool(weather_row.iloc[0]["is_dome"]),
        }
        if not weather_row.empty
        else {}
    )

    # Score distribution: approximate negative-binomial PMF from runs predictions
    home_mean = float(r["pred_home_runs"])
    away_mean = float(r["pred_away_runs"])
    home_std = float(r["runs_std_home"]) if "runs_std_home" in r.index else float(np.sqrt(home_mean) * 1.3)
    away_std = float(r["runs_std_away"]) if "runs_std_away" in r.index else float(np.sqrt(away_mean) * 1.3)
    distro = _score_distribution(home_mean, home_std, away_mean, away_std)

    panel = _feature_panel(target, game_pk)

    # Effective total line (market when available, baseline otherwise).
    if "effective_total_line" in r.index and pd.notna(r["effective_total_line"]):
        eff_total_line = float(r["effective_total_line"])
    elif pd.notna(r.get("market_total_close")):
        eff_total_line = float(r["market_total_close"])
    elif pd.notna(r.get("total_line")):
        eff_total_line = float(r["total_line"])
    else:
        eff_total_line = None
    line_source = (
        str(r["total_line_source"]) if pd.notna(r.get("total_line_source"))
        else ("market" if pd.notna(r.get("market_total_close")) else "baseline")
    )

    return GameDetail(
        game_pk=int(r["game_pk"]),
        game_date=target,
        away_team_abbr=str(r["away_team_abbr"]),
        home_team_abbr=str(r["home_team_abbr"]),
        venue=venue,
        pred_home_runs=home_mean,
        pred_away_runs=away_mean,
        pred_home_std=home_std,
        pred_away_std=away_std,
        p_home_win=float(r["p_home_win"]),
        p_home_runline_cover=float(r["p_home_runline_cover"]),
        p_total_over=float(r["p_total_over"]) if pd.notna(r["p_total_over"]) else None,
        total_line=eff_total_line,
        total_line_source=line_source,
        market_ml_home_prob=float(r["market_ml_home_close_prob"]) if pd.notna(r["market_ml_home_close_prob"]) else None,
        home_sp_id=int(r["home_sp_id"]) if pd.notna(r["home_sp_id"]) else None,
        away_sp_id=int(r["away_sp_id"]) if pd.notna(r["away_sp_id"]) else None,
        home_sp_name=home_sp_name,
        away_sp_name=away_sp_name,
        weather=weather,
        score_distribution=distro,
        feature_panel=panel,
    )


def _score_distribution(
    home_mean: float, home_std: float, away_mean: float, away_std: float
) -> dict[str, Any]:
    """Compute P(runs = k) up to k=15 for each team from neg-binomial."""
    from scipy.stats import nbinom

    def pmf(mean: float, std: float) -> list[float]:
        mean = max(mean, 0.1)
        var = max(std**2, mean + 0.1)
        r = mean**2 / (var - mean)
        p = r / (r + mean)
        return [float(nbinom.pmf(k, r, p)) for k in range(16)]

    return {
        "k": list(range(16)),
        "home": pmf(home_mean, home_std),
        "away": pmf(away_mean, away_std),
    }


def _feature_panel(target: date_cls, game_pk: int) -> list[dict[str, Any]]:
    """Return a hand-curated panel of features for the game.

    Pulls directly from the feature builders so the UI can show
    something like "home SP ERA last 5 starts: 3.21, league avg 4.10".
    """
    from mlb_model.features.assemble import build_features_table

    season = target.year
    feats = build_features_table(season, season)
    if feats.empty:
        return []
    row = feats[feats["game_pk"] == game_pk]
    if row.empty:
        return []
    r = row.iloc[0]

    def get(col: str) -> float | None:
        if col in r.index and pd.notna(r[col]):
            return float(r[col])
        return None

    panel: list[dict[str, Any]] = []

    def add(label: str, group: str, home_col: str, away_col: str, unit: str = "", fmt: str = "{:.2f}") -> None:
        h, a = get(home_col), get(away_col)
        if h is None and a is None:
            return
        panel.append(
            {
                "label": label,
                "group": group,
                "home": fmt.format(h) + unit if h is not None else "—",
                "away": fmt.format(a) + unit if a is not None else "—",
                "home_raw": h,
                "away_raw": a,
            }
        )

    # Starting pitcher (boxscore-based rolling). Names match feature builders:
    # `_r5g` = last 5 starts, `_r30d` = last 30 days.
    add("ERA, last 5 starts",      "Starting pitcher", "home_sp_box_era_per_start_r5g",  "away_sp_box_era_per_start_r5g")
    add("WHIP, last 5 starts",     "Starting pitcher", "home_sp_box_whip_per_start_r5g", "away_sp_box_whip_per_start_r5g")
    add("K/9, last 5 starts",      "Starting pitcher", "home_sp_box_k_per_9_r5g",        "away_sp_box_k_per_9_r5g")
    add("BB/9, last 5 starts",     "Starting pitcher", "home_sp_box_bb_per_9_r5g",       "away_sp_box_bb_per_9_r5g")
    add("IP, last 5 starts (avg)", "Starting pitcher", "home_sp_box_innings_pitched_r5g","away_sp_box_innings_pitched_r5g")

    # Bullpen (note: builder prefixes value cols with "bullpen_" already, then
    # the assemble step re-prefixes with "home_bullpen_"/"away_bullpen_").
    add("Bullpen ERA, last 30d",  "Bullpen", "home_bullpen_bullpen_era_r30d",   "away_bullpen_bullpen_era_r30d")
    add("Bullpen WHIP, last 30d", "Bullpen", "home_bullpen_bullpen_whip_r30d",  "away_bullpen_bullpen_whip_r30d")
    add("Bullpen K%, last 30d",   "Bullpen", "home_bullpen_bullpen_k_pct_r30d", "away_bullpen_bullpen_k_pct_r30d", unit="%", fmt="{:.1%}")

    # Team form (rolling 10 games)
    add("Runs scored, last 10g",   "Offense", "home_team_runs_scored_r10g",  "away_team_runs_scored_r10g")
    add("Runs allowed, last 10g",  "Defense", "home_team_runs_allowed_r10g", "away_team_runs_allowed_r10g")
    add("Run differential, last 10g", "Form", "home_team_run_diff_r10g",     "away_team_run_diff_r10g", fmt="{:+.2f}")
    add("HR per game, last 30d",   "Offense", "home_team_home_runs_r30d",    "away_team_home_runs_r30d")

    # Lineup quality
    add("Lineup ISO, last 30d",   "Lineup", "home_lineup_avg_iso_r30d",     "away_lineup_avg_iso_r30d", fmt="{:.3f}")
    add("Lineup OPS proxy, 30d",  "Lineup", "home_lineup_avg_ops_proxy_r30d","away_lineup_avg_ops_proxy_r30d", fmt="{:.3f}")
    add("Top power bat ISO",      "Lineup", "home_lineup_max_iso_r30d",     "away_lineup_max_iso_r30d", fmt="{:.3f}")
    add("Active batters w/ data", "Lineup", "home_lineup_n_active_batters", "away_lineup_n_active_batters", fmt="{:.0f}")

    # Schedule context
    add("Days of rest",            "Context", "home_days_rest",     "away_days_rest", fmt="{:.0f}")
    add("Travel miles, last game", "Context", "home_travel_miles",  "away_travel_miles", fmt="{:.0f}")

    return panel


# ---------------------------------------------------------------------------
# Performance / model card
# ---------------------------------------------------------------------------


def load_backtest() -> pd.DataFrame:
    """Read the latest backtest CSV used by the model card."""
    candidates = [
        settings.logs_dir / "backtest_v4.csv",
        settings.logs_dir / "backtest_v3.csv",
        settings.logs_dir / "backtest_v2.csv",
        settings.logs_dir / "backtest_v1.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


def load_2026_summary() -> dict[str, Any]:
    """Return the live 2026-season backtest if present."""
    path = settings.logs_dir / "backtest_2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Picks log (persisted)
# ---------------------------------------------------------------------------


PICKS_LOG_PATH = settings.cache_dir / "picks_log.parquet"


def _read_log_raw() -> pd.DataFrame:
    if not PICKS_LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_parquet(PICKS_LOG_PATH)
    # Backfill ``pick_id`` for rows logged before that column existed.
    if "pick_id" not in df.columns or df["pick_id"].isna().any():
        import uuid

        if "pick_id" not in df.columns:
            df["pick_id"] = ""
        df["pick_id"] = df["pick_id"].fillna("").astype(str)
        for idx in df.index[df["pick_id"] == ""]:
            df.at[idx, "pick_id"] = uuid.uuid4().hex
    if "stake_units" not in df.columns:
        df["stake_units"] = 1.0
    df["stake_units"] = pd.to_numeric(df["stake_units"], errors="coerce").fillna(1.0)
    return df


def append_logged_pick(record: dict[str, Any]) -> str:
    """Persist a "I'm taking this pick" record. Returns the new pick_id."""
    import uuid

    PICKS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = {
        **record,
        "pick_id": record.get("pick_id") or uuid.uuid4().hex,
        "logged_at": datetime.now(UTC).isoformat(),
        "stake_units": float(record.get("stake_units", 1.0)),
    }
    existing = _read_log_raw()
    if existing.empty:
        df = pd.DataFrame([record])
    else:
        # If the user re-clicks "Log" on the same (game, market, pick), refresh
        # the existing row in place rather than creating a duplicate.
        mask = (
            (existing["game_pk"].astype("int64") == int(record["game_pk"]))
            & (existing["market"] == record["market"])
            & (existing["pick"] == record["pick"])
        )
        if mask.any():
            for col, val in record.items():
                # Preserve the original logged_at + pick_id so the row keeps its identity.
                if col in ("logged_at", "pick_id"):
                    continue
                existing.loc[mask, col] = val
            df = existing
        else:
            df = pd.concat([existing, pd.DataFrame([record])], ignore_index=True)
    df.to_parquet(PICKS_LOG_PATH, index=False)
    return str(record["pick_id"])


def delete_logged_pick(pick_id: str) -> bool:
    """Remove a single logged pick by its pick_id. Returns True if deleted."""
    df = _read_log_raw()
    if df.empty:
        return False
    mask = df["pick_id"].astype(str) == str(pick_id)
    if not mask.any():
        return False
    df = df.loc[~mask].reset_index(drop=True)
    if df.empty:
        # Pandas can't write an empty parquet with no schema; just delete the file.
        PICKS_LOG_PATH.unlink(missing_ok=True)
    else:
        df.to_parquet(PICKS_LOG_PATH, index=False)
    return True


def update_logged_pick(pick_id: str, *, stake_units: float | None = None) -> bool:
    """Edit a logged pick. Returns True if a row was updated."""
    df = _read_log_raw()
    if df.empty:
        return False
    mask = df["pick_id"].astype(str) == str(pick_id)
    if not mask.any():
        return False
    if stake_units is not None:
        df.loc[mask, "stake_units"] = max(0.0, float(stake_units))
    df.to_parquet(PICKS_LOG_PATH, index=False)
    return True


def season_performance(season: int | None = None) -> dict[str, Any]:
    """Aggregate the prediction journal into everything the /season page needs.

    Returns a dict with keys:

    * ``season`` -- the season we filtered to (or ``None`` for all-time)
    * ``available_seasons`` -- sorted list of seasons present in the journal
    * ``summary`` -- list of ``SeasonSummary``-as-dict rows, one per market
    * ``calibration`` -- per-market list of calibration bins
    * ``rolling`` -- per-market daily rolling-30 accuracy series
    * ``tier_breakdown`` -- per-market accuracy by confidence tier
    * ``recent_results`` -- the most recent 50 graded picks (for a tail table)
    * ``journal_size`` -- total rows in the journal (diagnostic)
    """
    from mlb_model.journal import (
        calibration_bins,
        grade_journal,
        rolling_accuracy,
        season_summary,
        slice_breakdown,
    )
    from mlb_model.journal.record import JOURNAL_PATH

    if not JOURNAL_PATH.exists():
        return {
            "season": season,
            "available_seasons": [],
            "summary": [],
            "calibration": {},
            "rolling": {},
            "tier_breakdown": {},
            "recent_results": [],
            "journal_size": 0,
        }

    journal = pd.read_parquet(JOURNAL_PATH)
    available_seasons = sorted(
        int(s) for s in pd.unique(pd.to_numeric(journal["season"], errors="coerce").dropna())
    )
    if season is None and available_seasons:
        season = max(available_seasons)

    graded = grade_journal(journal=journal, season=season)
    summary = [s.as_dict() for s in season_summary(graded)]

    markets = ["moneyline", "runline", "total"]
    calibration: dict[str, list[dict[str, Any]]] = {}
    rolling: dict[str, list[dict[str, Any]]] = {}
    tier_breakdown: dict[str, list[dict[str, Any]]] = {}

    for market in markets:
        cb = calibration_bins(graded, market=market)
        if not cb.empty:
            calibration[market] = cb.to_dict("records")
        ra = rolling_accuracy(graded, market=market, window_days=30)
        if not ra.empty:
            ra_records = ra.to_dict("records")
            for rec in ra_records:
                # JSON-safe date serialization for Chart.js.
                rec["game_date"] = rec["game_date"].isoformat() if rec["game_date"] else None
            rolling[market] = ra_records
        sub = graded[graded["market"] == market]
        sb = slice_breakdown(sub, by="tier")
        if not sb.empty:
            tier_breakdown[market] = sb.to_dict("records")

    # Tail of most recent graded picks for transparency.
    if not graded.empty:
        tail = (
            graded[graded["result"].isin({"win", "loss", "push"})]
            .sort_values("game_date", ascending=False)
            .head(50)
        )
        recent_results = tail[
            [
                "game_date", "away_team_abbr", "home_team_abbr",
                "market", "pick_long", "model_prob", "tier",
                "result", "roi_units",
            ]
        ].copy()
        recent_results["game_date"] = recent_results["game_date"].astype(str)
        recent_results = recent_results.to_dict("records")
    else:
        recent_results = []

    # If we wrote an end-of-season sweep report for this season, link it
    # so the user can open it from the dashboard without hunting through
    # ``reports/``. We surface BOTH the markdown report (most useful) and
    # the JSON payload (for power users / tooling).
    eos_links: dict[str, str] | None = None
    if season is not None:
        try:
            from mlb_model.season.end_of_season import REPORTS_ROOT

            report_dir = REPORTS_ROOT / f"end_of_season_{season}"
            md_path = report_dir / "report.md"
            json_path = report_dir / "report.json"
            if md_path.exists() or json_path.exists():
                eos_links = {}
                if md_path.exists():
                    eos_links["markdown"] = str(md_path)
                if json_path.exists():
                    eos_links["json"] = str(json_path)
                eos_links["dir"] = str(report_dir)
        except Exception:  # noqa: BLE001
            eos_links = None

    # Hitter-prop tier breakdown -- grade any newly-settled rows first
    # so the page always reflects the freshest results.
    prop_breakdown: dict[str, list[dict]] = {}
    try:
        from mlb_model.journal import props as _props
        _props.grade_props()
        prop_breakdown = _props.tier_breakdown(season=season)
    except Exception:  # noqa: BLE001
        log.exception("season.prop_breakdown.failed")

    return {
        "season": season,
        "available_seasons": available_seasons,
        "summary": summary,
        "calibration": calibration,
        "rolling": rolling,
        "tier_breakdown": tier_breakdown,
        "prop_breakdown": prop_breakdown,
        "recent_results": recent_results,
        "journal_size": int(len(journal)),
        "eos_report": eos_links,
    }


def clear_logged_picks() -> int:
    """Drop every logged pick. Returns the number removed."""
    df = _read_log_raw()
    n = int(len(df))
    PICKS_LOG_PATH.unlink(missing_ok=True)
    return n


def load_picks_log() -> pd.DataFrame:
    df = _read_log_raw()
    if df.empty:
        return df
    # Score against finalized outcomes
    return _grade_picks(df)


def _grade_picks(picks: pd.DataFrame) -> pd.DataFrame:
    """Join logged picks against finalized game results."""
    from mlb_model.data.warehouse import query

    if picks.empty:
        return picks
    pks = picks["game_pk"].astype(int).unique().tolist()
    if not pks:
        return picks
    finals = query(
        f"""
        SELECT game_pk, home_score, away_score, home_win, status
        FROM games WHERE game_pk IN ({",".join("?" for _ in pks)})
        """,
        tuple(pks),
    )
    if finals.empty:
        picks["result"] = "pending"
        picks["roi_units"] = float("nan")
        return picks
    merged = picks.merge(finals, on="game_pk", how="left")

    def _outcome(row: pd.Series) -> str:
        if row.get("status") != "Final":
            return "pending"
        market = row["market"]
        if market == "moneyline":
            # A regular-season MLB tie is now nearly impossible (extra-innings
            # rule) but it can happen in suspended games and historical data.
            # Treat it the same as the journal grader does: a push, not a
            # loss for HOME / win for AWAY. Prior versions of this branch
            # silently mis-graded any tie.
            hs = float(row["home_score"])
            as_ = float(row["away_score"])
            if hs == as_:
                return "push"
            won = (hs > as_) == (row["pick"] == "HOME")
        elif market == "runline":
            margin = float(row["home_score"]) - float(row["away_score"])
            home_covers = margin > 1.5
            won = home_covers == (row["pick"] == "HOME")
        elif market == "total":
            total = float(row["home_score"]) + float(row["away_score"])
            line = float(row.get("total_line", float("nan")))
            if line != line:  # NaN
                return "pending"
            if total == line:
                return "push"
            went_over = total > line
            won = went_over == (row["pick"] == "OVER")
        else:
            return "pending"
        return "win" if won else "loss"

    merged["result"] = merged.apply(_outcome, axis=1)

    # Simple ROI per unit assuming -110 unless we have the actual American odds.
    def _roi_per_unit(result: str) -> float:
        if result == "win":
            return 0.909  # win 100, return 90.9 profit at -110
        if result == "loss":
            return -1.0
        if result == "push":
            return 0.0
        return float("nan")

    per_unit = merged["result"].map(_roi_per_unit)
    stakes = pd.to_numeric(merged.get("stake_units", 1.0), errors="coerce").fillna(1.0)
    merged["roi_units"] = per_unit * stakes
    return merged
