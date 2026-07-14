"""Daily prediction pipeline.

Workflow for producing today's picks:
  1) Pull schedule + probable pitchers for the target date (or window).
  2) Pull (or refresh) any missing per-team/SP context from recent games.
  3) Pull weather forecast for outdoor games.
  4) Assemble feature rows for the upcoming games.
  5) Apply the latest trained models to produce ML / RL / OU probabilities.
  6) Apply the calibrators.
  7) Output a ranked picks table by confidence tier.
"""

from __future__ import annotations

from datetime import date as date_cls, timedelta
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from mlb_model.config import settings
from mlb_model.data.pipeline import pull_date_range
from mlb_model.features.assemble import build_features_table
from mlb_model.logging import get_logger
from mlb_model.model.calibrate import load_calibrator
from mlb_model.model.feature_matrix import (
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)
from mlb_model.model.runs import load_model
from mlb_model.model.simulate import simulate_games

log = get_logger("predict.daily")


def predict_for_date(target_date: date_cls, *, refresh_data: bool = True) -> pd.DataFrame:
    """Produce calibrated picks for all games on ``target_date``.

    Returns one row per game with model and market columns plus tier flags.
    """
    season = target_date.year
    log.info("predict.start", target_date=target_date)

    if refresh_data:
        # Pull schedule for ~next 3 days so probable pitchers are populated.
        pull_date_range(target_date, target_date + timedelta(days=2))

        # Pull a weather forecast for outdoor games on target_date. We
        # silently skip any game already weather-cached. Errors here are
        # non-fatal -- worst case the features impute to league-average
        # weather, which is exactly what happened before we wired this
        # in. Best case it adds a noticeable signal to totals and runline.
        try:
            from mlb_model.data.sources import weather as _weather
            from mlb_model.data.venues_seed import seed_venues_df
            from mlb_model.data.warehouse import query

            slate_for_weather = query(
                """
                SELECT g.game_pk, g.venue_id, g.scheduled_start, g.home_team_abbr
                FROM games g
                WHERE g.game_date = ?
                  AND g.game_pk NOT IN (SELECT game_pk FROM weather)
                """,
                (target_date,),
            )
            if not slate_for_weather.empty:
                wrote = _weather.ingest_weather_for_games(
                    slate_for_weather, seed_venues_df()
                )
                log.info("predict.weather.ingested", rows=wrote)
        except Exception:  # noqa: BLE001
            log.exception("predict.weather.failed")

        # Pull today's live odds from The Odds API so the per-game
        # market line (ML / RL / total) is available for the market
        # features join. No-ops cleanly when ``MLB_ODDS_API_KEY`` is
        # unset -- predictions then fall back to the league-average
        # baseline as before. One ``/odds`` call covers the whole slate
        # (3 credits on the free tier).
        try:
            from mlb_model.data.sources import odds_api as _odds_api

            wrote = _odds_api.ingest_live_slate()
            log.info("predict.odds_api.ingested", rows=wrote)
        except Exception:  # noqa: BLE001
            log.exception("predict.odds_api.failed")

    # Build features. We pass the whole season as training so rolling
    # windows for SP / team form / etc. include all season-to-date games.
    features = build_features_table(season, season)
    if features.empty:
        log.warning("predict.no_games", target_date=target_date)
        return features

    target_mask = features["game_date"].dt.date == target_date
    target = features[target_mask].copy()
    if target.empty:
        log.warning("predict.no_target_games", target_date=target_date)
        return target

    # Drop games MLB has already postponed / cancelled so we don't show
    # bogus predictions for slate slots that aren't going to happen. We
    # have to consult the schedule directly because ``status`` isn't a
    # feature column (it would leak the outcome for finalized games).
    try:
        from mlb_model.data.warehouse import query

        pks = target["game_pk"].astype(int).tolist()
        if pks:
            placeholders = ", ".join(["?"] * len(pks))
            status_df = query(
                f"SELECT game_pk, status FROM games WHERE game_pk IN ({placeholders})",
                tuple(pks),
            )
            dropped = status_df["status"].str.lower().isin(
                {"postponed", "cancelled", "canceled", "suspended", "delayed start"}
            )
            bad = set(status_df.loc[dropped, "game_pk"].astype(int))
            if bad:
                before = len(target)
                target = target[~target["game_pk"].isin(bad)].copy()
                log.info(
                    "predict.dropped_unplayable",
                    target_date=target_date,
                    dropped=before - len(target),
                    game_pks=sorted(bad),
                )
            if target.empty:
                log.warning("predict.all_postponed", target_date=target_date)
                return target
    except Exception:  # noqa: BLE001 -- never let status filtering crash predict
        log.exception("predict.status_filter_failed")

    # Prefer the persisted FeatureSpec from training so we use the exact
    # same imputation values; fall back to fitting on the historical
    # portion if no persisted spec exists.
    spec_path = settings.model_dir / "feature_spec.joblib"
    if spec_path.exists():
        spec = joblib.load(spec_path)
    else:
        historical = features[~target_mask]
        if historical.empty:
            log.warning("predict.no_historical_features")
            return pd.DataFrame()
        spec = fit_spec(historical)

    home_model = load_model("home_runs")
    away_model = load_model("away_runs")

    # Backwards compat: specs persisted before ``final_feature_cols`` was
    # introduced won't have the locked column list. Recover it from the
    # trained model so the inference matrix matches LightGBM's expectations.
    if not getattr(spec, "final_feature_cols", None):
        spec.final_feature_cols = list(home_model.feature_cols) or list(away_model.feature_cols)

    X_home, mh, _ = build_runs_matrix(target, spec)
    X_away, ma, _ = build_runs_matrix_away(target, spec)
    valid = mh & ma
    rows = target.loc[valid].reset_index(drop=True)

    home_mean, home_std = home_model.predict_distribution(X_home[valid])
    away_mean, away_std = away_model.predict_distribution(X_away[valid])

    # Market closing total may be missing for today's slate (live-odds
    # ingestion is offline / not implemented). Substitute a season-aware
    # league-average baseline so the over/under market remains usable.
    market_total = rows["market_total_close"].astype("float64").to_numpy()
    from mlb_model.predict.totals_baseline import league_avg_total

    baseline = float(league_avg_total(int(target_date.year)))
    total_line_source = np.where(np.isfinite(market_total), "market", "baseline")
    total_lines = np.where(np.isfinite(market_total), market_total, baseline)

    # Runline framing: use the market's home line (-1.5 when home is the
    # favorite, +1.5 when the away team is) so we never offer a side the
    # book doesn't post. Fall back to the historical home -1.5 assumption
    # only when no market line exists.
    if "market_rl_home_close" in rows.columns:
        market_rl = rows["market_rl_home_close"].astype("float64").to_numpy()
    else:
        market_rl = np.full(len(rows), np.nan)
    rl_line_home = np.where(np.isfinite(market_rl), market_rl, -1.5)
    rl_line_source = np.where(np.isfinite(market_rl), "market", "assumed")

    # Daily inference uses the faster (still accurate) draw count --
    # see ``monte_carlo_iterations_inference`` for rationale.
    preds = simulate_games(
        game_pks=rows["game_pk"].to_numpy(),
        pred_home=(home_mean, home_std),
        pred_away=(away_mean, away_std),
        total_lines=total_lines,
        home_rl_lines=rl_line_home,
        n_sims=settings.monte_carlo_iterations_inference,
        seed=settings.random_seed,
    )

    # Direct OU classifier if persisted. The classifier was trained
    # against real market lines, so we only trust it when we *have* a
    # market line; otherwise we fall back to the simulated p_over against
    # the league-average baseline.
    totals_path = settings.model_dir / "totals.joblib"
    if totals_path.exists():
        state = joblib.load(totals_path)
        booster = lgb.Booster(model_str=state["model_str"])
        p_over_direct = np.clip(booster.predict(X_home[valid]), 1e-6, 1 - 1e-6)
        used_market = total_line_source == "market"
        sim_p = np.array([p.p_total_over for p in preds], dtype=np.float64)
        p_over_final = np.where(used_market, p_over_direct, sim_p)
        for i, pred in enumerate(preds):
            pred.p_total_over = float(p_over_final[i])

    pred_df = pd.DataFrame([p.__dict__ for p in preds])
    pred_df["total_line_source"] = total_line_source
    # Runline framing context (aligned with ``rows``/``preds`` order):
    # the home team's line and whether it came from the market.
    pred_df["rl_line_home"] = rl_line_home
    pred_df["rl_line_source"] = rl_line_source

    # Apply calibrators if available. The totals calibrator is special:
    # it was trained on direct-classifier p_over values (centered ~0.50)
    # for games with real market lines. When we fall back to the sim's
    # p_over against the league-average baseline, those inputs are in a
    # different distribution (typically 0.2-0.3 here) and the isotonic
    # mapping collapses them to ~0. Skip the totals calibrator on
    # baseline rows -- the raw sim p_over is already sensible there.
    market_line_mask = pred_df["total_line_source"].to_numpy() == "market"
    for market_col, market in [
        ("p_home_win", "moneyline"),
        ("p_home_runline_cover", "runline"),
        ("p_total_over", "total"),
    ]:
        try:
            cal = load_calibrator(market)
        except FileNotFoundError:
            log.debug("calibrator.missing", market=market)
            continue
        raw = pred_df[market_col].to_numpy()
        pred_df[market_col + "_raw"] = raw
        if market == "total":
            calibrated = cal.transform(raw)
            pred_df[market_col] = np.where(market_line_mask, calibrated, raw)
        elif market == "runline":
            # The calibrator was trained on the fixed home -1.5 frame
            # (P(team covers -1.5)). For games where the market posts
            # home +1.5, our raw prob is P(home +1.5) = 1 - P(away -1.5);
            # calibrate the away -1.5 complement and mirror back so the
            # calibrator always sees the event family it was trained on.
            cal_home_frame = cal.transform(raw)
            cal_away_frame = 1.0 - cal.transform(1.0 - raw)
            pred_df[market_col] = np.where(rl_line_home < 0, cal_home_frame, cal_away_frame)
        else:
            pred_df[market_col] = cal.transform(raw)

    # Add context columns
    # Forward the new market-derived RL/OU probabilities into the
    # output so the UI can show edge for those markets too. When a
    # column doesn't exist (e.g. very early features), default it.
    extra_market_cols = [
        c for c in (
            "market_rl_home_close_prob",
            "market_total_over_close_prob",
        )
        if c in rows.columns
    ]
    output = rows[
        [
            "game_pk", "game_date", "home_team_abbr", "away_team_abbr",
            "home_sp_id", "away_sp_id",
            "market_ml_home_close_prob", "market_total_close",
            *extra_market_cols,
        ]
    ].merge(pred_df, on="game_pk", how="left")
    # Effective line presented to the user: market when available,
    # baseline otherwise. ``total_line`` is whatever was used in the sim;
    # ``market_total_close`` retains its NaN when no market line exists
    # so the UI can distinguish the two.
    output["effective_total_line"] = output["total_line"]

    # Confidence tiers
    output["confidence_ml"] = (output["p_home_win"] - 0.5).abs() * 2.0
    output["confidence_rl"] = (output["p_home_runline_cover"] - 0.5).abs() * 2.0
    output["confidence_ou"] = (output["p_total_over"] - 0.5).abs() * 2.0
    output = output.sort_values("confidence_ml", ascending=False).reset_index(drop=True)

    # Append every prediction to the long-running journal so the
    # /season page can track ongoing accuracy. Failure here MUST NOT
    # break the prediction pipeline -- it's pure bookkeeping.
    try:
        from mlb_model.journal.record import record_predictions_from_df

        record_predictions_from_df(output)
    except Exception:  # noqa: BLE001
        log.exception("predict.journal.record_failed")

    return output


def write_picks_csv(picks: pd.DataFrame, target_date: date_cls) -> Path:
    """Persist daily picks to a CSV for review."""
    out_dir = settings.project_root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"picks_{target_date.isoformat()}.csv"
    picks.to_csv(path, index=False)
    log.info("predict.csv.written", path=str(path), rows=len(picks))
    return path
