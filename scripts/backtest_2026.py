"""Backtest the 2026 season-to-date with two model variants.

Variant A: full feature set (market features imputed to median when absent).
Variant B: explicitly drop the five ``market_*`` features so we're
comparing apples-to-apples with how the model would actually run on
2026 production data where no closing line is available.

Writes:
    logs/backtest_2026.csv    -- one row per variant
    logs/backtest_2026.md     -- markdown summary
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_model.backtest.metrics import (
    accuracy,
    accuracy_by_confidence,
    brier_score,
    closing_line_value,
    log_loss,
)
from mlb_model.config import settings
from mlb_model.features.assemble import build_features_table
from mlb_model.logging import configure_logging, get_logger
from mlb_model.model.feature_matrix import (
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)
from mlb_model.model.runs import train_runs_model
from mlb_model.model.simulate import simulate_games
from mlb_model.model.totals import train_totals_model

log = get_logger("backtest_2026")

MARKET_COLS = [
    "market_ml_home_close_prob",
    "market_ml_away_close_prob",
    "market_ml_movement_home",
    "market_rl_home_close_price",
    "market_total_close",
]

CUTOFF = date(2026, 5, 18)


def _label_outcomes(features: pd.DataFrame, preds: list) -> pd.DataFrame:
    pred_df = pd.DataFrame([p.__dict__ for p in preds])
    actuals = features[
        [
            "game_pk", "target_home_win", "target_home_score",
            "target_away_score", "target_total_runs",
        ]
    ]
    df = pred_df.merge(actuals, on="game_pk", how="left")
    df["target_home_win"] = df["target_home_win"].astype("boolean").astype("float64")
    rl_margin = (df["target_home_score"] - df["target_away_score"]).astype("Float64")
    df["target_home_runline_cover"] = (rl_margin > 1.5).astype("Float64").astype("float64")
    total_line_f = pd.to_numeric(df["total_line"], errors="coerce")
    total_runs_f = pd.to_numeric(df["target_total_runs"], errors="coerce")
    df["target_over"] = (total_runs_f > total_line_f).astype("Float64").astype("float64")
    df.loc[total_line_f.isna() | total_runs_f.isna(), "target_over"] = float("nan")
    return df


def run_variant(
    variant: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    drop_cols: list[str] | None,
) -> dict:
    log.info("variant.start", variant=variant)
    train = train.dropna(subset=["target_home_score", "target_away_score"]).copy()
    test = test.dropna(subset=["target_home_score", "target_away_score"]).copy()
    if drop_cols:
        train = train.drop(columns=[c for c in drop_cols if c in train.columns])
        test = test.drop(columns=[c for c in drop_cols if c in test.columns])

    spec = fit_spec(train)
    train_sorted = train.sort_values("game_date").reset_index(drop=True)
    vc = int(len(train_sorted) * 0.85)
    tp, vp = train_sorted.iloc[:vc].copy(), train_sorted.iloc[vc:].copy()

    Xht, mht, feat_cols = build_runs_matrix(tp, spec)
    Xat, mat, _ = build_runs_matrix_away(tp, spec)
    spec.final_feature_cols = feat_cols
    Xhv, mhv, _ = build_runs_matrix(vp, spec)
    Xav, mav, _ = build_runs_matrix_away(vp, spec)

    home = train_runs_model(
        Xht[mht], tp["target_home_score"].to_numpy()[mht], feat_cols,
        eval_X=Xhv[mhv], eval_y=vp["target_home_score"].to_numpy()[mhv],
    )
    away = train_runs_model(
        Xat[mat], tp["target_away_score"].to_numpy()[mat], feat_cols,
        eval_X=Xav[mav], eval_y=vp["target_away_score"].to_numpy()[mav],
    )

    Xh, mh, _ = build_runs_matrix(test, spec)
    Xa, ma, _ = build_runs_matrix_away(test, spec)
    valid = mh & ma
    test_use = test.loc[valid].reset_index(drop=True)

    hm, hs = home.predict_distribution(Xh[valid])
    am, ase = away.predict_distribution(Xa[valid])
    tl = pd.to_numeric(
        test_use.get("market_total_close", pd.Series([np.nan] * len(test_use))),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(tl).any():
        tl = np.full(len(test_use), np.nan)

    preds = simulate_games(
        game_pks=test_use["game_pk"].to_numpy(),
        pred_home=(hm, hs), pred_away=(am, ase),
        total_lines=tl,
        n_sims=settings.monte_carlo_iterations,
        seed=settings.random_seed + 2026,
    )

    # Direct OU classifier (will be skipped when no 2026 odds)
    train_total_line = pd.to_numeric(
        tp.get("market_total_close", pd.Series([np.nan] * len(tp))),
        errors="coerce",
    ).to_numpy(dtype=np.float64)
    train_total_runs = (
        pd.to_numeric(tp["target_home_score"], errors="coerce")
        + pd.to_numeric(tp["target_away_score"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    y_ou_t = np.where(
        np.isfinite(train_total_line) & np.isfinite(train_total_runs),
        (train_total_runs > train_total_line).astype(np.float64),
        np.nan,
    )
    try:
        totals_model = train_totals_model(Xht[mht], y_ou_t[mht], feat_cols)
        p_over_direct = totals_model.predict(Xh[valid])
        no_line = ~np.isfinite(tl)
        if no_line.any():
            sim_p = np.array([p.p_total_over for p in preds], dtype=np.float64)
            p_over_direct = np.where(no_line, sim_p, p_over_direct)
        for i, pred in enumerate(preds):
            pred.p_total_over = float(p_over_direct[i])
    except ValueError as exc:  # pragma: no cover
        log.warning("totals.skip", reason=str(exc))

    scored = _label_outcomes(test_use, preds)

    def to_f(s):
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

    ml_p = to_f(scored["p_home_win"])
    ml_y = to_f(scored["target_home_win"])
    rl_p = to_f(scored["p_home_runline_cover"])
    rl_y = to_f(scored["target_home_runline_cover"])
    ou_p = to_f(scored["p_total_over"])
    ou_y = to_f(scored["target_over"])

    ml_decile = accuracy_by_confidence(ml_p, ml_y)
    rl_decile = accuracy_by_confidence(rl_p, rl_y)
    ou_decile = accuracy_by_confidence(ou_p, ou_y)

    return {
        "variant": variant,
        "n_games": int(len(scored)),
        "ml_brier": brier_score(ml_p, ml_y),
        "ml_logloss": log_loss(ml_p, ml_y),
        "ml_accuracy": accuracy(ml_p, ml_y),
        "ml_top3_acc": ml_decile.win_rate_top_3pct,
        "ml_top10_acc": ml_decile.win_rate_top_10pct,
        "ml_top30_acc": ml_decile.win_rate_top_30pct,
        "rl_accuracy": accuracy(rl_p, rl_y),
        "rl_top10_acc": rl_decile.win_rate_top_10pct,
        "rl_top30_acc": rl_decile.win_rate_top_30pct,
        "ou_accuracy": accuracy(ou_p, ou_y),
        "ou_top10_acc": ou_decile.win_rate_top_10pct,
    }


def main() -> None:
    configure_logging()
    log.info("starting 2026 evaluation", cutoff=CUTOFF.isoformat())

    # Build features for the full historical window plus 2026
    log.info("building features...")
    train_full = build_features_table(2018, 2025)
    test_2026 = build_features_table(2026, 2026)
    # Restrict 2026 to games through CUTOFF and only Finals
    test_2026 = test_2026[test_2026["game_date"].dt.date <= CUTOFF].copy()
    log.info(
        "data ready",
        train_rows=len(train_full),
        test_rows=len(test_2026),
        test_finals=int(test_2026["target_home_win"].notna().sum()),
    )

    variants = [
        ("full_features", None),
        ("no_market_features", MARKET_COLS),
    ]
    rows = []
    for name, drop in variants:
        rows.append(run_variant(name, train_full.copy(), test_2026.copy(), drop))

    df = pd.DataFrame(rows)
    out_csv = settings.logs_dir / "backtest_2026.csv"
    df.to_csv(out_csv, index=False)
    log.info("wrote", path=str(out_csv))

    # Markdown summary
    lines = [
        f"# 2026 season backtest -- through {CUTOFF.isoformat()}",
        "",
        f"Sample size: **{rows[0]['n_games']} final games** "
        f"(~7 weeks of regular-season play).",
        "",
        "| Metric | Full features | No-market features | 2019-2025 backtest avg |",
        "|---|---|---|---|",
    ]
    ref = {
        "ml_accuracy": 0.607,
        "ml_top30_acc": 0.692,
        "ml_top10_acc": 0.743,
        "ml_top3_acc":  0.759,
        "rl_accuracy": 0.652,
        "rl_top10_acc": 0.807,
        "ou_accuracy": 0.520,
    }
    labels = {
        "ml_accuracy":   "ML accuracy, all picks",
        "ml_top30_acc":  "ML top-30% confidence",
        "ml_top10_acc":  "ML top-10% confidence",
        "ml_top3_acc":   "ML top-3% conviction",
        "rl_accuracy":   "RL accuracy, all picks",
        "rl_top10_acc":  "RL top-10% confidence",
        "ou_accuracy":   "OU accuracy",
    }
    full = rows[0]
    no_mkt = rows[1]
    for k, label in labels.items():
        fmt = lambda x: f"{x:.3f}" if pd.notna(x) and x == x else "n/a"
        lines.append(
            f"| {label} | {fmt(full[k])} | {fmt(no_mkt[k])} | "
            f"{ref[k]:.3f} |"
        )
    summary_path = settings.logs_dir / "backtest_2026.md"
    summary_path.write_text("\n".join(lines))
    log.info("summary", path=str(summary_path))

    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
