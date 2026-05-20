"""Break down the 2026 performance by month to check for early-season effects."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from mlb_model.backtest.metrics import accuracy, accuracy_by_confidence
from mlb_model.config import settings
from mlb_model.features.assemble import build_features_table
from mlb_model.logging import configure_logging
from mlb_model.model.feature_matrix import (
    build_runs_matrix,
    build_runs_matrix_away,
    fit_spec,
)
from mlb_model.model.runs import train_runs_model
from mlb_model.model.simulate import simulate_games

CUTOFF = date(2026, 5, 18)


def main() -> None:
    configure_logging()
    train_full = build_features_table(2018, 2025).dropna(
        subset=["target_home_score", "target_away_score"]
    )
    test_2026 = build_features_table(2026, 2026)
    test_2026 = test_2026[test_2026["game_date"].dt.date <= CUTOFF].dropna(
        subset=["target_home_score", "target_away_score"]
    ).copy()

    spec = fit_spec(train_full)
    train_sorted = train_full.sort_values("game_date").reset_index(drop=True)
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

    Xh, mh, _ = build_runs_matrix(test_2026, spec)
    Xa, ma, _ = build_runs_matrix_away(test_2026, spec)
    valid = mh & ma
    test_use = test_2026.loc[valid].reset_index(drop=True)
    hm, hs = home.predict_distribution(Xh[valid])
    am, ase = away.predict_distribution(Xa[valid])
    tl = pd.to_numeric(test_use["market_total_close"], errors="coerce").to_numpy(np.float64)
    preds = simulate_games(
        game_pks=test_use["game_pk"].to_numpy(),
        pred_home=(hm, hs), pred_away=(am, ase), total_lines=tl,
        n_sims=settings.monte_carlo_iterations, seed=settings.random_seed + 2026,
    )
    p_ml = np.array([p.p_home_win for p in preds])
    test_use["p_home_win"] = p_ml
    test_use["target_home_win_f"] = test_use["target_home_win"].astype("boolean").astype("float64")

    test_use["month_label"] = test_use["game_date"].dt.to_period("W").astype(str)

    print("\nWeekly performance (2026):")
    weeks = (
        test_use.sort_values("game_date")
        .assign(week=lambda d: d["game_date"].dt.to_period("W"))
        .groupby("week", sort=True)
    )
    rows = []
    for week, sub in weeks:
        p = sub["p_home_win"].to_numpy()
        y = sub["target_home_win_f"].to_numpy()
        rows.append({
            "week": str(week),
            "games": len(sub),
            "ml_acc": accuracy(p, y),
            "top30_acc": accuracy_by_confidence(p, y).win_rate_top_30pct,
        })
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    print("\nCumulative performance (2026):")
    test_sorted = test_use.sort_values("game_date").reset_index(drop=True)
    n = len(test_sorted)
    for cut in [100, 200, 300, 400, 500, 600, n]:
        sub = test_sorted.iloc[:cut]
        p = sub["p_home_win"].to_numpy()
        y = sub["target_home_win_f"].to_numpy()
        print(
            f"  first {cut:>4d} games: "
            f"ml_acc={accuracy(p, y):.3f}, "
            f"top10={accuracy_by_confidence(p, y).win_rate_top_10pct:.3f}, "
            f"top30={accuracy_by_confidence(p, y).win_rate_top_30pct:.3f}"
        )


if __name__ == "__main__":
    main()
