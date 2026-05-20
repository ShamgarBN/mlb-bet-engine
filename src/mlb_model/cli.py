"""Typer-based command-line interface.

Exposes the project's pipelines:

    uv run mlb-model data init
    uv run mlb-model data pull --season 2024
    uv run mlb-model data pull-range 2014 2025
    uv run mlb-model train --through-season 2024
    uv run mlb-model backtest --start 2016 --end 2024
    uv run mlb-model predict --date today
"""

from __future__ import annotations

from datetime import date as date_cls, datetime, timedelta
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mlb_model.config import settings
from mlb_model.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, help="MLB prediction & betting model CLI")
data_app = typer.Typer(no_args_is_help=True, help="Data ingestion commands")
app.add_typer(data_app, name="data")

console = Console()
log = get_logger("cli")


def _parse_date(s: str) -> date_cls:
    if s.lower() == "today":
        return date_cls.today()
    if s.lower() == "yesterday":
        return date_cls.today() - timedelta(days=1)
    if s.lower() == "tomorrow":
        return date_cls.today() + timedelta(days=1)
    return datetime.strptime(s, "%Y-%m-%d").date()


@data_app.command("init")
def data_init() -> None:
    """Initialize the warehouse schema and seed venue metadata."""
    configure_logging()
    from mlb_model.data.pipeline import ensure_venues
    from mlb_model.data.warehouse import init_schema

    init_schema()
    n = ensure_venues()
    console.print(f"[green]Warehouse ready at[/green] {settings.warehouse_path}")
    console.print(f"[green]Venues seeded:[/green] {n}")


@data_app.command("pull")
def data_pull(
    season: Annotated[int, typer.Option(help="Season year to pull")] = date_cls.today().year,
    no_statcast: Annotated[bool, typer.Option(help="Skip Statcast pulls")] = False,
    no_weather: Annotated[bool, typer.Option(help="Skip weather pulls")] = False,
) -> None:
    """Pull all sources for one season into the warehouse."""
    configure_logging()
    from mlb_model.data.pipeline import pull_season

    counts = pull_season(season, with_statcast=not no_statcast, with_weather=not no_weather)
    table = Table(title=f"Season {season} ingest counts")
    table.add_column("source")
    table.add_column("rows", justify="right")
    for k, v in counts.items():
        table.add_row(str(k), str(v))
    console.print(table)


@data_app.command("pull-range")
def data_pull_range(
    start: Annotated[int, typer.Argument(help="First season")],
    end: Annotated[int, typer.Argument(help="Last season (inclusive)")],
    no_statcast: Annotated[bool, typer.Option(help="Skip Statcast pulls")] = False,
    no_weather: Annotated[bool, typer.Option(help="Skip weather pulls")] = False,
) -> None:
    """Pull all sources across [start, end] (inclusive)."""
    configure_logging()
    from mlb_model.data.pipeline import pull_season

    for season in range(start, end + 1):
        try:
            pull_season(season, with_statcast=not no_statcast, with_weather=not no_weather)
        except Exception as exc:  # noqa: BLE001 -- continue across season failures
            log.error("season.failed", season=season, error=str(exc))


@app.command()
def train(
    through_season: Annotated[int, typer.Option(help="Train on data up to and including this season")],
    train_start: Annotated[int, typer.Option(help="Earliest season to include")] = 2018,
) -> None:
    """Train final production models + calibrators on [train_start, through_season].

    Pipeline:
      1) Assemble features for the full window.
      2) Fit ``FeatureSpec`` (medians for imputation) on the entire window.
      3) Train home/away ``RunsModel`` with a 15% tail validation set.
      4) Train a direct over/under classifier on the same training matrix.
      5) Walk forward season-by-season to produce **out-of-fold** predictions
         for the calibrator -- this is the only way to get unbiased calibration
         on a model that has seen every other game.
      6) Fit isotonic regressors per market and persist all artifacts.
    """
    import numpy as np
    import pandas as pd

    configure_logging()
    from mlb_model.config import settings as _settings
    from mlb_model.features.assemble import build_features_table
    from mlb_model.model.calibrate import fit_calibrator, save_calibrator
    from mlb_model.model.feature_matrix import (
        build_runs_matrix,
        build_runs_matrix_away,
        fit_spec,
    )
    from mlb_model.model.runs import save_model, train_runs_model
    from mlb_model.model.simulate import simulate_games
    from mlb_model.model.totals import train_totals_model
    import joblib

    features = build_features_table(train_start, through_season).dropna(
        subset=["target_home_score", "target_away_score"]
    )
    if features.empty:
        console.print("[red]No training data available -- run `data pull` first.[/red]")
        raise typer.Exit(code=1)

    spec = fit_spec(features)
    feat_path = _settings.model_dir / "feature_spec.joblib"
    _settings.model_dir.mkdir(parents=True, exist_ok=True)

    # ---- final production models on ALL data ----
    features_sorted = features.sort_values("game_date").reset_index(drop=True)
    val_cut = int(len(features_sorted) * 0.85)
    train_part = features_sorted.iloc[:val_cut]
    val_part = features_sorted.iloc[val_cut:]

    X_home_t, mh_t, feat_cols = build_runs_matrix(train_part, spec)
    X_away_t, ma_t, _ = build_runs_matrix_away(train_part, spec)
    # Persist the locked column list with the spec so inference matches.
    spec.final_feature_cols = feat_cols
    joblib.dump(spec, feat_path, compress=("zlib", 3))
    X_home_v, mh_v, _ = build_runs_matrix(val_part, spec)
    X_away_v, ma_v, _ = build_runs_matrix_away(val_part, spec)

    y_home_t = train_part["target_home_score"].to_numpy()[mh_t]
    y_away_t = train_part["target_away_score"].to_numpy()[ma_t]
    y_home_v = val_part["target_home_score"].to_numpy()[mh_v]
    y_away_v = val_part["target_away_score"].to_numpy()[ma_v]

    home_model = train_runs_model(
        X_home_t[mh_t], y_home_t, feat_cols,
        eval_X=X_home_v[mh_v], eval_y=y_home_v,
    )
    away_model = train_runs_model(
        X_away_t[ma_t], y_away_t, feat_cols,
        eval_X=X_away_v[ma_v], eval_y=y_away_v,
    )
    save_model(home_model, "home_runs")
    save_model(away_model, "away_runs")

    # Direct OU classifier
    train_total_line = pd.to_numeric(
        train_part["market_total_close"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    train_total_runs = (
        pd.to_numeric(train_part["target_home_score"], errors="coerce")
        + pd.to_numeric(train_part["target_away_score"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    y_ou_t = np.where(
        np.isfinite(train_total_line) & np.isfinite(train_total_runs),
        (train_total_runs > train_total_line).astype(np.float64),
        np.nan,
    )
    val_total_line = pd.to_numeric(
        val_part["market_total_close"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    val_total_runs = (
        pd.to_numeric(val_part["target_home_score"], errors="coerce")
        + pd.to_numeric(val_part["target_away_score"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    y_ou_v = np.where(
        np.isfinite(val_total_line) & np.isfinite(val_total_runs),
        (val_total_runs > val_total_line).astype(np.float64),
        np.nan,
    )
    try:
        totals_model = train_totals_model(
            X_home_t[mh_t], y_ou_t[mh_t], feat_cols,
            X_val=X_home_v[mh_v], y_val=y_ou_v[mh_v],
        )
        joblib.dump(
            {
                "feature_cols": totals_model.feature_cols,
                "model_str": totals_model.booster.model_to_string(),
            },
            _settings.model_dir / "totals.joblib",
            compress=("zlib", 3),
        )
    except ValueError as exc:
        log.warning("totals.train.failed", reason=str(exc))

    # ---- out-of-fold predictions for calibration ----
    # We refit per season chronologically using the existing walk-forward
    # routine, then aggregate per-game probabilities to fit isotonic.
    console.print("[cyan]Computing OOF predictions for calibration...[/cyan]")
    oof_rows: list[dict] = []
    seasons = sorted(features["season"].unique().tolist())
    for s in seasons:
        if s == train_start:
            continue  # need at least one prior season to train on
        prior = features[features["season"] < s]
        this_season = features[features["season"] == s]
        if prior.empty or this_season.empty:
            continue
        sub_spec = fit_spec(prior)
        Xh, mh, _ = build_runs_matrix(this_season, sub_spec)
        Xa, ma, _ = build_runs_matrix_away(this_season, sub_spec)
        valid = mh & ma
        if valid.sum() == 0:
            continue
        rows = this_season.loc[valid].reset_index(drop=True)
        ph_X, pa_X = Xh[valid], Xa[valid]

        # Fit smaller-window models per season for OOF (chronological 85/15 tail)
        prior_sorted = prior.sort_values("game_date").reset_index(drop=True)
        vc = int(len(prior_sorted) * 0.85)
        tp, vp = prior_sorted.iloc[:vc], prior_sorted.iloc[vc:]
        Xht, mht, _ = build_runs_matrix(tp, sub_spec)
        Xat, mat, _ = build_runs_matrix_away(tp, sub_spec)
        Xhv, mhv, _ = build_runs_matrix(vp, sub_spec)
        Xav, mav, _ = build_runs_matrix_away(vp, sub_spec)
        oof_home = train_runs_model(
            Xht[mht], tp["target_home_score"].to_numpy()[mht], feat_cols,
            eval_X=Xhv[mhv], eval_y=vp["target_home_score"].to_numpy()[mhv],
        )
        oof_away = train_runs_model(
            Xat[mat], tp["target_away_score"].to_numpy()[mat], feat_cols,
            eval_X=Xav[mav], eval_y=vp["target_away_score"].to_numpy()[mav],
        )
        hm, hs = oof_home.predict_distribution(ph_X)
        am, ase = oof_away.predict_distribution(pa_X)
        tl = pd.to_numeric(rows["market_total_close"], errors="coerce").to_numpy(np.float64)
        preds = simulate_games(
            game_pks=rows["game_pk"].to_numpy(),
            pred_home=(hm, hs), pred_away=(am, ase),
            total_lines=tl, n_sims=_settings.monte_carlo_iterations,
            seed=_settings.random_seed + int(s),
        )
        runs_actual = (
            pd.to_numeric(rows["target_home_score"], errors="coerce")
            + pd.to_numeric(rows["target_away_score"], errors="coerce")
        ).to_numpy(np.float64)
        for i, p in enumerate(preds):
            oof_rows.append(
                {
                    "p_ml": p.p_home_win,
                    "y_ml": float(rows["target_home_win"].iloc[i]) if pd.notna(rows["target_home_win"].iloc[i]) else np.nan,
                    "p_rl": p.p_home_runline_cover,
                    "y_rl": float((rows["target_home_score"].iloc[i] - rows["target_away_score"].iloc[i]) > 1.5),
                    "p_ou": p.p_total_over,
                    "y_ou": float(runs_actual[i] > tl[i]) if np.isfinite(tl[i]) else np.nan,
                }
            )

    oof = pd.DataFrame(oof_rows)
    if not oof.empty:
        for market, p_col, y_col in [
            ("moneyline", "p_ml", "y_ml"),
            ("runline", "p_rl", "y_rl"),
            ("total", "p_ou", "y_ou"),
        ]:
            sub = oof.dropna(subset=[p_col, y_col])
            if len(sub) < 100:
                console.print(f"[yellow]Skipping {market} calibration (n={len(sub)})[/yellow]")
                continue
            cal = fit_calibrator(market, sub[p_col].to_numpy(), sub[y_col].to_numpy())
            save_calibrator(cal)
            console.print(
                f"[green]Calibrated {market}[/green] on {len(sub):,} OOF games"
            )

    console.print(
        f"[green]Trained on {len(features):,} games "
        f"({train_start}-{through_season}).[/green]"
    )


@app.command()
def backtest(
    start: Annotated[int, typer.Option(help="First target season")] = 2018,
    end: Annotated[int, typer.Option(help="Last target season")] = date_cls.today().year - 1,
    output: Annotated[str, typer.Option(help="Output CSV path")] = "backtest_results.csv",
) -> None:
    """Run walk-forward backtest and write per-season metrics to CSV."""
    configure_logging()
    from mlb_model.backtest.walkforward import run_walkforward

    results = run_walkforward(start, end)
    if results.empty:
        console.print("[red]Backtest produced no results.[/red]")
        raise typer.Exit(code=1)

    results.to_csv(output, index=False)
    console.print(f"[green]Wrote[/green] {output}")

    table = Table(title=f"Backtest {start}-{end}")
    for col in ["season", "n_games", "ml_accuracy", "ml_top10_acc", "ml_top3_acc",
                "rl_accuracy", "ou_accuracy", "clv_ml"]:
        table.add_column(col)
    for _, row in results.iterrows():
        table.add_row(
            str(row["season"]),
            str(row["n_games"]),
            f"{row['ml_accuracy']:.3f}",
            f"{row['ml_top10_acc']:.3f}",
            f"{row['ml_top3_acc']:.3f}",
            f"{row['rl_accuracy']:.3f}",
            f"{row['ou_accuracy']:.3f}",
            f"{row['clv_ml']:.2f}",
        )
    console.print(table)
    console.print(
        "\n[bold]Realistic targets:[/bold] ml_accuracy 0.55-0.58, "
        "ml_top10_acc 0.66-0.72, ml_top3_acc 0.70+, clv_ml > 0."
    )


@app.command()
def predict(
    date: Annotated[str, typer.Option(help="Target date YYYY-MM-DD, or 'today'")] = "today",
    refresh: Annotated[bool, typer.Option(help="Refresh data first")] = True,
) -> None:
    """Produce calibrated picks for ``date`` and write a CSV."""
    configure_logging()
    target = _parse_date(date)
    from mlb_model.predict.daily import predict_for_date, write_picks_csv

    picks = predict_for_date(target, refresh_data=refresh)
    if picks.empty:
        console.print("[yellow]No games to predict.[/yellow]")
        return

    path = write_picks_csv(picks, target)
    console.print(f"[green]Wrote picks to[/green] {path}")

    table = Table(title=f"Picks for {target.isoformat()}")
    for col in ["away_team_abbr", "home_team_abbr", "p_home_win",
                "p_home_runline_cover", "p_total_over", "confidence_ml"]:
        table.add_column(col)
    for _, row in picks.iterrows():
        table.add_row(
            str(row["away_team_abbr"]),
            str(row["home_team_abbr"]),
            f"{row['p_home_win']:.3f}",
            f"{row['p_home_runline_cover']:.3f}",
            f"{row['p_total_over']:.3f}" if not (isinstance(row["p_total_over"], float) and (row["p_total_over"] != row["p_total_over"])) else "n/a",
            f"{row['confidence_ml']:.3f}",
        )
    console.print(table)


@app.command("model-card")
def model_card(
    csv_path: Annotated[
        str, typer.Option(help="Backtest CSV path to summarize")
    ] = "logs/backtest_v4.csv",
) -> None:
    """Print an honest performance summary suitable for a model card."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    console.print(f"\n[bold]Model Card — backtest from {csv_path}[/bold]")

    agg = df[
        [
            "ml_accuracy", "ml_top3_acc", "ml_top10_acc", "ml_top30_acc",
            "rl_accuracy", "rl_top10_acc", "ou_accuracy", "ou_top10_acc",
            "clv_ml",
        ]
    ].mean()

    targets: dict[str, tuple[float, str]] = {
        "ml_accuracy":    (0.55, "Moneyline accuracy, all picks"),
        "ml_top30_acc":   (0.60, "Moneyline accuracy, top-30% confidence"),
        "ml_top10_acc":   (0.66, "Moneyline accuracy, top-10% confidence"),
        "ml_top3_acc":    (0.70, "Moneyline accuracy, top-3% conviction"),
        "rl_accuracy":    (0.55, "Run-line accuracy, all picks"),
        "rl_top10_acc":   (0.70, "Run-line accuracy, top-10% confidence"),
        "ou_accuracy":    (0.52, "Over/under accuracy, all picks"),
        "ou_top10_acc":   (0.55, "Over/under accuracy, top-10% confidence"),
        "clv_ml":         (0.0,  "Closing-line value vs market (cents/game)"),
    }

    table = Table(title=f"Backtest aggregate across {len(df)} seasons")
    table.add_column("Metric")
    table.add_column("Description")
    table.add_column("Target", justify="right")
    table.add_column("Measured", justify="right")
    table.add_column("Status")
    for metric, (target, desc) in targets.items():
        val = float(agg.get(metric, float("nan")))
        if val != val:
            status, value_str = "[grey]n/a[/grey]", "n/a"
        else:
            value_str = f"{val:.3f}" if metric != "clv_ml" else f"{val:+.2f}"
            status = (
                "[green]met[/green]" if val >= target else "[red]below[/red]"
            )
        target_str = f"{target:.3f}" if metric != "clv_ml" else f"{target:+.2f}"
        table.add_row(metric, desc, target_str, value_str, status)
    console.print(table)

    console.print("\n[bold]Per-season breakdown:[/bold]")
    season_table = Table()
    for col in [
        "season", "n_games", "ml_accuracy", "ml_top10_acc",
        "ml_top3_acc", "rl_top10_acc", "ou_accuracy", "clv_ml",
    ]:
        season_table.add_column(col)
    for _, row in df.iterrows():
        season_table.add_row(
            str(int(row["season"])),
            str(int(row["n_games"])),
            f"{row['ml_accuracy']:.3f}",
            f"{row['ml_top10_acc']:.3f}",
            f"{row['ml_top3_acc']:.3f}",
            f"{row['rl_top10_acc']:.3f}",
            f"{row['ou_accuracy']:.3f}" if pd.notna(row["ou_accuracy"]) else "n/a",
            f"{row['clv_ml']:+.2f}" if pd.notna(row["clv_ml"]) else "n/a",
        )
    console.print(season_table)


def main() -> None:  # entry-point alias for [project.scripts]
    app()


if __name__ == "__main__":
    main()
