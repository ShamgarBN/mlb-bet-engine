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
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mlb_model.config import settings
from mlb_model.logging import configure_logging, get_logger

app = typer.Typer(no_args_is_help=True, help="MLB prediction & betting model CLI")
data_app = typer.Typer(no_args_is_help=True, help="Data ingestion commands")
app.add_typer(data_app, name="data")
slugger_app = typer.Typer(
    no_args_is_help=True,
    help="Track big HR hitters who have gone cold (slumping-slugger analysis)",
)
app.add_typer(slugger_app, name="slugger")
alert_app = typer.Typer(
    no_args_is_help=True,
    help="Discord alerts for high-confidence (Premium/Strong) picks",
)
app.add_typer(alert_app, name="alert")

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


@data_app.command("ingest-odds")
def data_ingest_odds(
    path: Annotated[
        str,
        typer.Argument(help="File OR directory. Defaults to ./kaggle-data/"),
    ] = "kaggle-data",
) -> None:
    """Ingest historical/live odds from local files.

    Supports the Kaggle dataset pair (oddsDataMLB.csv + oddsData.csv) and
    the long-form snapshot dump (mlb_odds_kaggle_states.csv). Anything
    else under the directory is also tried via the generic CSV ingester.
    Safe to re-run -- upserts are keyed on (date, home, away, book).
    """
    configure_logging()
    from mlb_model.data.sources.odds_kaggle import ingest_all
    from mlb_model.data.sources import odds_csv

    target = Path(path)
    results: dict[str, int] = {}

    if target.is_file():
        n = odds_csv.ingest_csv(target)
        results[target.name] = n
    elif target.is_dir():
        # Try the Kaggle-specific layouts first, then sweep up anything
        # else with the generic CSV ingester.
        results.update(ingest_all(target))
        # Generic sweep for any odds CSVs not picked up by the Kaggle
        # ingester (skip files we already touched via the bespoke path).
        already = {"oddsDataMLB.csv", "oddsData.csv", "mlb_odds_kaggle_states.csv",
                   "mlb_odds_kaggle_states_summary.csv"}
        for sub_path in target.rglob("*"):
            if sub_path.is_dir():
                continue
            if sub_path.suffix.lower() not in {".csv", ".tsv", ".xlsx", ".xls"}:
                continue
            if sub_path.name in already:
                continue
            n = odds_csv.ingest_csv(sub_path)
            if n > 0:
                results[f"csv:{sub_path.name}"] = n
    else:
        console.print(f"[red]Path not found:[/red] {target}")
        raise typer.Exit(code=1)

    table = Table(title=f"Odds ingest from {target}")
    table.add_column("source")
    table.add_column("rows", justify="right")
    total = 0
    for k, v in results.items():
        table.add_row(str(k), f"{v:,}")
        total += v
    table.add_row("[bold]TOTAL[/bold]", f"[bold]{total:,}[/bold]")
    console.print(table)


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

    # Drop phantom rows: unplayed games (Scheduled / In Progress / Postponed /
    # Cancelled) have placeholder 0-0 scores in the warehouse, not NULL, so
    # the dropna above doesn't filter them. Restrict training to completed
    # games via the ``games.status`` column.
    from mlb_model.data.warehouse import query as _wh_query

    _final_pks = _wh_query(
        "SELECT game_pk FROM games WHERE status = 'Final'"
    )
    if not _final_pks.empty:
        before = len(features)
        features = features[features["game_pk"].isin(_final_pks["game_pk"])]
        console.print(
            f"[dim]Filtered to status=Final: {before:,} → {len(features):,} rows[/dim]"
        )

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
        # Build the prior-window training matrix FIRST, then lock the
        # encoded column list so every later matrix (validation tail and
        # this_season) is forced to the same shape. Without this, a
        # partial season with fewer categorical values produces a
        # narrower one-hot encoding and LightGBM fails at predict time.
        prior_sorted = prior.sort_values("game_date").reset_index(drop=True)
        vc = int(len(prior_sorted) * 0.85)
        tp, vp = prior_sorted.iloc[:vc], prior_sorted.iloc[vc:]
        Xht, mht, oof_feat_cols = build_runs_matrix(tp, sub_spec)
        sub_spec.final_feature_cols = oof_feat_cols
        Xat, mat, _ = build_runs_matrix_away(tp, sub_spec)
        Xhv, mhv, _ = build_runs_matrix(vp, sub_spec)
        Xav, mav, _ = build_runs_matrix_away(vp, sub_spec)
        Xh, mh, _ = build_runs_matrix(this_season, sub_spec)
        Xa, ma, _ = build_runs_matrix_away(this_season, sub_spec)
        valid = mh & ma
        if valid.sum() == 0:
            continue
        rows = this_season.loc[valid].reset_index(drop=True)
        ph_X, pa_X = Xh[valid], Xa[valid]
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

        # OOF totals classifier. We previously calibrated the O/U
        # market using the *simulator's* p_over, but at inference time
        # the production pipeline OVERWRITES that with the direct
        # LightGBM classifier whenever a market line exists. So the
        # calibrator was being trained on one distribution of inputs
        # and applied to a different one -- guaranteed miscalibration.
        # Train the OOF classifier on the same chronological prior data
        # and use its predictions to fit the totals calibrator. When
        # there's no market line we fall back to the sim's p_over
        # (which is also the inference fallback).
        oof_totals = None
        try:
            tp_tl = pd.to_numeric(tp["market_total_close"], errors="coerce").to_numpy(np.float64)
            tp_runs = (
                pd.to_numeric(tp["target_home_score"], errors="coerce")
                + pd.to_numeric(tp["target_away_score"], errors="coerce")
            ).to_numpy(np.float64)
            tp_y_ou = np.where(
                np.isfinite(tp_tl) & np.isfinite(tp_runs),
                (tp_runs > tp_tl).astype(np.float64),
                np.nan,
            )
            vp_tl = pd.to_numeric(vp["market_total_close"], errors="coerce").to_numpy(np.float64)
            vp_runs = (
                pd.to_numeric(vp["target_home_score"], errors="coerce")
                + pd.to_numeric(vp["target_away_score"], errors="coerce")
            ).to_numpy(np.float64)
            vp_y_ou = np.where(
                np.isfinite(vp_tl) & np.isfinite(vp_runs),
                (vp_runs > vp_tl).astype(np.float64),
                np.nan,
            )
            oof_totals = train_totals_model(
                Xht[mht], tp_y_ou[mht], feat_cols,
                X_val=Xhv[mhv], y_val=vp_y_ou[mhv],
            )
        except (ValueError, Exception) as exc:  # noqa: BLE001
            log.warning("oof_totals.train.failed", season=int(s), reason=str(exc))

        preds = simulate_games(
            game_pks=rows["game_pk"].to_numpy(),
            pred_home=(hm, hs), pred_away=(am, ase),
            total_lines=tl, n_sims=_settings.monte_carlo_iterations,
            seed=_settings.random_seed + int(s),
        )
        # Direct-classifier p_over on the validation slate (this season).
        # Used in place of the simulator's p_over whenever the market
        # line exists -- matching the inference path exactly.
        if oof_totals is not None:
            direct_p_over = oof_totals.predict(ph_X)
        else:
            direct_p_over = None

        runs_actual = (
            pd.to_numeric(rows["target_home_score"], errors="coerce")
            + pd.to_numeric(rows["target_away_score"], errors="coerce")
        ).to_numpy(np.float64)
        for i, p in enumerate(preds):
            has_market_line = bool(np.isfinite(tl[i]))
            if has_market_line and direct_p_over is not None:
                p_ou_for_calibration = float(direct_p_over[i])
            else:
                p_ou_for_calibration = float(p.p_total_over)
            oof_rows.append(
                {
                    "p_ml": p.p_home_win,
                    "y_ml": float(rows["target_home_win"].iloc[i]) if pd.notna(rows["target_home_win"].iloc[i]) else np.nan,
                    "p_rl": p.p_home_runline_cover,
                    "y_rl": float((rows["target_home_score"].iloc[i] - rows["target_away_score"].iloc[i]) > 1.5),
                    "p_ou": p_ou_for_calibration,
                    "y_ou": float(runs_actual[i] > tl[i]) if has_market_line else np.nan,
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


@app.command("morning-sync")
def morning_sync_cmd(
    force: Annotated[bool, typer.Option(help="Run even if already done today")] = False,
) -> None:
    """Pull yesterday's finalized scores + today's slate.

    Designed to be safe to run multiple times per day; it short-circuits
    if the marker file says we already ran. Use ``--force`` to override.
    """
    configure_logging()
    from mlb_model.automation import morning_sync

    if not force and not morning_sync.needs_run():
        console.print(
            f"[chrome]Morning sync already ran today "
            f"(last: {morning_sync.last_run_date()}). Use --force to re-run.[/chrome]"
        )
        return
    counts = morning_sync.run_morning_sync()
    console.print(counts)


@app.command("weekly-train")
def weekly_train_cmd(
    force: Annotated[bool, typer.Option(help="Run even if not yet Sunday or already trained")] = False,
    through_season: Annotated[
        int | None, typer.Option(help="Train through this season (default: current year)")
    ] = None,
) -> None:
    """Retrain home/away runs models + totals model + calibrators.

    Default policy: only runs on Sundays when the last training was ≥7
    days ago. Use ``--force`` to retrain right now. If the new model's
    rolling 14-day moneyline accuracy regresses by more than 2 pp, the
    previous artefacts are restored automatically.
    """
    configure_logging()
    from mlb_model.automation import weekly_train

    if not force and not weekly_train.needs_run():
        last = weekly_train.last_run_date()
        console.print(
            f"[chrome]Not Sunday yet, or last retrain ({last}) was less than 7 days ago. "
            f"Use --force to retrain now.[/chrome]"
        )
        return

    result = weekly_train.run_weekly_train(through_season=through_season)
    if not result.get("promoted"):
        console.print(f"[yellow]New model NOT promoted: {result}[/yellow]")
    else:
        console.print(f"[green]New model promoted: {result}[/green]")


@app.command("install-schedule")
def install_schedule_cmd(
    uninstall: Annotated[bool, typer.Option(help="Remove the LaunchAgents instead of installing them")] = False,
) -> None:
    """Install macOS LaunchAgents for morning sync (daily) + weekly retrain.

    Drops two ``.plist`` files in ``~/Library/LaunchAgents`` and loads
    them with ``launchctl``. The morning sync runs every day at 7:00am;
    weekly retraining runs Sundays at 3:00am.

    The desktop app also runs these lazily on launch as a backstop, so
    you can skip the LaunchAgent install if you'd rather not touch
    ``~/Library``.
    """
    configure_logging()
    from mlb_model.automation.scheduler import install, uninstall as _uninstall

    if uninstall:
        _uninstall()
        console.print("[green]LaunchAgents removed.[/green]")
        return
    paths = install()
    for label, path in paths.items():
        console.print(f"[green]Installed {label}:[/green] {path}")


@app.command("app")
def app_cmd(
    width: Annotated[int, typer.Option(help="Window width in pixels")] = 1280,
    height: Annotated[int, typer.Option(help="Window height in pixels")] = 860,
) -> None:
    """Launch the MLB Forecast app in a native macOS window.

    This is the "no browser, no terminal" path: an embedded WebKit
    window backed by a hidden local server. Close the window to exit.
    """
    configure_logging()
    try:
        from mlb_model.app.desktop import launch_native_window
    except ImportError as exc:
        console.print(
            "[red]pywebview is not installed.[/red] "
            "Install it with `uv add pywebview` or use `mlb-model serve` instead."
        )
        raise typer.Exit(code=1) from exc

    try:
        launch_native_window(width=width, height=height)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command("end-of-season")
def end_of_season_cmd(
    season: Annotated[int, typer.Option(help="Season to evaluate (e.g. 2026)")],
    force: Annotated[bool, typer.Option(help="Skip the 'is the regular season over?' check")] = False,
    skip_backtest: Annotated[
        bool, typer.Option(help="Skip the slow walk-forward backtest (journal-only report)")
    ] = False,
    train_start: Annotated[
        int | None,
        typer.Option(help="Earliest training season for the backtest (default: season - 6)"),
    ] = None,
) -> None:
    """Run the end-of-season full sweep: deep self-evaluation + written report.

    Writes ``reports/end_of_season_<SEASON>/`` containing report.md,
    summary.json, backtest.csv, and a frozen snapshot of the model that
    produced this season's live predictions. Used both manually after
    October baseball wraps and automatically by the weekly-train job.
    """
    configure_logging()
    from mlb_model.season import run_end_of_season_sweep

    try:
        report = run_end_of_season_sweep(
            season=season, force=force, skip_backtest=skip_backtest,
            train_start=train_start,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]End-of-season sweep complete:[/green] {report.output_dir}")
    console.print(f"  Markdown report: {report.report_md_path}")
    if report.backtest_csv_path:
        console.print(f"  Backtest CSV:    {report.backtest_csv_path}")
    if report.slices_csv_path:
        console.print(f"  Worst slices:    {report.slices_csv_path}")
    if report.model_snapshot_dir:
        console.print(f"  Model snapshot:  {report.model_snapshot_dir}")


@app.command("season-status")
def season_status_cmd() -> None:
    """Show the current season's state (in-progress / ended / off-season)."""
    configure_logging()
    from mlb_model.season import detect_season_state

    status = detect_season_state()
    table = Table(title=f"{status.season} status", show_header=False)
    table.add_row("State", status.state)
    table.add_row("Finalized games", str(status.finalized_games))
    table.add_row("Description", status.description)
    console.print(table)


@app.command("journal-grade")
def journal_grade_cmd(
    season: Annotated[
        int | None, typer.Option(help="Filter to a single season (default: most recent)")
    ] = None,
) -> None:
    """Grade the prediction journal and print per-market summaries.

    Read-only -- the journal itself is untouched. Useful for spot-checking
    "how is the model doing this week / season" from the terminal.
    """
    configure_logging()
    from mlb_model.journal import grade_journal, season_summary

    graded = grade_journal(season=season)
    if graded is None or graded.empty:
        console.print(
            "[yellow]No graded predictions yet -- run `mlb-model predict` "
            "and wait for the games to finalize.[/yellow]"
        )
        return

    summaries = season_summary(graded)
    if not summaries:
        console.print("[yellow]Journal exists but no finalized outcomes joined.[/yellow]")
        return

    table = Table(title="Prediction journal summary")
    table.add_column("Season")
    table.add_column("Market")
    table.add_column("N", justify="right")
    table.add_column("W-L-P", justify="right")
    table.add_column("Win %", justify="right")
    table.add_column("Units", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Log-loss", justify="right")
    for row in summaries:
        wr = row.win_rate
        wr_s = f"{wr*100:.1f}%" if wr is not None and wr == wr else "—"
        wlp = f"{row.wins}-{row.losses}"
        if row.pushes:
            wlp += f"-{row.pushes}"
        brier_s = f"{row.brier:.3f}" if row.brier == row.brier else "—"
        ll_s = f"{row.log_loss:.3f}" if row.log_loss == row.log_loss else "—"
        table.add_row(
            str(row.season), row.market, str(row.n), wlp, wr_s,
            f"{row.roi_units:+.2f}", brier_s, ll_s,
        )
    console.print(table)


@app.command("props-backfill")
def props_backfill_cmd(
    path: Annotated[
        str, typer.Option(help="Backtest parquet to import")
    ] = "logs/hitter_backtest_2021_2025_v2.parquet",
) -> None:
    """Seed the hitter-prop journal from the as-of backtest output.

    Lets the /season page show per-tier success rates over 5 seasons of
    historical games without waiting for the Matchups page to accumulate
    enough live visits.
    """
    configure_logging()
    from mlb_model.journal.props import backfill_from_backtest, grade_props

    n = backfill_from_backtest(path)
    grade_props()  # no-op for already-settled rows but cheap to run
    console.print(f"[green]Backfilled {n:,} hitter-prop rows from {path}[/green]")


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind address (local-only by default)")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="HTTP port")] = 8765,
    open_browser: Annotated[bool, typer.Option(help="Open the app in your default browser")] = True,
    reload: Annotated[bool, typer.Option(help="Auto-reload on code changes (dev only)")] = False,
) -> None:
    """Launch the local desktop forecast app.

    The app binds to 127.0.0.1 only -- it is not reachable from other
    machines on your network. Stops with Ctrl+C.
    """
    configure_logging()
    import threading
    import time
    import webbrowser

    import uvicorn

    # Guard against accidental public exposure.
    if host not in {"127.0.0.1", "localhost"}:
        console.print(
            f"[red]Refusing to bind to {host!r}. The desktop app is local-only.[/red]"
        )
        raise typer.Exit(code=2)

    url = f"http://{host}:{port}"
    console.print(f"[green]MLB Forecast running at[/green] {url}")
    console.print("Press Ctrl+C to stop.")

    if open_browser:
        # Open browser a moment after the server starts.
        def _open() -> None:
            time.sleep(0.8)
            try:
                webbrowser.open(url, new=2)
            except Exception:  # noqa: BLE001 -- browser may not be available
                pass

        threading.Thread(target=_open, daemon=True).start()

    uvicorn.run(
        "mlb_model.app.main:app",
        host=host,
        port=port,
        log_level="info",
        reload=reload,
        access_log=False,
    )


@slugger_app.command("percent")
def slugger_percent(
    season: Annotated[int, typer.Option(help="Season year")] = date_cls.today().year,
    threshold: Annotated[
        int | None, typer.Option(help="Fixed HR bar; omit for the dynamic top-N% bar")
    ] = None,
    target_pct: Annotated[
        float, typer.Option(help="Dynamic bar: top % of qualified hitters by HR")
    ] = 10.0,
    track: Annotated[
        bool, typer.Option(help="Append today's snapshot to the moving-history CSV")
    ] = True,
) -> None:
    """Share of players at/above the HR bar, across three denominators.

    The bar defaults to the top ``--target-pct`` of qualified hitters by HR
    (rises through the season); pass ``--threshold`` to force a fixed bar.
    With --track (default) it appends today's snapshot to the history CSV.
    """
    configure_logging()
    from mlb_model.analysis import slugger_slump as ss

    hitters = ss.fetch_season_hitting(season)
    if threshold is None:
        threshold = ss.dynamic_threshold(hitters, target_pct=target_pct)
        console.print(f"[dim]Dynamic bar: top {target_pct:g}% → [bold]{threshold}+ HR[/bold][/dim]")
    bd = ss.threshold_breakdown(hitters, season=season, threshold=threshold)

    table = Table(title=f"{season}: players with {threshold}+ HR  (n = {bd.n_at_threshold})")
    table.add_column("denominator")
    table.add_column("count", justify="right")
    table.add_column(f"{threshold}+ HR share", justify="right")
    for share in bd.shares.values():
        table.add_row(share.label, str(share.denom_count), f"{share.pct:.1f}%")
    console.print(table)

    if track:
        path = ss.append_history(bd)
        console.print(f"[green]Snapshot saved →[/green] {path}")


@slugger_app.command("history")
def slugger_history(
    season: Annotated[int, typer.Option(help="Season year")] = date_cls.today().year,
    threshold: Annotated[int, typer.Option(help="HR threshold")] = 15,
    denominator: Annotated[
        str, typer.Option(help="all_pa | has_hr | qualified")
    ] = "qualified",
) -> None:
    """Print the recorded moving-percentage series for one denominator."""
    configure_logging()
    import csv as _csv

    from mlb_model.analysis import slugger_slump as ss

    if not ss.HISTORY_PATH.exists():
        console.print("[yellow]No history yet. Run 'slugger percent' first.[/yellow]")
        raise typer.Exit()

    with ss.HISTORY_PATH.open(newline="") as fh:
        rows = [
            r for r in _csv.DictReader(fh)
            if int(r["season"]) == season
            and int(r["threshold"]) == threshold
            and r["denominator"] == denominator
        ]
    if not rows:
        console.print("[yellow]No matching snapshots recorded.[/yellow]")
        raise typer.Exit()

    table = Table(title=f"{season}: {threshold}+ HR share over time ({denominator})")
    table.add_column("as of")
    table.add_column("at threshold", justify="right")
    table.add_column("denom", justify="right")
    table.add_column("share", justify="right")
    for r in sorted(rows, key=lambda x: x["as_of"]):
        table.add_row(r["as_of"], r["n_at_threshold"], r["denom_count"], f"{float(r['pct']):.1f}%")
    console.print(table)


@slugger_app.command("slumps")
def slugger_slumps(
    season: Annotated[int, typer.Option(help="Season year")] = date_cls.today().year,
    threshold: Annotated[
        int | None, typer.Option(help="Fixed HR bar; omit for the dynamic top-N% bar")
    ] = None,
    target_pct: Annotated[
        float, typer.Option(help="Dynamic bar: top % of qualified hitters by HR")
    ] = 10.0,
    min_drought: Annotated[
        int, typer.Option(help="Flag players with this many+ HR-less games")
    ] = 5,
    no_news: Annotated[bool, typer.Option(help="Skip the live news lookup")] = False,
    no_matchups: Annotated[
        bool, typer.Option(help="Skip the next-game pitching-matchup lookup")
    ] = False,
    csv_out: Annotated[
        str | None, typer.Option(help="Optional path to write the report as CSV")
    ] = None,
) -> None:
    """Find slumping sluggers above the HR bar and explain why.

    The bar defaults to the top ``--target-pct`` of qualified hitters by HR
    (rises through the season; pass ``--threshold`` for a fixed bar). Flags
    cold streaks + absences, verifies the cause against the MLB transactions
    feed (INJURY / EXTERNAL only when MLB logged the move; else UNCLEAR), and
    for bettable players grades the next opposing pitcher and whether they
    play today. News is context only.
    """
    configure_logging()
    from mlb_model.analysis import slugger_slump as ss

    with console.status("Pulling season totals, game logs, matchups, and news…"):
        hitters = ss.fetch_season_hitting(season)
        if threshold is None:
            threshold = ss.dynamic_threshold(hitters, target_pct=target_pct)
        flagged = ss.find_slumping_sluggers(
            season, threshold=threshold, min_drought=min_drought,
            with_news=not no_news, with_matchups=not no_matchups, hitters=hitters,
        )
    console.print(f"[dim]HR bar: [bold]{threshold}+[/bold] (top ~{target_pct:g}% of qualified hitters)[/dim]")

    if not flagged:
        console.print(
            f"[green]No {threshold}+ HR hitters are in a {min_drought}+ game drought today.[/green]"
        )
        raise typer.Exit()

    _color = {
        "INJURY (verified)": "red",
        "EXTERNAL (verified)": "yellow",
        "UNCLEAR": "chrome",
    }
    _mcolor = {"FAVORABLE": "green", "TOUGH": "red", "NEUTRAL": "chrome"}
    table = Table(
        title=f"{season}: {threshold}+ HR sluggers gone cold ({len(flagged)} flagged)"
    )
    table.add_column("player")
    table.add_column("HR", justify="right")
    table.add_column("drought", justify="right")
    table.add_column("last HR")
    table.add_column("status")
    table.add_column("next matchup", max_width=40)
    for s in flagged:
        drought_txt = "—" if s.is_absent and s.drought_games == 0 else f"{s.drought_games}G"
        label = s.status_label
        styled = f"[{_color.get(label, 'white')}]{label}[/]" if label != "UNCLEAR" else label
        # Matchup column for bettable players; injury/external just show the cause.
        if label == "UNCLEAR" and s.matchup_label != "NONE":
            mtag = f"[{_mcolor.get(s.matchup_label, 'white')}]{s.matchup_label.title()}[/]"
            matchup = f"{mtag} vs {s.matchup_pitcher}" if s.matchup_pitcher else mtag
        elif label == "UNCLEAR":
            matchup = "[chrome]no probable yet[/]"
        else:
            matchup = s.advisory.split(" — ", 1)[-1] if " — " in s.advisory else "—"
        table.add_row(
            f"{s.name} ({s.team})" if s.team else s.name,
            str(s.home_runs),
            drought_txt,
            s.last_hr_date.isoformat() if s.last_hr_date else "—",
            styled,
            matchup,
        )
    console.print(table)

    # Supporting news context (informational only — never drives the verdict).
    if not no_news:
        for s in flagged:
            if s.news_headlines:
                console.print(f"\n[bold]{s.name}[/bold] — recent news:")
                for h in s.news_headlines:
                    when = h.published.date().isoformat() if h.published else "?"
                    console.print(f"  • [{when}] {h.title}")

    if csv_out:
        import csv as _csv

        path = Path(csv_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [ss.status_to_row(s) for s in flagged]
        with path.open("w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        console.print(f"\n[green]Report written →[/green] {path}")


@alert_app.command("test")
def alert_test() -> None:
    """Send a test message to the configured Discord webhook."""
    configure_logging()
    from mlb_model.automation import alerts

    if not settings.discord_webhook_url:
        console.print(
            "[red]No webhook configured.[/red] Add MLB_DISCORD_WEBHOOK_URL to your .env "
            "(Discord → Edit Channel → Integrations → Webhooks → New Webhook → Copy URL)."
        )
        raise typer.Exit(code=1)
    ok = alerts.post_to_discord("✅ MLB Forecast test alert — your webhook is wired up.")
    console.print("[green]Sent![/green]" if ok else "[red]Failed to post — check logs/.[/red]")


@alert_app.command("run")
def alert_run(
    date: Annotated[str, typer.Option(help="Slate date (today/yesterday/YYYY-MM-DD)")] = "today",
    force: Annotated[bool, typer.Option(help="Ignore the per-day 'already sent' marker")] = False,
    dry_run: Annotated[bool, typer.Option(help="Build + print the message, don't post")] = False,
    full: Annotated[
        bool,
        typer.Option(help="Print the FULL uncapped pick list to the terminal (never posts)"),
    ] = False,
) -> None:
    """Build today's Premium/Strong alert and post it (or preview with --dry-run).

    Use --full when the Discord message was capped ("…and N more") to print the
    complete list here — it never posts, just prints.
    """
    configure_logging()
    from mlb_model.automation import alerts

    target = _parse_date(date)
    # --full is a local, read-only listing: force so it isn't skipped by the
    # daily marker, and never post (dry_run) — just print everything.
    with console.status("Gathering high-confidence game + prop picks…"):
        result = alerts.send_daily_alert(
            target,
            force=force or full,
            dry_run=dry_run or full,
            uncapped=full,
        )
    console.print(
        f"game picks: {result.get('n_game', 0)} · prop picks: {result.get('n_prop', 0)} · "
        f"sent: {result.get('sent')}"
        + (f" · {result['reason']}" if result.get("reason") else "")
    )
    if result.get("message"):
        console.print("\n[dim]--- message ---[/dim]")
        console.print(result["message"])


def main() -> None:  # entry-point alias for [project.scripts]
    app()


if __name__ == "__main__":
    main()
