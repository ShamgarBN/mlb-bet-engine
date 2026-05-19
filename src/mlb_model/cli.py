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
    train_start: Annotated[int, typer.Option(help="Earliest season to include")] = 2016,
) -> None:
    """Train and persist the runs models from [train_start, through_season]."""
    configure_logging()
    from mlb_model.features.assemble import build_features_table
    from mlb_model.model.feature_matrix import (
        build_runs_matrix,
        build_runs_matrix_away,
        fit_spec,
    )
    from mlb_model.model.runs import save_model, train_runs_model

    features = build_features_table(train_start, through_season).dropna(
        subset=["target_home_score", "target_away_score"]
    )
    if features.empty:
        console.print("[red]No training data available -- run `data pull` first.[/red]")
        raise typer.Exit(code=1)

    spec = fit_spec(features)
    X_home, mh, feat_cols = build_runs_matrix(features, spec)
    X_away, ma, _ = build_runs_matrix_away(features, spec)
    y_home = features["target_home_score"].to_numpy()
    y_away = features["target_away_score"].to_numpy()

    home_model = train_runs_model(X_home[mh], y_home[mh], feat_cols)
    away_model = train_runs_model(X_away[ma], y_away[ma], feat_cols)
    save_model(home_model, "home_runs")
    save_model(away_model, "away_runs")
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


def main() -> None:  # entry-point alias for [project.scripts]
    app()


if __name__ == "__main__":
    main()
