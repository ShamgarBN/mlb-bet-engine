"""Smoke tests for the local desktop app.

We use FastAPI's ``TestClient`` to hit the same endpoints the browser
does, with no need to spin up uvicorn. Tests are fast (sub-second) and
isolate the *routing + template + service layer* from the prediction
pipeline. The prediction pipeline itself is exercised once -- if it
fails we get a 200 response with an error banner, not a 500.
"""

from __future__ import annotations

from datetime import date

import pytest
from fastapi.testclient import TestClient

from mlb_model.app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_dashboard_loads(client: TestClient) -> None:
    """Dashboard renders even when there are no cached predictions."""
    r = client.get("/", params={"date": "2099-01-01"})  # winter — no games
    assert r.status_code == 200
    assert "MLB Forecast" in r.text
    # No games or error -- one of these two must be present.
    assert "No predictions match" in r.text or "Prediction pipeline error" in r.text


def test_filter_params_accepted(client: TestClient) -> None:
    """Every market/tier combination should produce a 200 response."""
    for market in ("moneyline", "runline", "total"):
        for tier in ("all", "lean", "edge", "strong", "premium"):
            r = client.get(
                "/",
                params={"date": "2099-01-01", "market": market, "tier": tier},
            )
            assert r.status_code == 200, f"{market}/{tier} -> {r.status_code}"


def test_performance_page(client: TestClient) -> None:
    r = client.get("/performance")
    assert r.status_code == 200
    assert "Performance" in r.text


def test_log_page(client: TestClient) -> None:
    r = client.get("/log")
    assert r.status_code == 200
    assert "My picks" in r.text


def test_picks_log_validation(client: TestClient) -> None:
    """Missing required fields must reject with 422."""
    r = client.post("/api/picks-log/add", json={"game_pk": 1})
    assert r.status_code == 422


def test_picks_log_round_trip(client: TestClient, tmp_path, monkeypatch) -> None:
    """Round-trip: POST a pick, GET the log page, see it listed."""
    # Redirect the picks-log persistence to a temp file so tests don't
    # touch the user's real log.
    from mlb_model.app import services

    monkeypatch.setattr(services, "PICKS_LOG_PATH", tmp_path / "picks_log.parquet")

    payload = {
        "game_pk": 99999,
        "game_date": date.today().isoformat(),
        "away_team_abbr": "TST",
        "home_team_abbr": "TGT",
        "market": "moneyline",
        "pick": "HOME",
        "pick_long": "TGT ML",
        "model_prob": 0.62,
    }
    r = client.post("/api/picks-log/add", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body.get("pick_id")  # non-empty pick_id returned

    r = client.get("/log")
    assert r.status_code == 200
    # The exact row may show as 'pending' since game 99999 doesn't exist.
    assert "TGT ML" in r.text or "Total picks" in r.text


def test_dashboard_bad_date(client: TestClient) -> None:
    r = client.get("/", params={"date": "not-a-date"})
    assert r.status_code == 400


def test_pick_delete_and_update(client: TestClient, tmp_path, monkeypatch) -> None:
    """A logged pick can be retrieved, edited, then deleted via the API."""
    from mlb_model.app import services

    monkeypatch.setattr(services, "PICKS_LOG_PATH", tmp_path / "picks_log.parquet")
    services.clear_logged_picks()

    payload = {
        "game_pk": 12345,
        "game_date": date.today().isoformat(),
        "away_team_abbr": "AAA",
        "home_team_abbr": "BBB",
        "market": "moneyline",
        "pick": "AWAY",
        "pick_long": "AAA ML",
        "model_prob": 0.60,
    }
    r = client.post("/api/picks-log/add", json=payload)
    assert r.status_code == 200
    pick_id = r.json()["pick_id"]
    assert pick_id

    # Edit stake
    r = client.post(
        "/api/picks-log/update",
        json={"pick_id": pick_id, "stake_units": 2.0},
    )
    assert r.status_code == 200

    df = services._read_log_raw()  # type: ignore[attr-defined]
    assert df.loc[df["pick_id"] == pick_id, "stake_units"].iloc[0] == 2.0

    # Delete
    r = client.post("/api/picks-log/delete", json={"pick_id": pick_id})
    assert r.status_code == 200

    df = services._read_log_raw()  # type: ignore[attr-defined]
    assert df.empty

    # Deleting again is a 404
    r = client.post("/api/picks-log/delete", json={"pick_id": pick_id})
    assert r.status_code == 404


def test_pick_update_validation(client: TestClient) -> None:
    r = client.post("/api/picks-log/update", json={"pick_id": "abc", "stake_units": -1})
    assert r.status_code == 422
    r = client.post("/api/picks-log/update", json={})
    assert r.status_code == 422


def test_totals_baseline_fallback() -> None:
    """``league_avg_total`` returns a sane value even when warehouse is empty."""
    from mlb_model.predict.totals_baseline import HARD_FALLBACK, league_avg_total

    val = league_avg_total(1999)  # season we never ingested
    assert val == HARD_FALLBACK


def test_automation_marker_lifecycle(tmp_path, monkeypatch) -> None:
    """``last_run_date`` / ``needs_run`` should reflect the marker file."""
    from mlb_model.automation import morning_sync, weekly_train

    marker_a = tmp_path / "morning.txt"
    marker_b = tmp_path / "weekly.txt"
    monkeypatch.setattr(morning_sync, "MARKER_PATH", marker_a)
    monkeypatch.setattr(weekly_train, "MARKER_PATH", marker_b)

    assert morning_sync.last_run_date() is None
    assert morning_sync.needs_run() is True

    morning_sync._record_success()
    assert morning_sync.last_run_date() == date.today()
    assert morning_sync.needs_run() is False

    # Weekly only "needs_run" on Sundays.
    assert weekly_train.last_run_date() is None
    is_sunday = date.today().weekday() == 6
    assert weekly_train.needs_run() is is_sunday


def test_scheduler_plist_payload(tmp_path, monkeypatch) -> None:
    """The LaunchAgent installer should produce loadable plist files."""
    import plistlib

    from mlb_model.automation import scheduler

    monkeypatch.setattr(scheduler, "LAUNCH_AGENTS_DIR", tmp_path)
    monkeypatch.setattr(scheduler, "_resolve_uv", lambda: "/usr/local/bin/uv")
    # Don't actually invoke launchctl in tests.
    monkeypatch.setattr(scheduler, "_launchctl", lambda *args, **kwargs: None)

    paths = scheduler.install()
    assert set(paths) == {
        scheduler.MORNING_LABEL, scheduler.WEEKLY_LABEL, scheduler.AFTERNOON_LABEL,
        scheduler.RECAP_LABEL,
    }

    for label, path in paths.items():
        assert path.exists()
        with path.open("rb") as f:
            payload = plistlib.load(f)
        assert payload["Label"] == label
        assert payload["WorkingDirectory"]
        assert "uv" in payload["ProgramArguments"][0]
        assert "mlb-model" in payload["ProgramArguments"]
        assert "StartCalendarInterval" in payload

    # Uninstall removes them.
    scheduler.uninstall()
    for path in paths.values():
        assert not path.exists()


def test_p_total_over_at_line_monotone() -> None:
    """P(over) must strictly decrease as line increases for a fixed game."""
    from mlb_model.app.services import p_total_over_at_line

    p_low = p_total_over_at_line(4.0, 2.5, 4.0, 2.5, line=6.0)
    p_mid = p_total_over_at_line(4.0, 2.5, 4.0, 2.5, line=8.0)
    p_high = p_total_over_at_line(4.0, 2.5, 4.0, 2.5, line=10.0)
    assert p_low > p_mid > p_high


def test_serve_refuses_public_bind() -> None:
    """The CLI must refuse to bind to anything other than localhost."""
    from typer.testing import CliRunner

    from mlb_model.cli import app as cli_app

    runner = CliRunner()
    result = runner.invoke(cli_app, ["serve", "--host", "0.0.0.0"])
    assert result.exit_code == 2
    assert "local-only" in result.stdout.lower()
