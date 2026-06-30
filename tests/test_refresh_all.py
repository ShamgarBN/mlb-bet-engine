"""Tests for the whole-tool refresh SSE endpoint (offline, mocked pipeline)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from mlb_model.app import services
from mlb_model.app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_refresh_all_streams_progress_and_done(client, monkeypatch):
    # Stub the heavy pipeline with a tiny deterministic generator.
    monkeypatch.setattr(
        services,
        "refresh_all",
        lambda target: iter([(0.0, "Step one"), (0.5, "Step one"), (1.0, "Step two")]),
    )
    body = client.get("/api/refresh-all?date=2026-06-30").text
    events = [ln[len("data: "):] for ln in body.splitlines() if ln.startswith("data: ")]
    assert any('"pct": 0.0' in e and "Step one" in e for e in events)
    assert any('"pct": 1.0' in e for e in events)
    # A terminal done event is always appended.
    assert events[-1].strip().endswith("}") and '"done": true' in events[-1]


def test_refresh_all_reports_error_in_stream(client, monkeypatch):
    def _boom(target):
        raise RuntimeError("kaboom")
        yield  # pragma: no cover -- makes this a generator

    monkeypatch.setattr(services, "refresh_all", _boom)
    body = client.get("/api/refresh-all").text
    assert "kaboom" in body and "error" in body
