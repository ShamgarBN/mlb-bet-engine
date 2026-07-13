"""Key-failover behaviour of the live odds fetcher (the-odds-api.com)."""

from __future__ import annotations

import httpx
import pytest

from mlb_model.data.sources import odds_api

QUOTA_401 = {
    "message": "Usage quota has been reached.",
    "error_code": "OUT_OF_USAGE_CREDITS",
}

SLATE = [
    {
        "id": "abc123",
        "sport_key": "baseball_mlb",
        "commence_time": "2026-07-16T23:05:00Z",
        "home_team": "NYY",
        "away_team": "BOS",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "NYY", "price": 1.8},
                            {"name": "BOS", "price": 2.1},
                        ],
                    },
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": 1.91, "point": 8.5},
                            {"name": "Under", "price": 1.91, "point": 8.5},
                        ],
                    },
                ],
            }
        ],
    }
]


def _install_transport(monkeypatch, handler):
    """Route httpx.Client() calls inside odds_api through a mock transport."""
    real_client = httpx.Client

    def fake_client(**kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(**kwargs)

    monkeypatch.setattr(odds_api.httpx, "Client", fake_client)


def _set_keys(monkeypatch, value):
    monkeypatch.setattr(odds_api.settings, "odds_api_key", value)
    monkeypatch.delenv("MLB_ODDS_API_KEY", raising=False)


def test_no_key_returns_empty(monkeypatch):
    _set_keys(monkeypatch, None)
    assert odds_api.fetch_live_slate().empty


def test_key_list_parsing(monkeypatch):
    _set_keys(monkeypatch, " key1 , key2,,key3 ")
    assert odds_api._api_keys() == ["key1", "key2", "key3"]


def test_single_key_success(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.url.params["apiKey"])
        return httpx.Response(
            200, json=SLATE, headers={"x-requests-remaining": "42"}
        )

    _set_keys(monkeypatch, "goodkey")
    _install_transport(monkeypatch, handler)
    df = odds_api.fetch_live_slate()
    assert seen == ["goodkey"]
    assert len(df) == 1
    row = df.iloc[0]
    assert row["home_team_abbr"] == "NYY"
    assert row["away_team_abbr"] == "BOS"
    assert row["total_close"] == 8.5


def test_failover_to_second_key(monkeypatch):
    seen = []

    def handler(request):
        key = request.url.params["apiKey"]
        seen.append(key)
        if key == "deadkey":
            return httpx.Response(401, json=QUOTA_401)
        return httpx.Response(200, json=SLATE)

    _set_keys(monkeypatch, "deadkey,goodkey")
    _install_transport(monkeypatch, handler)
    df = odds_api.fetch_live_slate()
    assert seen == ["deadkey", "goodkey"]
    assert len(df) == 1


def test_all_keys_exhausted_returns_empty(monkeypatch):
    def handler(request):
        return httpx.Response(401, json=QUOTA_401)

    _set_keys(monkeypatch, "deadkey1,deadkey2")
    _install_transport(monkeypatch, handler)
    assert odds_api.fetch_live_slate().empty


def test_rate_limited_key_fails_over(monkeypatch):
    seen = []

    def handler(request):
        key = request.url.params["apiKey"]
        seen.append(key)
        if key == "busykey":
            return httpx.Response(
                429, json={"error_code": "EXCEEDED_FREQ_LIMIT"}
            )
        return httpx.Response(200, json=SLATE)

    _set_keys(monkeypatch, "busykey,goodkey")
    _install_transport(monkeypatch, handler)
    assert len(odds_api.fetch_live_slate()) == 1
    assert seen == ["busykey", "goodkey"]


def test_server_error_does_not_rotate(monkeypatch):
    seen = []

    def handler(request):
        seen.append(request.url.params["apiKey"])
        return httpx.Response(500, text="boom")

    _set_keys(monkeypatch, "key1,key2")
    _install_transport(monkeypatch, handler)
    assert odds_api.fetch_live_slate().empty
    assert seen == ["key1"]
