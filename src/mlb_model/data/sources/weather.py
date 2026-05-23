"""Weather ingestion: Open-Meteo / NOAA archive + forecast, plus the
out-to-center-field wind component used as a model feature.

The wind component is the highest-leverage weather signal in the model.
Sign convention: positive = blowing OUT to CF (helps fly balls carry),
negative = blowing IN from CF (suppresses runs).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING

import pandas as pd

from mlb_model.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    pass

log = get_logger("data.sources.weather")


def wind_component_to_cf(
    wind_speed: float, wind_from_deg: float, cf_bearing: float
) -> float:
    """Project wind onto the home-plate→center-field axis.

    Meteorological convention: ``wind_from_deg`` is the compass bearing the
    wind is blowing FROM (0 = north, 90 = east, 180 = south, 270 = west).
    ``cf_bearing`` is the compass bearing FROM home plate TO center field
    for the park (e.g. Fenway's CF is roughly NE).

    A wind blowing FROM south (180°) into a park whose CF points north (0°)
    is blowing directly out to CF → returns +wind_speed.
    A wind blowing FROM north (0°) into the same park is blowing IN from
    CF → returns -wind_speed.
    Perpendicular wind → 0.
    """
    if (
        math.isnan(wind_speed)
        or math.isnan(wind_from_deg)
        or math.isnan(cf_bearing)
    ):
        return float("nan")
    # The wind blows TOWARD (wind_from_deg + 180) mod 360.
    # Project that "toward" vector onto the CF bearing unit vector.
    toward_deg = (wind_from_deg + 180.0) % 360.0
    delta_rad = math.radians(toward_deg - cf_bearing)
    return wind_speed * math.cos(delta_rad)


def _open_meteo_url(lat: float, lon: float, when: date, is_archive: bool) -> str:
    """Pick archive vs forecast endpoint based on how far back the date is."""
    base = (
        "https://archive-api.open-meteo.com/v1/archive"
        if is_archive
        else "https://api.open-meteo.com/v1/forecast"
    )
    iso = when.isoformat()
    return (
        f"{base}?latitude={lat:.4f}&longitude={lon:.4f}"
        f"&start_date={iso}&end_date={iso}"
        "&hourly=temperature_2m,relative_humidity_2m,surface_pressure,"
        "wind_speed_10m,wind_direction_10m&temperature_unit=fahrenheit"
        "&wind_speed_unit=mph&timezone=auto"
    )


def _pick_endpoint(when: date, today: date | None = None) -> bool:
    """Return True if the archive endpoint should be used.

    Open-Meteo's archive has ~5-day lag; anything within the last week
    goes to the forecast endpoint to be safe.
    """
    today = today or date.today()
    return when < today - timedelta(days=7)


def _fetch_hourly(lat: float, lon: float, when: date) -> dict | None:
    """Fetch a single day's hourly weather. Best-effort: returns None on error."""
    import httpx

    url = _open_meteo_url(lat, lon, when, is_archive=_pick_endpoint(when))
    try:
        with httpx.Client(timeout=20.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.json()
    except Exception:  # noqa: BLE001 -- per-game weather is best-effort
        log.warning("weather.fetch.failed", lat=lat, lon=lon, date=when.isoformat())
        return None


def _summarize_at_hour(payload: dict, target_hour: int) -> dict[str, float]:
    """Reduce a 24-hour Open-Meteo response to a single point estimate."""
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return {}
    # Find the hourly index closest to target_hour
    idx = 0
    best = 99
    for i, ts in enumerate(times):
        try:
            hour = int(ts.split("T")[1].split(":")[0])
        except (IndexError, ValueError):
            continue
        if abs(hour - target_hour) < best:
            best = abs(hour - target_hour)
            idx = i

    def _at(key: str) -> float:
        seq = hourly.get(key) or []
        if idx >= len(seq) or seq[idx] is None:
            return float("nan")
        try:
            return float(seq[idx])
        except (TypeError, ValueError):
            return float("nan")

    return {
        "temp_f": _at("temperature_2m"),
        "humidity_pct": _at("relative_humidity_2m"),
        "pressure_hpa": _at("surface_pressure"),
        "wind_speed_mph": _at("wind_speed_10m"),
        "wind_from_deg": _at("wind_direction_10m"),
    }


def _scheduled_hour(scheduled_start: str | None) -> int:
    """Extract the local-hour-of-day from an ISO timestamp; default 19 (7pm)."""
    if not scheduled_start:
        return 19
    try:
        dt = datetime.fromisoformat(str(scheduled_start).replace("Z", "+00:00"))
        return dt.hour
    except (TypeError, ValueError):
        return 19


def ingest_weather_for_games(
    games_df: pd.DataFrame, venues_df: pd.DataFrame
) -> int:
    """Fetch + upsert weather rows for games missing them.

    ``games_df`` must include: game_pk, scheduled_start, home_team_abbr
    (and optionally venue_id). ``venues_df`` comes from ``seed_venues_df()``
    and must include team_abbr, latitude, longitude, is_dome, cf_bearing_deg.

    Dome games are written with NaN weather + ``is_dome=True`` so downstream
    feature builders can branch on that without re-fetching.
    """
    if games_df is None or games_df.empty:
        return 0

    from mlb_model.data.warehouse import upsert_dataframe

    venues_lookup = venues_df.set_index("team_abbr") if not venues_df.empty else None
    rows: list[dict] = []
    for _, g in games_df.iterrows():
        pk = int(g.get("game_pk"))
        team_abbr = str(g.get("home_team_abbr") or "")
        if not team_abbr or venues_lookup is None or team_abbr not in venues_lookup.index:
            continue
        v = venues_lookup.loc[team_abbr]
        is_dome = bool(v.get("is_dome", False))
        row: dict = {
            "game_pk": pk,
            "temp_f": float("nan"),
            "humidity_pct": float("nan"),
            "pressure_hpa": float("nan"),
            "wind_speed_mph": float("nan"),
            "wind_out_to_cf": float("nan"),
            "is_dome": is_dome,
        }
        if is_dome:
            rows.append(row)
            continue

        # Parse the date out of scheduled_start (or game_date if available)
        sched = g.get("scheduled_start") or g.get("game_date")
        try:
            when = pd.to_datetime(sched).date()
        except (TypeError, ValueError):
            rows.append(row)
            continue

        payload = _fetch_hourly(
            lat=float(v["latitude"]), lon=float(v["longitude"]), when=when
        )
        if not payload:
            rows.append(row)
            continue

        summary = _summarize_at_hour(payload, _scheduled_hour(g.get("scheduled_start")))
        row.update({k: summary.get(k, float("nan")) for k in
                    ("temp_f", "humidity_pct", "pressure_hpa", "wind_speed_mph")})
        cf_bearing = float(v.get("cf_bearing_deg", float("nan")))
        row["wind_out_to_cf"] = wind_component_to_cf(
            row["wind_speed_mph"], summary.get("wind_from_deg", float("nan")), cf_bearing
        )
        rows.append(row)

    if not rows:
        return 0
    df = pd.DataFrame(rows)
    return upsert_dataframe(df, "weather", key_columns=["game_pk"])
