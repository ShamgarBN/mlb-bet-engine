"""Unit tests for wind-component math.

The CF wind component is the highest-leverage weather feature; getting the
sign convention right is critical. Tests verify edge cases.
"""

from __future__ import annotations

import math

import pytest

from mlb_model.data.sources.weather import wind_component_to_cf


@pytest.mark.parametrize(
    "wind_speed,wind_from_deg,cf_bearing,expected",
    [
        # Wind FROM south (180) blows TOWARD north (0). If CF bearing = 0,
        # the wind blows directly out to CF -> +wind_speed.
        (10.0, 180.0, 0.0, 10.0),
        # Wind FROM north (0) -> blows toward south. CF=0 -> blowing IN -> -wind_speed.
        (10.0, 0.0, 0.0, -10.0),
        # Wind FROM east (90) -> blows west. CF=0 (north) -> perpendicular -> 0.
        (10.0, 90.0, 0.0, 0.0),
        # Calm wind
        (0.0, 0.0, 45.0, 0.0),
        # CF pointing east; wind from west (270) blows east -> +wind_speed
        (5.0, 270.0, 90.0, 5.0),
    ],
)
def test_wind_component_signs(wind_speed: float, wind_from_deg: float, cf_bearing: float, expected: float) -> None:
    got = wind_component_to_cf(wind_speed, wind_from_deg, cf_bearing)
    assert math.isclose(got, expected, abs_tol=1e-6), f"expected {expected}, got {got}"


def test_wind_component_handles_nan() -> None:
    assert math.isnan(wind_component_to_cf(float("nan"), 0.0, 0.0))
    assert math.isnan(wind_component_to_cf(10.0, float("nan"), 0.0))
    assert math.isnan(wind_component_to_cf(10.0, 0.0, float("nan")))
