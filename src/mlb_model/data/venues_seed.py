"""Hand-curated metadata for all 30 MLB home venues.

The schedule API does NOT include latitude, longitude, dome/roof type, or
park orientation (CF compass bearing) — those are seeded here so weather
lookups and the CF-wind feature work from day one.

Coordinates are public-knowledge stadium centroids. ``cf_bearing_deg`` is
the compass bearing FROM home plate TO straight-away center field; values
are approximate but accurate enough for the wind-projection feature.
``altitude_ft`` matters for COL (and to a lesser extent ARI, ATL).

Re-running the model's training pipeline overwrites the venues table from
this file, so corrections here will propagate on the next ``data init``.
"""

from __future__ import annotations

import pandas as pd

# (team_abbr, name, latitude, longitude, altitude_ft, is_dome, roof_type,
#  cf_bearing_deg, cf_distance_ft)
_VENUES: list[tuple[str, str, float, float, int, bool, str, float, int]] = [
    ("ARI", "Chase Field",                        33.4453, -112.0667, 1086, True,  "retractable", 0.0,   407),
    ("ATL", "Truist Park",                        33.8908,  -84.4678,  1050, False, "open",        25.0,  400),
    ("BAL", "Oriole Park at Camden Yards",        39.2839,  -76.6217,    20, False, "open",        20.0,  400),
    ("BOS", "Fenway Park",                        42.3467,  -71.0972,    21, False, "open",        45.0,  420),
    ("CHC", "Wrigley Field",                      41.9484,  -87.6553,   600, False, "open",         5.0,  400),
    ("CWS", "Guaranteed Rate Field",              41.8300,  -87.6339,   595, False, "open",        37.0,  400),
    ("CIN", "Great American Ball Park",           39.0974,  -84.5076,   490, False, "open",        25.0,  404),
    ("CLE", "Progressive Field",                  41.4962,  -81.6852,   660, False, "open",         0.0,  410),
    ("COL", "Coors Field",                        39.7559, -104.9942,  5200, False, "open",         0.0,  415),
    ("DET", "Comerica Park",                      42.3390,  -83.0485,   600, False, "open",        15.0,  420),
    ("HOU", "Minute Maid Park",                   29.7572,  -95.3553,    50, True,  "retractable", 21.0,  409),
    ("KC",  "Kauffman Stadium",                   39.0517,  -94.4803,   750, False, "open",         0.0,  410),
    ("LAA", "Angel Stadium",                      33.8003, -117.8827,   160, False, "open",        45.0,  400),
    ("LAD", "Dodger Stadium",                     34.0739, -118.2400,   500, False, "open",         0.0,  395),
    ("MIA", "loanDepot park",                     25.7781,  -80.2197,    10, True,  "retractable", 67.0,  407),
    ("MIL", "American Family Field",              43.0280,  -87.9712,   635, True,  "retractable", 20.0,  400),
    ("MIN", "Target Field",                       44.9817,  -93.2776,   815, False, "open",       100.0,  404),
    ("NYM", "Citi Field",                         40.7571,  -73.8458,    10, False, "open",        20.0,  408),
    ("NYY", "Yankee Stadium",                     40.8296,  -73.9262,    20, False, "open",         0.0,  408),
    ("OAK", "Oakland Coliseum",                   37.7516, -122.2008,    50, False, "open",         0.0,  400),
    ("PHI", "Citizens Bank Park",                 39.9061,  -75.1665,    40, False, "open",        40.0,  401),
    ("PIT", "PNC Park",                           40.4469,  -80.0057,   730, False, "open",        90.0,  399),
    ("SD",  "Petco Park",                         32.7073, -117.1566,    13, False, "open",         0.0,  396),
    ("SEA", "T-Mobile Park",                      47.5914, -122.3325,    20, True,  "retractable", 21.0,  401),
    ("SF",  "Oracle Park",                        37.7786, -122.3893,    13, False, "open",        61.0,  399),
    ("STL", "Busch Stadium",                      38.6226,  -90.1928,   465, False, "open",        45.0,  400),
    ("TB",  "Tropicana Field",                    27.7682,  -82.6534,    15, True,  "fixed",       45.0,  404),
    ("TEX", "Globe Life Field",                   32.7474,  -97.0817,   550, True,  "retractable", 16.0,  407),
    ("TOR", "Rogers Centre",                      43.6414,  -79.3894,   265, True,  "retractable",  0.0,  400),
    ("WSH", "Nationals Park",                     38.8730,  -77.0074,    40, False, "open",        25.0,  402),
]


def seed_venues_df() -> pd.DataFrame:
    """Return all 30 venues as a DataFrame, schema-aligned with the warehouse."""
    return pd.DataFrame(
        _VENUES,
        columns=[
            "team_abbr",
            "name",
            "latitude",
            "longitude",
            "altitude_ft",
            "is_dome",
            "roof_type",
            "cf_bearing_deg",
            "cf_distance_ft",
        ],
    )
