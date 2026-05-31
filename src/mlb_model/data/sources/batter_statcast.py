"""Per-batter season Statcast aggregates from Baseball Savant.

Two pybaseball helpers cover what the Matchups page needs:

  * ``statcast_batter_expected_stats``  -> xBA, xSLG, xwOBA, PA, BIP
  * ``statcast_batter_exitvelo_barrels`` -> avg EV, HardHit% (ev95percent),
                                            Barrel% (brl_percent)

Both are single HTTP pulls (~1s each) and refresh nightly on Savant after
games settle. We aggregate into ``batter_statcast_season`` keyed by
(player_id, season) so the scoring layer can join it to the per-season
batter aggregates in one query.

The ingest is idempotent (UPSERT) and safe to call repeatedly. The
Matchups page calls :func:`maybe_refresh` on each visit, which respects
a 12-hour throttle marker so the actual HTTP work happens at most
twice a day.
"""

from __future__ import annotations

from datetime import date as date_cls, datetime
from pathlib import Path

import pandas as pd

from mlb_model.config import settings
from mlb_model.data.warehouse import upsert_dataframe
from mlb_model.logging import get_logger

log = get_logger("data.sources.batter_statcast")

# Per-day throttle marker (re-pull at most twice a day).
_MARKER_DIR = settings.cache_dir / "batter_statcast"
_THROTTLE_HOURS = 12


def _marker_path(season: int) -> Path:
    return _MARKER_DIR / f"last_pull_{season}.txt"


def _is_throttled(season: int) -> bool:
    path = _marker_path(season)
    if not path.exists():
        return False
    try:
        last = datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    age_hours = (datetime.now() - last).total_seconds() / 3600.0
    return age_hours < _THROTTLE_HOURS


def _mark_done(season: int) -> None:
    _MARKER_DIR.mkdir(parents=True, exist_ok=True)
    _marker_path(season).write_text(datetime.now().isoformat(timespec="seconds"), encoding="utf-8")


def _coerce(v) -> float | None:
    """Best-effort numeric cast; Savant returns strings + ``--`` sometimes."""
    if v is None:
        return None
    try:
        if isinstance(v, str):
            v = v.strip()
            if not v or v in ("--", "—", "N/A"):
                return None
        x = float(v)
        if x != x:  # NaN
            return None
        return x
    except (TypeError, ValueError):
        return None


def ingest_season(season: int, *, min_pa: int = 5) -> int:
    """Pull + upsert ``batter_statcast_season`` rows for ``season``.

    ``min_pa`` is passed to the expected-stats endpoint; we keep it low so
    early-season batters with few PA still get rows. The exitvelo merge is
    left-joined so a batter with PA but no batted-ball events still lands
    a row (xBA / xSLG without Statcast quality).

    Returns the number of rows written.
    """
    import pybaseball

    try:
        exp = pybaseball.statcast_batter_expected_stats(season, minPA=min_pa)
    except Exception:  # noqa: BLE001 -- network blip shouldn't crash app
        log.exception("batter_statcast.expected.failed", season=season)
        return 0
    try:
        bb = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=5)
    except Exception:  # noqa: BLE001 -- exitvelo is optional
        log.exception("batter_statcast.exitvelo.failed", season=season)
        bb = pd.DataFrame(columns=["player_id"])

    if exp is None or exp.empty:
        log.warning("batter_statcast.expected.empty", season=season)
        return 0

    # Normalize: bring the two frames to a tidy schema, then merge.
    exp = exp.rename(columns={
        "player_id": "player_id",
        "pa": "pa", "bip": "bip",
        "est_ba": "xba", "est_slg": "xslg", "est_woba": "xwoba",
    })[["player_id", "pa", "bip", "xba", "xslg", "xwoba"]]

    if not bb.empty:
        bb = bb.rename(columns={
            "player_id": "player_id",
            "avg_hit_speed": "ev_avg",
            "ev95percent":   "hardhit_pct_raw",
            "brl_percent":   "barrel_pct_raw",
        })[["player_id", "ev_avg", "hardhit_pct_raw", "barrel_pct_raw"]]
    else:
        bb = pd.DataFrame(columns=["player_id", "ev_avg", "hardhit_pct_raw", "barrel_pct_raw"])

    merged = exp.merge(bb, on="player_id", how="left")
    # Savant reports Barrel% / HardHit% as percent-of-BBE (e.g. ``12.4``);
    # convert to 0..1 fractions to match the scoring module's conventions.
    merged["barrel_pct"]  = merged["barrel_pct_raw"].apply(_coerce).map(
        lambda x: x / 100.0 if x is not None else None
    )
    merged["hardhit_pct"] = merged["hardhit_pct_raw"].apply(_coerce).map(
        lambda x: x / 100.0 if x is not None else None
    )
    # Cast the numerics safely.
    for col in ("pa", "bip"):
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype("Int64")
    for col in ("xba", "xslg", "xwoba", "ev_avg"):
        merged[col] = merged[col].apply(_coerce)

    merged["season"] = int(season)
    merged["last_updated"] = pd.Timestamp.utcnow()
    out = merged[[
        "player_id", "season", "pa", "bip",
        "xba", "xslg", "xwoba",
        "barrel_pct", "hardhit_pct", "ev_avg",
        "last_updated",
    ]].dropna(subset=["player_id"])
    out["player_id"] = out["player_id"].astype(int)

    if out.empty:
        return 0
    n = upsert_dataframe(out, "batter_statcast_season",
                         key_columns=["player_id", "season"])
    _mark_done(season)
    log.info("batter_statcast.ingested", season=season, rows=n)
    return n


def maybe_refresh(season: int | None = None) -> int:
    """Refresh ``season`` if the 12-hour throttle has elapsed.

    Designed to be called from request paths (Matchups page load,
    morning-sync, etc.) without worrying about over-pulling Savant.
    """
    season = season or date_cls.today().year
    if _is_throttled(season):
        return 0
    return ingest_season(season)
