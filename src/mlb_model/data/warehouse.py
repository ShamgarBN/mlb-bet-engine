"""DuckDB-backed local warehouse.

One file (``data/warehouse.duckdb``) holds every cleaned fact the model
needs: schedule, boxscores, lineups, pitcher / batter game stats, weather,
umpires, Statcast daily summaries, venues, and historical odds.

The schema is created idempotently on first call to :func:`init_schema`.
Tables are intentionally wide and de-normalized — the workload is
read-heavy analytical SQL, not OLTP.

Concurrency: DuckDB supports a single writer; multiple readers are fine
via separate connections. We open a fresh connection per call so the
FastAPI worker, the CLI, and notebooks can coexist without holding a
long-lived handle.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator

import duckdb
import pandas as pd

from mlb_model.config import settings
from mlb_model.logging import get_logger

log = get_logger("data.warehouse")

_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY = False


# --------------------------------------------------------------------------- #
# Schema                                                                      #
# --------------------------------------------------------------------------- #
#
# Tables are listed in dependency order (parents before children). Column
# choices are driven by the SQL queries in features/ and predict/.

_DDL: list[str] = [
    # venues: static metadata seeded from venues_seed.py.
    """
    CREATE TABLE IF NOT EXISTS venues (
        team_abbr        VARCHAR PRIMARY KEY,
        name             VARCHAR,
        venue_id         INTEGER,
        latitude         DOUBLE,
        longitude        DOUBLE,
        altitude_ft      INTEGER,
        is_dome          BOOLEAN DEFAULT FALSE,
        roof_type        VARCHAR,
        cf_bearing_deg   DOUBLE,
        cf_distance_ft   INTEGER
    )
    """,
    # games: one row per scheduled MLB game.
    """
    CREATE TABLE IF NOT EXISTS games (
        game_pk          BIGINT PRIMARY KEY,
        game_date        DATE NOT NULL,
        season           INTEGER NOT NULL,
        scheduled_start  TIMESTAMP,
        home_team_id     INTEGER,
        away_team_id     INTEGER,
        home_team_abbr   VARCHAR,
        away_team_abbr   VARCHAR,
        venue_id         INTEGER,
        venue_name       VARCHAR,
        status           VARCHAR,
        home_score       INTEGER,
        away_score       INTEGER,
        home_win         BOOLEAN,
        doubleheader     VARCHAR,
        game_number      INTEGER
    )
    """,
    # team_boxscores: two rows per finalized game (home + away).
    """
    CREATE TABLE IF NOT EXISTS team_boxscores (
        game_pk          BIGINT NOT NULL,
        team_id          INTEGER NOT NULL,
        is_home          BOOLEAN,
        runs             INTEGER,
        hits             INTEGER,
        home_runs        INTEGER,
        doubles          INTEGER,
        triples          INTEGER,
        walks            INTEGER,
        strikeouts       INTEGER,
        at_bats          INTEGER,
        plate_appearances INTEGER,
        total_bases      INTEGER,
        left_on_base     INTEGER,
        PRIMARY KEY (game_pk, team_id)
    )
    """,
    # probable_pitchers: one row per game per side, populated from schedule.
    """
    CREATE TABLE IF NOT EXISTS probable_pitchers (
        game_pk          BIGINT NOT NULL,
        team_id          INTEGER,
        pitcher_id       INTEGER NOT NULL,
        pitcher_name     VARCHAR,
        is_home          BOOLEAN,
        pitcher_throws   VARCHAR,
        PRIMARY KEY (game_pk, pitcher_id)
    )
    """,
    # lineups: 9-row block per team per finalized game.
    """
    CREATE TABLE IF NOT EXISTS lineups (
        game_pk          BIGINT NOT NULL,
        team_id          INTEGER NOT NULL,
        batting_order    INTEGER NOT NULL,
        player_id        INTEGER,
        position         VARCHAR,
        bats             VARCHAR,
        PRIMARY KEY (game_pk, team_id, batting_order)
    )
    """,
    # pitcher_game_stats: one row per pitcher per appearance.
    """
    CREATE TABLE IF NOT EXISTS pitcher_game_stats (
        game_pk          BIGINT NOT NULL,
        pitcher_id       INTEGER NOT NULL,
        team_id          INTEGER,
        is_starter       BOOLEAN,
        innings_pitched  DOUBLE,
        batters_faced    INTEGER,
        hits             INTEGER,
        runs             INTEGER,
        earned_runs      INTEGER,
        strikeouts       INTEGER,
        walks            INTEGER,
        home_runs        INTEGER,
        pitches_thrown   INTEGER,
        strikes_thrown   INTEGER,
        PRIMARY KEY (game_pk, pitcher_id)
    )
    """,
    # batter_game_stats: one row per batter per game.
    """
    CREATE TABLE IF NOT EXISTS batter_game_stats (
        game_pk          BIGINT NOT NULL,
        batter_id        INTEGER NOT NULL,
        team_id          INTEGER,
        at_bats          INTEGER,
        plate_appearances INTEGER,
        hits             INTEGER,
        doubles          INTEGER,
        triples          INTEGER,
        home_runs        INTEGER,
        walks            INTEGER,
        strikeouts       INTEGER,
        hbp              INTEGER,
        total_bases      INTEGER,
        PRIMARY KEY (game_pk, batter_id)
    )
    """,
    # weather: one row per game (dome games get NaN sensors + is_dome=TRUE).
    """
    CREATE TABLE IF NOT EXISTS weather (
        game_pk          BIGINT PRIMARY KEY,
        temp_f           DOUBLE,
        humidity_pct     DOUBLE,
        pressure_hpa     DOUBLE,
        wind_speed_mph   DOUBLE,
        wind_out_to_cf   DOUBLE,
        is_dome          BOOLEAN DEFAULT FALSE
    )
    """,
    # umpires: plate ump for each game.
    """
    CREATE TABLE IF NOT EXISTS umpires (
        game_pk          BIGINT PRIMARY KEY,
        ump_name         VARCHAR
    )
    """,
    # statcast_pitcher_daily: per-game Statcast aggregate (2015+).
    """
    CREATE TABLE IF NOT EXISTS statcast_pitcher_daily (
        game_pk          BIGINT NOT NULL,
        pitcher_id       INTEGER NOT NULL,
        pitches_thrown   INTEGER,
        swinging_strikes INTEGER,
        called_strikes   INTEGER,
        in_zone_pct      DOUBLE,
        chase_pct        DOUBLE,
        avg_velocity     DOUBLE,
        spin_rate_avg    DOUBLE,
        xwoba_against    DOUBLE,
        woba_against     DOUBLE,
        barrels_against  INTEGER,
        hard_hits_against INTEGER,
        PRIMARY KEY (game_pk, pitcher_id)
    )
    """,
    # odds_history: one row per (date, matchup, book).
    """
    CREATE TABLE IF NOT EXISTS odds_history (
        game_date            DATE NOT NULL,
        home_team_abbr       VARCHAR NOT NULL,
        away_team_abbr       VARCHAR NOT NULL,
        book                 VARCHAR NOT NULL,
        ml_open_home         INTEGER,
        ml_open_away         INTEGER,
        ml_close_home        INTEGER,
        ml_close_away        INTEGER,
        rl_close_home        DOUBLE,
        rl_close_home_price  INTEGER,
        rl_close_away_price  INTEGER,
        total_close          DOUBLE,
        total_close_over     INTEGER,
        total_close_under    INTEGER,
        PRIMARY KEY (game_date, home_team_abbr, away_team_abbr, book)
    )
    """,
]

_INDEXES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_games_date ON games(game_date)",
    "CREATE INDEX IF NOT EXISTS idx_games_season ON games(season)",
    "CREATE INDEX IF NOT EXISTS idx_games_status ON games(status)",
    "CREATE INDEX IF NOT EXISTS idx_pgs_pitcher ON pitcher_game_stats(pitcher_id)",
    "CREATE INDEX IF NOT EXISTS idx_bgs_batter ON batter_game_stats(batter_id)",
    "CREATE INDEX IF NOT EXISTS idx_tbs_team ON team_boxscores(team_id)",
    "CREATE INDEX IF NOT EXISTS idx_odds_date ON odds_history(game_date)",
]


def _warehouse_path() -> Path:
    """Resolve the on-disk path; honors settings overrides at test time."""
    path = Path(settings.warehouse_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open a short-lived DuckDB connection. Always closed via context manager."""
    path = _warehouse_path()
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def init_schema(force: bool = False) -> None:
    """Create every table + index. Idempotent; cheap to call repeatedly.

    Errors during DDL are logged but not re-raised — a missing warehouse
    must never break app boot.
    """
    global _SCHEMA_READY
    if _SCHEMA_READY and not force:
        return
    with _SCHEMA_LOCK:
        if _SCHEMA_READY and not force:
            return
        try:
            with _connect() as con:
                for ddl in _DDL:
                    con.execute(ddl)
                for idx in _INDEXES:
                    try:
                        con.execute(idx)
                    except Exception:  # noqa: BLE001 -- indexes are best-effort
                        log.warning("warehouse.index.failed", sql=idx)
            _SCHEMA_READY = True
            log.info("warehouse.schema.ready", path=str(_warehouse_path()))
        except Exception:  # noqa: BLE001 -- never crash boot
            log.exception("warehouse.schema.failed")


def query(sql: str, params: Iterable | None = None) -> pd.DataFrame:
    """Run a parameterized SELECT and return the result as a DataFrame.

    Returns an empty DataFrame on any error so callers can branch on
    ``df.empty`` rather than wrapping every call in try/except.
    """
    init_schema()
    try:
        with _connect(read_only=False) as con:
            if params is None:
                rel = con.execute(sql)
            else:
                # DuckDB accepts list/tuple for positional params
                rel = con.execute(sql, list(params))
            return rel.fetch_df()
    except Exception:  # noqa: BLE001 -- caller branches on .empty
        log.exception("warehouse.query.failed", sql=sql[:200])
        return pd.DataFrame()


def upsert_dataframe(
    df: pd.DataFrame, table: str, key_columns: list[str]
) -> int:
    """Insert-or-update every row from ``df`` into ``table``.

    Implementation: DuckDB doesn't have a portable ON CONFLICT for every
    case, so we use the documented "INSERT INTO … ON CONFLICT (keys) DO
    UPDATE SET col=excluded.col" pattern. Existing rows are updated, new
    rows are inserted. Returns the number of rows attempted.
    """
    if df is None or df.empty:
        return 0
    init_schema()

    cols = list(df.columns)
    if not cols:
        return 0
    col_list = ", ".join(cols)
    placeholders = ", ".join(["?"] * len(cols))
    # Build the update clause: every non-key column gets bumped to the
    # excluded (new) value.
    non_key_cols = [c for c in cols if c not in key_columns]
    if non_key_cols:
        update_clause = ", ".join(f"{c}=excluded.{c}" for c in non_key_cols)
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT ({', '.join(key_columns)}) DO UPDATE SET {update_clause}"
        )
    else:
        # All columns are keys -- do nothing on conflict.
        sql = (
            f"INSERT INTO {table} ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT DO NOTHING"
        )

    # Convert NaN→None so DuckDB sees SQL NULL rather than the float NaN.
    # ``df.where(df.notna(), None)`` keeps numeric dtypes, so NaNs survive
    # as floats and DuckDB can't cast them to INT. Cast to object first so
    # every cell is a Python value and ``None`` is preserved as SQL NULL.
    clean = df.astype(object).where(df.notna(), None)
    rows = list(clean.itertuples(index=False, name=None))
    n = 0
    try:
        with _connect() as con:
            con.executemany(sql, rows)
            n = len(df)
    except Exception:
        log.exception("warehouse.upsert.failed", table=table, rows=len(df))
        return 0
    return n
