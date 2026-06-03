"""PyInstaller entry point for the standalone MLB Forecast.app.

This is the single Python script PyInstaller bundles into a macOS .app.
It boots the local FastAPI server in a daemon thread and opens a native
WebKit window pointed at it -- the same machinery as ``mlb-model app``,
but called directly without going through the Typer CLI (so the bundled
binary has nothing to do with command-line argv parsing).

Project-relative paths (templates, static, models, warehouse) work
because we put the whole ``src/mlb_model`` tree + a snapshot of
``data/warehouse.duckdb`` + ``models/`` + ``data/raw/`` into the .app's
Resources via the .spec file. ``Settings.project_root`` resolves to a
writable location under ``~/Library/Application Support/MLB Forecast/``
at first launch; see :func:`_prepare_app_support_root` below.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME = "MLB Forecast"
SUPPORT_SUBDIR = "MLB Forecast"


def _bundle_root() -> Path:
    """Path to the read-only resources baked into the .app.

    When frozen by PyInstaller, ``sys._MEIPASS`` is the directory where
    the bundle extracted its data files. When running unfrozen (i.e.
    ``python packaging/mlb_forecast_main.py``), fall back to the project
    root so developers can sanity-check this script.
    """
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent


def _app_support_root() -> Path:
    """User-writable home for the runtime warehouse + models + logs."""
    base = Path.home() / "Library" / "Application Support" / SUPPORT_SUBDIR
    base.mkdir(parents=True, exist_ok=True)
    return base


def _prepare_app_support_root() -> Path:
    """On first launch copy the bundled warehouse + models into the
    user's Application Support folder so subsequent runs can write to
    them. Idempotent: skips already-installed files.
    """
    bundle = _bundle_root()
    support = _app_support_root()

    # ``warehouse.duckdb`` is the largest payload (~hundreds of MB). We
    # only copy it on first launch.
    src_warehouse = bundle / "data" / "warehouse.duckdb"
    dst_warehouse = support / "data" / "warehouse.duckdb"
    if src_warehouse.exists() and not dst_warehouse.exists():
        dst_warehouse.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_warehouse, dst_warehouse)

    # Trained models -- small (~1 MB), but the user can retrain in place
    # so we mirror them at first launch and then leave them alone.
    src_models = bundle / "models"
    dst_models = support / "models"
    if src_models.exists() and not dst_models.exists():
        shutil.copytree(src_models, dst_models)

    # Pre-built raw odds dataset (for re-ingest on retrain). Optional;
    # the app works without it.
    src_raw = bundle / "data" / "raw"
    dst_raw = support / "data" / "raw"
    if src_raw.exists() and not dst_raw.exists():
        shutil.copytree(src_raw, dst_raw)

    # Backtest CSVs + backtest parquet power /performance and the
    # props-backfill command. Copy each file individually so we never
    # clobber the live mlb_model.log the running app writes to.
    src_logs = bundle / "logs"
    dst_logs = support / "logs"
    if src_logs.exists():
        dst_logs.mkdir(parents=True, exist_ok=True)
        for src in src_logs.glob("*.csv"):
            target = dst_logs / src.name
            if not target.exists():
                shutil.copy2(src, target)
        for src in src_logs.glob("*.parquet"):
            target = dst_logs / src.name
            if not target.exists():
                shutil.copy2(src, target)

    # Pre-populated hitter-prop journal so /season shows tier success
    # rates from day one (~591k graded rows over 5 seasons). Only seed
    # when the support root doesn't already have one -- never overwrite
    # the user's accumulated history.
    src_journal = bundle / "data" / "journal" / "prop_predictions.parquet"
    dst_journal = support / "data" / "journal" / "prop_predictions.parquet"
    if src_journal.exists() and not dst_journal.exists():
        dst_journal.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_journal, dst_journal)

    return support


def _load_env_file(support_root: Path) -> None:
    """Load ``<support_root>/.env`` into ``os.environ`` BEFORE mlb_model
    imports.

    Why this is necessary: pydantic-settings binds its ``env_file`` to a
    path computed from ``config.py``'s module location at import time. In
    a PyInstaller bundle that resolves to a frozen path inside the .app,
    NOT the user-writable support root, so a key the user saved via the
    Settings page would never be re-read on the next launch. We sidestep
    that entirely by parsing the .env ourselves and seeding os.environ,
    which both pydantic and ``odds_api._api_key()`` honor.
    """
    env_path = support_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            # Don't clobber a variable explicitly exported in the shell.
            os.environ.setdefault(key, value)
    except OSError:
        pass


def _wire_settings(support_root: Path) -> None:
    """Point :data:`mlb_model.config.settings` at the writable support
    root. The package reads ``MLB_*`` env vars before its Settings class
    instantiates, so we set them here BEFORE importing anything from
    ``mlb_model``.
    """
    os.environ.setdefault("MLB_PROJECT_ROOT", str(support_root))
    os.environ.setdefault("MLB_DATA_DIR", str(support_root / "data"))
    os.environ.setdefault("MLB_RAW_DIR", str(support_root / "data" / "raw"))
    os.environ.setdefault("MLB_INTERIM_DIR", str(support_root / "data" / "interim"))
    os.environ.setdefault("MLB_PROCESSED_DIR", str(support_root / "data" / "processed"))
    os.environ.setdefault("MLB_CACHE_DIR", str(support_root / "data" / "cache"))
    os.environ.setdefault("MLB_WAREHOUSE_PATH", str(support_root / "data" / "warehouse.duckdb"))
    os.environ.setdefault("MLB_MODEL_DIR", str(support_root / "models"))
    os.environ.setdefault("MLB_LOGS_DIR", str(support_root / "logs"))


def _install_hard_exit_atexit() -> None:
    """Bypass __cxa_finalize on shutdown -- catches Cmd-Q too.

    The v1.1.2 fix put ``os._exit(0)`` in ``main()``'s ``finally:`` block,
    which works for the window-close path because pywebview's
    ``webview.start()`` returns to Python on close. But ``Cmd-Q`` dispatches
    ``-[NSApplication terminate:]`` from AppKit's event loop -- AppKit
    calls libc ``exit()`` directly, never returning to Python, so the
    ``finally:`` never runs.

    ``exit()`` runs registered C atexit handlers LIFO. Python's interpreter
    cleanup is one of those (Py_FinalizeEx), and Py_FinalizeEx in turn
    runs Python-module ``atexit`` handlers. So a handler registered here
    fires BEFORE libc's exit continues into ``__cxa_finalize``, which is
    where DuckDB's C++ destructor crashes when calling back into the
    half-torn-down Python interpreter.

    ``os._exit(0)`` is a direct ``_exit(2)`` syscall -- it bypasses all
    remaining cleanup. Safe here: every write in the app (warehouse,
    journal, picks log, predictions cache) is synchronous-to-disk before
    its handler returns, and loguru flushes per write.
    """
    import atexit
    atexit.register(lambda: os._exit(0))


def main() -> None:
    _install_hard_exit_atexit()
    support_root = _prepare_app_support_root()
    _load_env_file(support_root)
    _wire_settings(support_root)

    # Import after env vars are set so Settings picks them up.
    from mlb_model.app.desktop import launch_native_window
    from mlb_model.logging import configure_logging

    configure_logging()
    try:
        launch_native_window()
    finally:
        # Shutdown-order crash workaround: when AppKit's terminate: fires
        # exit() at quit, __cxa_finalize_ranges runs C++ static destructors.
        # DuckDB's destructor tries to release Python references and call
        # PyEval_SaveThread, but Python's interpreter is already gone --
        # NULL-deref at 0xb0, SIGSEGV, crash report every time. By calling
        # os._exit() we skip the broken cleanup pass entirely. Safe here
        # because: the journal + picks-log + DuckDB writes all happen
        # synchronously per-request, loguru flushes per write, and the
        # uvicorn server has already been told to stop by the pywebview
        # `closed` callback. The OS reclaims memory the same as a clean
        # exit would.
        os._exit(0)


if __name__ == "__main__":
    main()
