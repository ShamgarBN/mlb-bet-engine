# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the standalone MLB Forecast.app.

Build with:
    uv run pyinstaller --noconfirm packaging/mlb_forecast.spec

This produces a directory bundle at ``dist/MLB Forecast.app``. Wrap it
into a DMG via ``packaging/make_dmg.sh``.
"""
from pathlib import Path
from PyInstaller.utils.hooks import collect_all, collect_submodules


PROJECT_ROOT = Path(SPECPATH).resolve().parent

# --- Data files baked into the bundle ---------------------------------------
# Each tuple is (source, dest-inside-bundle). The dest path is relative
# to the bundle's resources root (a.k.a. ``sys._MEIPASS`` at runtime).
datas = [
    (str(PROJECT_ROOT / "src" / "mlb_model" / "app" / "templates"), "mlb_model/app/templates"),
    (str(PROJECT_ROOT / "src" / "mlb_model" / "app" / "static"), "mlb_model/app/static"),
    (str(PROJECT_ROOT / "models"), "models"),
    (str(PROJECT_ROOT / "data" / "warehouse.duckdb"), "data"),
]
# Include the Arnav odds JSON if it's present so the user can re-ingest
# 2021-2025 odds after retraining if needed. Optional; build still works
# without it.
arnav_path = PROJECT_ROOT / "data" / "raw" / "odds_scraped" / "mlb_odds_dataset.json"
if arnav_path.exists():
    datas.append((str(arnav_path), "data/raw/odds_scraped"))

# Walk-forward backtest results power the /performance page. Bundle the
# canonical CSV so the page has data on first launch instead of an empty
# "No backtest CSVs found" state.
backtest_csv = PROJECT_ROOT / "logs" / "backtest_v4.csv"
if backtest_csv.exists():
    datas.append((str(backtest_csv), "logs"))
backtest_2026 = PROJECT_ROOT / "logs" / "backtest_2026.csv"
if backtest_2026.exists():
    datas.append((str(backtest_2026), "logs"))

# --- Hidden imports ---------------------------------------------------------
# PyInstaller's static analyzer misses several things:
#   1. anything imported via a string in uvicorn / FastAPI / pywebview
#   2. lightgbm + scikit-learn pure-Python helpers loaded by C extensions
#   3. duckdb's plugin-style submodules
hiddenimports = []
hiddenimports += collect_submodules("mlb_model")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fastapi")
hiddenimports += collect_submodules("starlette")
hiddenimports += collect_submodules("webview")
hiddenimports += collect_submodules("sklearn")
hiddenimports += collect_submodules("lightgbm")
hiddenimports += collect_submodules("duckdb")
hiddenimports += collect_submodules("pandas")
hiddenimports += collect_submodules("numpy")
hiddenimports += collect_submodules("scipy")
hiddenimports += [
    # FastAPI form parsing uses these via string-based discovery.
    "multipart",
    "email_validator",
    # Jinja2 autoescape backend.
    "jinja2.ext",
    # pywebview macOS backend.
    "webview.platforms.cocoa",
    # uvicorn loop backends.
    "uvicorn.loops.asyncio",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "h11",
]

# --- Binary collection for libs that ship .dylib / .so files ----------------
# ``collect_all`` returns (datas, binaries, hiddenimports) for a package.
binaries = []
for pkg in ("lightgbm", "duckdb", "sklearn", "numpy", "scipy", "pandas"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

# --- Tell PyInstaller about the Python entry point --------------------------
a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "mlb_forecast_main.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Dev-only tooling -- keeps the bundle smaller.
        "pytest", "_pytest", "hypothesis", "ruff", "mypy",
        "jupyterlab", "ipykernel", "ipywidgets", "notebook",
        "PIL.ImageQt", "PyQt5", "PyQt6", "PySide2", "PySide6",
        "tkinter",
    ],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MLB Forecast",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # no terminal -- macOS app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="arm64",     # Apple Silicon only (user choice)
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "packaging" / "MLBForecast.icns"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="MLB Forecast",
)

app = BUNDLE(
    coll,
    name="MLB Forecast.app",
    icon=str(PROJECT_ROOT / "packaging" / "MLBForecast.icns"),
    bundle_identifier="com.mlbforecast.app",
    version="1.1.2",
    info_plist={
        "CFBundleDisplayName": "MLB Forecast",
        "CFBundleName": "MLB Forecast",
        "CFBundleShortVersionString": "1.1.2",
        "CFBundleVersion": "1.1.2",
        # Keep the app out of the Dock's "Recent" list spam; user can
        # still cmd-tab to it. The window is opened by pywebview.
        "LSBackgroundOnly": False,
        "NSHighResolutionCapable": True,
        # MLB Forecast doesn't access camera / mic / location / contacts;
        # we list nothing here on purpose so macOS doesn't prompt.
    },
)
