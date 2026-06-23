# Packaging

Build a self-contained, Apple Silicon-only `MLB Forecast.app` and wrap it into a DMG.

## One-shot build

```bash
# 0. Ensure lightgbm can load OpenMP (arm64 libomp). REQUIRED on a fresh
#    venv — the PyPI wheel doesn't bundle libomp, and without it both the
#    dev app and the packaged .app crash with "Library not loaded:
#    @rpath/libomp.dylib". Idempotent; see scripts/fix_libomp.sh.
bash scripts/fix_libomp.sh

# 1. PyInstaller produces dist/MLB Forecast.app/  (~740 MB)
uv run pyinstaller --noconfirm packaging/mlb_forecast.spec

# 2. hdiutil produces MLB-Forecast-arm64.dmg at the project root (~234 MB)
bash packaging/make_dmg.sh
```

> The libomp fix lives in `.venv` (gitignored), so re-run step 0 any time you
> recreate the virtualenv. The PyInstaller spec collects the resulting
> `libomp.dylib` automatically, so it ends up in the bundle.

## What's in the bundle

| Path inside the .app | Source | Purpose |
|---|---|---|
| `Contents/MacOS/MLB Forecast` | PyInstaller bootloader | Entry point |
| `Contents/Resources/data/warehouse.duckdb` | `data/warehouse.duckdb` | Initial 8-season warehouse, ~187 MB |
| `Contents/Resources/models/` | `models/` | Trained joblibs + calibrators |
| `Contents/Resources/mlb_model/app/templates/` | source | Jinja2 templates |
| `Contents/Resources/mlb_model/app/static/` | source | CSS / JS |

## First-launch behavior

The bundled app copies the read-only warehouse + models out of the .app into a
writable location at `~/Library/Application Support/MLB Forecast/` so the user
can retrain in place and ingest new data. The `.app` itself is never modified.

Settings paths are overridden via `MLB_*` env vars set by
`packaging/mlb_forecast_main.py` BEFORE `mlb_model.config.settings` instantiates.

## Activating live odds on a fresh install

The Odds API key isn't bundled (it's per-user). After the user installs the
.app:

```bash
echo "MLB_ODDS_API_KEY=their_key_here" > "$HOME/Library/Application Support/MLB Forecast/.env"
```

Then reopen the app. The `/api/refresh` endpoint will start pulling live lines.

## What's NOT in the bundle (intentionally)

- `mlb-model` CLI — only the desktop GUI is shipped. Power users who want
  CLI access need the dev install (`uv sync --all-extras`).
- launchd auto-Sunday retrain — replaced by the **Retrain & analyze** button
  on the `/season` page.
- Test suite + dev tooling (pytest, ruff, mypy, jupyterlab).

## Rebuilding from scratch

If you've modified the source, just rerun step 1 and 2 above. PyInstaller
caches in `build/`; pass `--clean` to force a fresh build.

```bash
uv run pyinstaller --clean --noconfirm packaging/mlb_forecast.spec
bash packaging/make_dmg.sh
```

## Build host requirements

- Apple Silicon Mac (arm64). The bundle is single-arch — building on Intel /
  Rosetta produces a broken x86_64 binary that PyInstaller refuses to convert.
  If your venv is x86_64 (e.g. Intel Homebrew Python), do:
  ```bash
  uv python install 3.13
  rm -rf .venv
  uv sync --all-extras --python 3.13
  ```
- ~3 GB free disk during the build (intermediate files in `build/`).
