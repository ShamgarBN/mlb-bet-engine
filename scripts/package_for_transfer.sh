#!/usr/bin/env bash
# Package the project into a single ``.tar.gz`` you can copy to another
# Mac (AirDrop, iCloud Drive, external SSD, scp, whatever).
#
# What goes in:
#   - all source code, pyproject.toml, uv.lock, README, tests
#   - data/warehouse.duckdb        (the DuckDB store of every game/odds/pitch we've pulled)
#   - data/raw/                    (raw upstream caches -- statcast etc.)
#   - data/cache/predictions/      (per-day prediction parquets)
#   - data/cache/picks_log.parquet (your tracked picks, if any)
#   - data/cache/last_*_sync.txt   (so the destination knows when jobs last ran)
#   - data/cache/odds/             (manual odds artefacts)
#   - models/                      (trained ML artefacts + archive)
#   - MLB Forecast.app/            (the launcher bundle, re-signed on dest)
#   - .git/                        (your commit history)
#
# What's left out (auto-regenerated on the destination):
#   - .venv/                       (~600 MB, platform-specific wheels)
#   - logs/                        (rebuilt on first launch)
#   - __pycache__/                 (bytecode)
#   - data/cache/http_cache.sqlite (~3.8 GB; cached MLB API responses --
#                                   nice to have but rebuilds itself)
#
# Use ``--full`` to also include the HTTP cache. That makes the archive
# ~4 GB but means the destination Mac never has to re-call any MLB
# Stats API endpoint we've already seen.

set -euo pipefail

ROOT="$( cd "$( dirname "$0" )/.." && pwd )"
PROJECT_NAME="$( basename "${ROOT}" )"

INCLUDE_HTTP_CACHE=0
OUTPUT_DIR="${HOME}/Desktop"
for arg in "$@"; do
  case "${arg}" in
    --full)        INCLUDE_HTTP_CACHE=1 ;;
    --lean)        INCLUDE_HTTP_CACHE=0 ;;
    --output=*)    OUTPUT_DIR="${arg#--output=}" ;;
    -h|--help)
      cat <<EOF
Usage: $(basename "$0") [--full|--lean] [--output=DIR]

  --lean (default)  Skip data/cache/http_cache.sqlite (~3.8 GB). Resulting
                    tarball is ~80-100 MB. The destination Mac re-caches
                    HTTP responses on demand; first morning-sync runs a
                    little slower, after that it's identical.

  --full            Include the HTTP cache. Tarball balloons to ~1 GB
                    compressed / ~4 GB uncompressed but the destination
                    needs zero re-fetching.

  --output=DIR      Where to write the tarball. Default: ~/Desktop
EOF
      exit 0 ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      exit 1 ;;
  esac
done

mkdir -p "${OUTPUT_DIR}"

# Refuse to package while the app is running -- DuckDB would copy in an
# inconsistent state and the new Mac would see corruption.
if pgrep -f 'mlb-model app' >/dev/null 2>&1 || pgrep -f 'mlb-model serve' >/dev/null 2>&1; then
  cat >&2 <<EOF
The MLB Forecast app appears to be running.

Close it (Cmd-Q the window or kill the process) before packaging, otherwise
the DuckDB warehouse will be copied mid-transaction and the destination Mac
won't be able to open it.

  pkill -f 'mlb-model'

then re-run this script.
EOF
  exit 1
fi

TIMESTAMP="$( date +%Y%m%d_%H%M%S )"
ARCHIVE="${OUTPUT_DIR}/${PROJECT_NAME}-transfer-${TIMESTAMP}.tar.gz"

# tar reads exclude patterns relative to the archive root. Each pattern
# is a path *inside* the project dir.
EXCLUDES=(
  "--exclude=${PROJECT_NAME}/.venv"
  "--exclude=${PROJECT_NAME}/logs"
  "--exclude=${PROJECT_NAME}/__pycache__"
  "--exclude=*.pyc"
  "--exclude=.DS_Store"
  "--exclude=${PROJECT_NAME}/.mypy_cache"
  "--exclude=${PROJECT_NAME}/.pytest_cache"
  "--exclude=${PROJECT_NAME}/.ruff_cache"
  "--exclude=${PROJECT_NAME}/.ipynb_checkpoints"
)
if [[ "${INCLUDE_HTTP_CACHE}" -eq 0 ]]; then
  EXCLUDES+=( "--exclude=${PROJECT_NAME}/data/cache/http_cache.sqlite" )
fi

# Verbosity: pipe to wc -l so the user sees a progress hint without 50k lines of file names.
echo "Packaging from: ${ROOT}"
echo "Output archive: ${ARCHIVE}"
echo "HTTP cache included: $([[ "${INCLUDE_HTTP_CACHE}" -eq 1 ]] && echo yes || echo "no (default)")"
echo ""

cd "$( dirname "${ROOT}" )"
tar -czf "${ARCHIVE}" \
    "${EXCLUDES[@]}" \
    "${PROJECT_NAME}"

SIZE="$( du -h "${ARCHIVE}" | awk '{print $1}' )"
SHA="$( shasum -a 256 "${ARCHIVE}" | awk '{print $1}' )"

cat <<EOF

Built ${ARCHIVE}
Size: ${SIZE}
SHA-256: ${SHA}

Next steps (on the destination Mac):

  1. Install uv if it's not already there:
        curl -LsSf https://astral.sh/uv/install.sh | sh

  2. Copy the tarball over (AirDrop / iCloud Drive / scp / external SSD).
     Put it somewhere stable -- NOT in ~/Desktop, ~/Documents, or
     ~/Downloads if you can avoid it (TCC adds friction on the .app).
     Recommended landing spot: ~/Projects/ or ~/Applications/.

  3. Extract it:
        cd ~/Projects
        tar -xzf ~/Downloads/${PROJECT_NAME}-transfer-${TIMESTAMP}.tar.gz

  4. Strip any Gatekeeper quarantine flags the transfer added:
        xattr -dr com.apple.quarantine ~/Projects/${PROJECT_NAME}

  5. Re-sign the .app on this Mac (ad-hoc, no Apple Developer account
     needed). The original signature was made on the source machine;
     re-signing locally avoids LaunchServices weirdness:
        cd ~/Projects/${PROJECT_NAME}
        ./scripts/build_app_bundle.sh

  6. Double-click "MLB Forecast.app". First launch takes 30-60 s while
     uv rebuilds the .venv for this CPU architecture. After that it's
     instant.

Done. Both Macs now have identical models, warehouse, and picks log.
EOF
