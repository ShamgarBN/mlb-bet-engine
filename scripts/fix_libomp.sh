#!/usr/bin/env bash
# Make lightgbm loadable in the project venv on Apple Silicon.
#
# WHY: lightgbm's `lib_lightgbm.dylib` (arm64) depends on `@rpath/libomp.dylib`
# (the OpenMP runtime), which is NOT bundled in the PyPI wheel. lightgbm only
# searches Apple-Silicon Homebrew paths (`/opt/homebrew/opt/libomp/lib`) and
# MacPorts. If you only have x86_64 libomp (Intel Homebrew at /usr/local) — or
# no libomp at all — lightgbm fails to load with:
#   Library not loaded: @rpath/libomp.dylib
# and the prediction pages (Today / Matchups / Backtest) error out.
#
# This script fetches an arm64 `libomp.dylib`, drops it next to lightgbm inside
# the venv, points lightgbm's rpath at it, and re-signs — no system changes, no
# second Homebrew. Run it after creating/recreating the venv and before
# building the .app:
#
#   uv sync
#   bash scripts/fix_libomp.sh
#   uv run pyinstaller --noconfirm packaging/mlb_forecast.spec
#
# Idempotent: safe to run repeatedly.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${PROJECT_ROOT}/.venv"

LGDIR="$(find "${VENV}/lib" -type d -path "*/lightgbm/lib" 2>/dev/null | head -1)"
if [[ -z "${LGDIR}" ]]; then
  echo "error: lightgbm/lib not found under ${VENV}. Run 'uv sync' first." >&2
  exit 1
fi
LGLIB="${LGDIR}/lib_lightgbm.dylib"

# Already working? (libomp present + lightgbm imports) -> nothing to do.
if [[ -f "${LGDIR}/libomp.dylib" ]] && \
   "${VENV}/bin/python" -c "import lightgbm" >/dev/null 2>&1; then
  echo "libomp already in place and lightgbm imports cleanly — nothing to do."
  exit 0
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "error: Homebrew not found. Install libomp another way, or install brew." >&2
  exit 1
fi

# Fetch an arm64 libomp bottle (works even from an Intel brew via --bottle-tag).
echo "Fetching arm64 libomp bottle…"
FETCH_LOG="$(mktemp)"
for tag in arm64_tahoe arm64_sequoia arm64_sonoma arm64_ventura; do
  if brew fetch --force --bottle-tag="${tag}" libomp >"${FETCH_LOG}" 2>&1; then
    echo "  got bottle for ${tag}"
    break
  fi
done

BOTTLE="$(find "$(brew --cache)/downloads" -iname "*libomp*${tag}*.bottle.tar.gz" 2>/dev/null | head -1)"
[[ -z "${BOTTLE}" ]] && BOTTLE="$(find "$(brew --cache)" -iname "*libomp*arm64*.bottle.tar.gz" 2>/dev/null | head -1)"
if [[ -z "${BOTTLE}" ]]; then
  echo "error: could not locate a downloaded arm64 libomp bottle." >&2
  exit 1
fi

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
tar -xzf "${BOTTLE}" -C "${WORK}"
SRC="$(find "${WORK}" -name libomp.dylib | head -1)"
if ! file "${SRC}" | grep -q arm64; then
  echo "error: extracted libomp is not arm64." >&2
  exit 1
fi

echo "Installing libomp.dylib into ${LGDIR}…"
cp "${SRC}" "${LGDIR}/libomp.dylib"
chmod u+w "${LGDIR}/libomp.dylib" "${LGLIB}"
install_name_tool -id @rpath/libomp.dylib "${LGDIR}/libomp.dylib"
# Add @loader_path so lightgbm finds the sibling libomp (idempotent).
otool -l "${LGLIB}" | grep -q "@loader_path" || install_name_tool -add_rpath @loader_path "${LGLIB}"
codesign --force --sign - "${LGDIR}/libomp.dylib"
codesign --force --sign - "${LGLIB}"

echo "Verifying lightgbm imports…"
"${VENV}/bin/python" -c "import lightgbm, numpy as np; m=lightgbm.LGBMClassifier(n_estimators=3).fit(np.random.rand(40,3), np.random.randint(0,2,40)); print('lightgbm OK', lightgbm.__version__)"
echo "Done."
