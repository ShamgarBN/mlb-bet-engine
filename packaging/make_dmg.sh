#!/usr/bin/env bash
# Build a distributable DMG from ``dist/MLB Forecast.app``.
#
# Usage:
#   bash packaging/make_dmg.sh           # writes dist/MLB-Forecast-arm64.dmg
#   bash packaging/make_dmg.sh OUT.dmg   # writes to a custom path
#
# Run AFTER the PyInstaller build:
#   uv run pyinstaller --noconfirm packaging/mlb_forecast.spec
#   bash packaging/make_dmg.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_PATH="${PROJECT_ROOT}/dist/MLB Forecast.app"
# Default output is the project root so the finished installer is easy to
# find (no digging through dist/). Pass an explicit path to override.
OUT_DMG="${1:-${PROJECT_ROOT}/MLB-Forecast-arm64.dmg}"
VOL_NAME="MLB Forecast"

if [[ ! -d "${APP_PATH}" ]]; then
  echo "error: ${APP_PATH} not found. Run pyinstaller first." >&2
  exit 1
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# Stage the .app + a symlink to /Applications so the user can drag it.
mkdir -p "${WORK_DIR}/stage"
cp -R "${APP_PATH}" "${WORK_DIR}/stage/"
ln -s /Applications "${WORK_DIR}/stage/Applications"

# Remove existing DMG so hdiutil doesn't refuse to overwrite.
rm -f "${OUT_DMG}"

# UDZO = compressed read-only. -ov = overwrite, -fs HFS+ for widest compat.
hdiutil create \
  -volname "${VOL_NAME}" \
  -srcfolder "${WORK_DIR}/stage" \
  -ov \
  -format UDZO \
  -fs "HFS+" \
  "${OUT_DMG}"

echo
echo "DMG built:"
ls -lh "${OUT_DMG}"
echo
echo "Verify it mounts cleanly:"
echo "  hdiutil attach \"${OUT_DMG}\" -nobrowse -readonly"
echo "  hdiutil detach /Volumes/${VOL_NAME}"
