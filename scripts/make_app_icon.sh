#!/usr/bin/env bash
# Generate the .icns icon for the "MLB Forecast.app" bundle from the
# repo's SVG favicon. Re-run this if you change the SVG.
#
# Requires macOS (uses sips + iconutil).

set -euo pipefail

ROOT="$( cd "$( dirname "$0" )/.." && pwd )"
SVG="${ROOT}/src/mlb_model/app/static/favicon.svg"
# Write the icns to the project root; ``build_app_bundle.sh`` copies it
# into the bundle as ``applet.icns`` (replacing the AppleScript droplet).
OUTPUT_ICNS="${ROOT}/MLB Forecast.icns"
ICONSET=$(mktemp -d -t mlbforecast_iconset.XXXXX)

if [[ ! -f "${SVG}" ]]; then
  echo "Source SVG not found at ${SVG}" >&2
  exit 1
fi

mkdir -p "${ICONSET}"

# Convert SVG -> 1024x1024 PNG via qlmanage (ships with macOS).
TMP_PNG=$(mktemp -t mlbforecast_icon).png
qlmanage -t -s 1024 -o "$(dirname "${TMP_PNG}")" "${SVG}" >/dev/null
mv "$(dirname "${TMP_PNG}")/favicon.svg.png" "${TMP_PNG}"

declare -a SIZES=(16 32 64 128 256 512 1024)
for s in "${SIZES[@]}"; do
  sips -z "${s}" "${s}" "${TMP_PNG}" --out "${ICONSET}/icon_${s}x${s}.png" >/dev/null
  if (( s != 1024 )); then
    dbl=$((s * 2))
    sips -z "${dbl}" "${dbl}" "${TMP_PNG}" --out "${ICONSET}/icon_${s}x${s}@2x.png" >/dev/null
  fi
done

iconutil -c icns -o "${OUTPUT_ICNS}" "${ICONSET}"
rm -rf "${ICONSET}" "${TMP_PNG}"

echo "Wrote ${OUTPUT_ICNS}"
