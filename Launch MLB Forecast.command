#!/usr/bin/env bash
# Double-click this file to open the MLB Forecast app.
#
# Identical to opening "MLB Forecast.app" -- it's just a fallback for
# anyone who prefers a Terminal launch over a Finder bundle (e.g. when
# the .app's quarantine attributes are blocking it).

set -euo pipefail

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/"
  read -r -p "Press return to close…" _
  exit 1
fi

exec uv run mlb-model app
