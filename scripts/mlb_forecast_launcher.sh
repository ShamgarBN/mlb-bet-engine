#!/bin/bash
#
# Launcher embedded inside ``MLB Forecast.app``. Called by the .app's
# AppleScript entry point via ``do shell script``.
#
# Goals (same as before):
#   - NEVER die silently. Surface every failure via a native dialog.
#   - Survive macOS LaunchServices weirdness: minimal PATH, deleted-CWD
#     warnings, TCC denials on Desktop/Documents, etc.
#   - Logs to ~/Library/Logs/MLBForecast/launcher.log, which TCC always
#     permits.

LOG_DIR="${HOME}/Library/Logs/MLBForecast"
mkdir -p "${LOG_DIR}" 2>/dev/null
LOG_FILE="${LOG_DIR}/launcher.log"

log() {
  printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "${LOG_FILE}" 2>/dev/null
}

show_alert() {
  /usr/bin/osascript \
    -e "display alert \"MLB Forecast\" message \"$1\" as critical buttons {\"OK\"} default button \"OK\"" \
    >/dev/null 2>&1
}

log "----"
log "launcher.start  pid=$$  arg0=$0"

# This launcher lives at .../MLB Forecast.app/Contents/Resources/launcher.sh.
# The project root is two directories above the .app bundle.
SELF="$0"
APP_RESOURCES="$(cd "$(dirname "$SELF")" 2>/dev/null && pwd)"
APP_BUNDLE="$(cd "${APP_RESOURCES}/.." 2>/dev/null && pwd)"
APP_BUNDLE="$(cd "${APP_BUNDLE}/.." 2>/dev/null && pwd)"
PROJECT_ROOT="$(cd "${APP_BUNDLE}/.." 2>/dev/null && pwd)"

log "APP_BUNDLE=${APP_BUNDLE}"
log "PROJECT_ROOT=${PROJECT_ROOT}"

if [ -z "${PROJECT_ROOT}" ] || [ ! -d "${PROJECT_ROOT}" ]; then
  log "fatal: project root inaccessible (${PROJECT_ROOT})"
  show_alert "macOS is blocking access to the folder containing the .app.\n\nFix: System Settings → Privacy & Security → Files and Folders, and grant 'MLB Forecast' access to your Desktop (or wherever the project lives).\n\nAlternative: move the project out of ~/Desktop, ~/Documents, or ~/Downloads."
  exit 1
fi
if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then
  log "fatal: pyproject.toml missing at ${PROJECT_ROOT}"
  show_alert "Couldn't find the project at:\n${PROJECT_ROOT}\n\nThe .app must live in the project root, next to pyproject.toml."
  exit 1
fi

export PATH="${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export LC_ALL="en_US.UTF-8"
export LANG="en_US.UTF-8"

UV=""
for candidate in "${HOME}/.local/bin/uv" "/opt/homebrew/bin/uv" "/usr/local/bin/uv" "$(command -v uv 2>/dev/null)"; do
  if [ -n "${candidate}" ] && [ -x "${candidate}" ]; then
    UV="${candidate}"
    break
  fi
done
log "uv=${UV}"

if [ -z "${UV}" ]; then
  log "fatal: uv not found on PATH"
  show_alert "Cannot find 'uv'. Install it from https://docs.astral.sh/uv/ then double-click the app again.\n\nQuick install:\n  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

log "launching: ${UV} run --directory ${PROJECT_ROOT} mlb-model app"
"${UV}" run --directory "${PROJECT_ROOT}" mlb-model app >> "${LOG_FILE}" 2>&1
exit_code=$?
log "child exited with code=${exit_code}"

if [ "${exit_code}" -ne 0 ]; then
  show_alert "MLB Forecast exited with an error (code ${exit_code}).\n\nFull logs:\n${LOG_FILE}"
fi
exit "${exit_code}"
