#!/usr/bin/env bash
# Make lightgbm loadable in the project venv (Apple Silicon OR Intel).
#
# WHY: lightgbm's `lib_lightgbm.dylib` depends on `@rpath/libomp.dylib` (the
# OpenMP runtime), which the PyPI wheel does NOT bundle, and whose default
# search paths often don't match where Homebrew put it. Result:
#   Library not loaded: @rpath/libomp.dylib
# and the prediction pages (Today / Matchups / Backtest) error out.
#
# This finds a `libomp.dylib` of the right architecture, drops it next to
# lightgbm inside the venv, points lightgbm's rpath at it, and re-signs:
#   * Intel venv  -> use the brew-installed libomp (run `brew install libomp`).
#   * arm64 venv  -> use brew's if it's arm64, else fetch an arm64 bottle
#                    (covers an arm64 venv on a machine with Intel Homebrew).
#
#   uv sync
#   bash scripts/fix_libomp.sh
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
  echo "error: Homebrew not found. Install it, then 'brew install libomp'." >&2
  exit 1
fi

# Architecture lightgbm needs (matches the venv's Python).
NEED_ARCH="$(file "${LGLIB}" | grep -oE 'arm64|x86_64' | head -1)"
echo "lightgbm needs ${NEED_ARCH} libomp."

WORK="$(mktemp -d)"; trap 'rm -rf "${WORK}"' EXIT
SRC=""

# 1) Prefer a brew-installed libomp of the right arch (the Intel path: brew at
#    /usr/local ships x86_64, which matches an x86_64 venv).
BREW_OMP="$(brew --prefix libomp 2>/dev/null)/lib/libomp.dylib"
if [[ -f "${BREW_OMP}" ]] && file "${BREW_OMP}" | grep -q "${NEED_ARCH}"; then
  SRC="${BREW_OMP}"
  echo "Using brew libomp: ${SRC}"
fi

# 2) Otherwise fetch a bottle of the right arch (the arm64-venv-on-Intel-brew case).
if [[ -z "${SRC}" ]]; then
  if [[ "${NEED_ARCH}" == "arm64" ]]; then
    TAGS="arm64_tahoe arm64_sequoia arm64_sonoma arm64_ventura"
  else
    TAGS="sonoma ventura monterey"
  fi
  echo "Fetching a ${NEED_ARCH} libomp bottle…"
  for tag in ${TAGS}; do
    if NONINTERACTIVE=1 brew fetch --force --bottle-tag="${tag}" libomp >/dev/null 2>&1; then
      BOTTLE="$(find "$(brew --cache)" -iname "*libomp*${tag}*.bottle.tar.gz" 2>/dev/null | head -1)"
      [[ -n "${BOTTLE}" ]] && { tar -xzf "${BOTTLE}" -C "${WORK}"; CAND="$(find "${WORK}" -name libomp.dylib | head -1)"; file "${CAND}" | grep -q "${NEED_ARCH}" && { SRC="${CAND}"; break; }; }
    fi
  done
fi

if [[ -z "${SRC}" ]]; then
  echo "error: could not obtain a ${NEED_ARCH} libomp. On Intel, run 'brew install libomp'." >&2
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
