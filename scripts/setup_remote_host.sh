#!/usr/bin/env bash
# One-shot setup for an always-on "alerting host" (e.g. a Plex Mac Mini).
#
# Run this ON THE MINI, from the project root, AFTER you've put the code in
# place (git clone, or extracted a package_for_transfer.sh bundle) and dropped
# the warehouse + models into data/ and models/.
#
#   bash scripts/setup_remote_host.sh
#
# It installs deps, makes lightgbm loadable, checks your .env, and installs the
# daily LaunchAgent (default 11:00 local) that runs morning-sync → Discord alert.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m!\033[0m %s\n" "$1"; }

echo "▸ Setting up MLB Forecast alerting host in $ROOT"

# 1) uv
if ! command -v uv >/dev/null 2>&1; then
  echo "error: 'uv' not found. Install it first: https://docs.astral.sh/uv/" >&2
  exit 1
fi
ok "uv: $(uv --version)"

# 2) deps
echo "▸ uv sync (this can take a few minutes the first time)…"
uv sync
ok "virtualenv ready"

# 3) lightgbm / libomp (works on Apple Silicon and Intel)
if uv run python -c "import lightgbm" >/dev/null 2>&1; then
  ok "lightgbm imports"
else
  if [[ "$(uname -m)" == "x86_64" ]] && command -v brew >/dev/null 2>&1; then
    echo "▸ Intel Mac — installing libomp via brew…"
    brew install libomp || warn "brew install libomp failed; see brew output above"
  fi
  echo "▸ Making lightgbm loadable — running scripts/fix_libomp.sh…"
  bash scripts/fix_libomp.sh && ok "libomp installed" || warn "libomp fix failed; check the output above"
fi

# 4) .env / webhook check
if [[ -f .env ]] && grep -q "MLB_DISCORD_WEBHOOK_URL=.\+" .env; then
  ok "Discord webhook configured in .env"
else
  warn "No MLB_DISCORD_WEBHOOK_URL in .env — alerts will no-op until you add it."
  warn "  Discord → Edit Channel → Integrations → Webhooks → New Webhook → Copy URL"
  warn "  then: echo 'MLB_DISCORD_WEBHOOK_URL=...' >> .env"
fi
if [[ -f .env ]] && grep -q "MLB_ODDS_API_KEY=.\+" .env; then
  ok "Odds API key configured (live closing-line edges)"
else
  warn "No MLB_ODDS_API_KEY — game O/U picks fall back to a baseline total line."
fi

# 5) data sanity
[[ -f data/warehouse.duckdb ]] && ok "warehouse present ($(du -h data/warehouse.duckdb | cut -f1))" \
  || warn "data/warehouse.duckdb missing — copy it over before the first run."
[[ -f models/feature_spec.joblib ]] && ok "models present" \
  || warn "models/ missing — copy the trained model files over before the first run."

# 6) schedule
echo "▸ Installing the daily LaunchAgent…"
uv run mlb-model install-schedule
ok "scheduled (morning-sync runs daily at the configured hour; default 11:00 local)"

echo ""
echo "Done. Verify the webhook now with:"
echo "    uv run mlb-model alert test"
echo "Preview today's alert without sending:"
echo "    uv run mlb-model alert run --dry-run"
echo ""
echo "Tip: set the run time in .env (Mini's local tz), e.g. for 11 AM ET set the"
echo "Mini to Eastern, or use MLB_MORNING_SYNC_HOUR=11, then re-run install-schedule."
