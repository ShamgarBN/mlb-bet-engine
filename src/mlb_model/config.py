"""Central configuration for the MLB model.

Loads paths and tunable parameters from environment variables (and an optional
``.env`` file at the project root) with sensible defaults. All paths resolve
relative to the project root so the package works no matter where it's invoked.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for data, modeling, and backtesting."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        env_prefix="MLB_",
        extra="ignore",
    )

    # --- Paths ---
    project_root: Path = Field(default=PROJECT_ROOT)
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    interim_dir: Path = Field(default=PROJECT_ROOT / "data" / "interim")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    cache_dir: Path = Field(default=PROJECT_ROOT / "data" / "cache")
    warehouse_path: Path = Field(default=PROJECT_ROOT / "data" / "warehouse.duckdb")
    model_dir: Path = Field(default=PROJECT_ROOT / "models")
    logs_dir: Path = Field(default=PROJECT_ROOT / "logs")

    # --- Data window ---
    backtest_start_year: int = 2014
    backtest_end_year: int = 2025
    statcast_available_from: int = 2015  # pitch-level Statcast starts 2015

    # --- HTTP behavior ---
    http_user_agent: str = (
        "mlb-model/0.1 (research; personal use; respects robots.txt and rate limits)"
    )
    http_timeout_seconds: float = 30.0
    http_max_retries: int = 4
    http_backoff_seconds: float = 2.0

    # --- External API keys (loaded from .env via MLB_ prefix) ---
    # The Odds API (the-odds-api.com) — free-tier signup gives 500
    # credits/month which is plenty for one live-slate pull per day
    # (~3 credits / call). Without a key, live odds ingest no-ops and
    # the predict pipeline falls back to a league-average baseline
    # total line. Accepts a comma-separated list of keys; when the
    # active key's monthly quota runs out, fetches fail over to the
    # next key in the list.
    odds_api_key: str | None = None

    # --- Alerts (Discord) ---
    # Create a webhook in your Discord channel (Edit Channel → Integrations →
    # Webhooks → New Webhook → Copy URL) and set MLB_DISCORD_WEBHOOK_URL.
    # When set, the daily morning sync posts any Premium/Strong game + prop
    # picks. Without it, the alert step no-ops.
    discord_webhook_url: str | None = None

    # Cap the alert so it stays a tight "top plays" list, not a wall. Picks are
    # Premium-first then by edge; the rest are summarized as "+N more". A single
    # high-K pitcher can make every opposing batter a "1+ K" pick, so the prop
    # cap matters. Raise these (or set very high) to see everything.
    alert_max_game_picks: int = 8
    # The grouped prop format (one line per team-vs-pitcher) is ~4x denser
    # than the old one-pick-per-line layout, so the cap affords more picks
    # in less screen space.
    alert_max_prop_picks: int = 24

    # --- Automation schedule (local time of the Mac) ---
    # Games starting before this local hour are "early-day" (covered by the
    # 11 AM window); the rest are "late-day" (covered by the 4:30 PM window).
    early_late_cutoff_hour: int = 17  # 5 PM local

    # morning-sync is now DATA-ONLY (schedule / odds / weather / cache) and runs
    # before the alert streams so they read fresh predictions. Its old combined
    # 11 AM Discord alert has been split into the staggered streams below.
    morning_sync_hour: int = 10
    morning_sync_minute: int = 45

    # Staggered Discord alert streams. Each is its own LaunchAgent + CLI call,
    # deduped independently. (hour, minute) in the Mac's local timezone.
    # Early window (games starting before early_late_cutoff_hour):
    alert_early_pitchers_hm: tuple[int, int] = (11, 0)   # high-K starters
    alert_early_hitters_hm: tuple[int, int] = (11, 7)    # hit/HR/TB props
    alert_cold_hitters_hm: tuple[int, int] = (11, 15)    # cold .300 bats
    alert_cold_sluggers_hm: tuple[int, int] = (11, 30)   # cold HR bats
    alert_game_markets_hm: tuple[int, int] = (11, 45)    # ML/RL/O-U, whole slate
    # Late window (games starting at/after the cutoff):
    alert_late_hitters_hm: tuple[int, int] = (16, 30)    # remaining hitters
    alert_late_pitchers_hm: tuple[int, int] = (16, 37)   # remaining starters
    alert_late_games_hm: tuple[int, int] = (16, 45)      # late ML/RL/O-U if changed

    # Retained for back-compat with the old single-shot afternoon job and any
    # manual `afternoon-props` call; the scheduled late window now uses the
    # streams above.
    afternoon_props_hour: int = 16
    afternoon_props_minute: int = 30
    # The daily-recap LaunchAgent posts yesterday's scorecard (win rates by
    # tier for alerted picks) each morning before the picks alerts.
    daily_recap_hour: int = 8
    daily_recap_minute: int = 0

    # --- Modeling ---
    # Backtest needs lots of draws to get tight per-game tail estimates
    # we can compare against historical truth; daily inference doesn't
    # benefit from more than ~10k draws (the win-probability swings
    # between 9999 vs 50000 draws is < 0.005 -- well below the rest of
    # our model's error budget) but takes 5x longer at 50k.
    monte_carlo_iterations: int = 50_000               # default / backtest
    monte_carlo_iterations_inference: int = 10_000     # daily slate
    random_seed: int = 42

    # --- Confidence tiers used to slice accuracy by edge ---
    tier_top_3_pct: float = 0.03
    tier_top_10_pct: float = 0.10
    tier_top_30_pct: float = 0.30

    # --- Bet sizing / Kelly (for ROI calc only; we don't place bets) ---
    kelly_fraction: float = 0.25  # quarter-Kelly to limit variance

    def ensure_dirs(self) -> None:
        """Create on-disk directories the model needs (idempotent)."""
        for path in (
            self.data_dir,
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.cache_dir,
            self.model_dir,
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
