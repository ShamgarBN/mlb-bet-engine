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
