"""Loguru-based logging with structured JSON for files and rich console output.

Sensitive fields are never logged. Sanitization on string interpolation is
the caller's responsibility -- this module focuses on routing, rotation, and
correlation ids for the data pipeline and modeling stages.
"""

from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from mlb_model.config import settings

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "| <level>{level: <8}</level> "
    "| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> "
    "- <level>{message}</level>"
)

_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Configure global logger sinks.

    Idempotent: safe to call multiple times.
    """
    global _configured
    if _configured:
        return

    logger.remove()  # remove default
    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=level,
        backtrace=True,
        diagnose=False,  # diagnose=True leaks variables -- never enable in prod
    )

    settings.logs_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        settings.logs_dir / "mlb_model.log",
        rotation="20 MB",
        retention="30 days",
        level=level,
        serialize=True,  # JSON lines
        backtrace=True,
        diagnose=False,
    )
    _configured = True


def get_logger(name: str | None = None, **bound: Any) -> Any:
    """Return a logger bound with a name and optional context fields."""
    configure_logging()
    return logger.bind(component=name or "mlb_model", **bound)
