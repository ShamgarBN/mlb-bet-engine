"""Install macOS ``launchd`` agents for the automation jobs.

We use user-scope LaunchAgents (``~/Library/LaunchAgents``) so the
installer does not need ``sudo`` and the jobs only run while the user
is logged in -- which is exactly when we want them to (the desktop
app is also user-facing).

Two agents are installed:

* ``com.local.mlbforecast.morning-sync`` -- runs daily at
  ``settings.morning_sync_hour:morning_sync_minute`` (default 11:00 local).
* ``com.local.mlbforecast.weekly-train``  -- runs Sundays at 03:00.

The plist's ``ProgramArguments`` calls ``uv run`` from the project root
so the launcher reuses the same virtualenv the user already has set up.
"""

from __future__ import annotations

import plistlib
import shutil
import subprocess
from pathlib import Path
from typing import Final

from mlb_model.logging import get_logger

log = get_logger("automation.scheduler")

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
LAUNCH_AGENTS_DIR: Final[Path] = Path.home() / "Library" / "LaunchAgents"

MORNING_LABEL: Final[str] = "com.local.mlbforecast.morning-sync"
WEEKLY_LABEL: Final[str] = "com.local.mlbforecast.weekly-train"


def _resolve_uv() -> str:
    """Find a full path to ``uv``. LaunchAgents run with a stripped PATH."""
    candidate = shutil.which("uv")
    if candidate:
        return candidate
    # Common install locations that aren't on the default ``launchd`` PATH.
    for guess in ("/opt/homebrew/bin/uv", "/usr/local/bin/uv", f"{Path.home()}/.local/bin/uv"):
        if Path(guess).exists():
            return guess
    raise RuntimeError(
        "Cannot find `uv`. Install it (https://docs.astral.sh/uv/) "
        "before installing the LaunchAgents."
    )


def _build_plist(
    label: str,
    *,
    args: list[str],
    calendar_interval: dict[str, int],
) -> dict[str, object]:
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return {
        "Label": label,
        "ProgramArguments": args,
        "WorkingDirectory": str(PROJECT_ROOT),
        "StartCalendarInterval": calendar_interval,
        "RunAtLoad": False,
        "StandardOutPath": str(log_dir / f"{label}.out.log"),
        "StandardErrorPath": str(log_dir / f"{label}.err.log"),
        # Keep the LaunchAgent's PATH small and explicit. Inheriting from
        # ``os.environ`` would carry whatever junk happened to be on PATH
        # when install was run -- bad for reproducibility.
        "EnvironmentVariables": {
            "PATH": (
                f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/local/bin:"
                "/usr/bin:/bin:/usr/sbin:/sbin"
            ),
            "LC_ALL": "en_US.UTF-8",
            "LANG": "en_US.UTF-8",
        },
    }


def _write_plist(label: str, payload: dict[str, object]) -> Path:
    LAUNCH_AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    path = LAUNCH_AGENTS_DIR / f"{label}.plist"
    with path.open("wb") as f:
        plistlib.dump(payload, f, sort_keys=True)
    return path


def _launchctl(*args: str) -> None:
    """Run ``launchctl`` and log a warning on failure (non-fatal)."""
    try:
        subprocess.run(["launchctl", *args], check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        log.warning(
            "scheduler.launchctl.failed",
            args=list(args),
            stderr=exc.stderr.decode("utf-8", errors="replace"),
        )


def install() -> dict[str, Path]:
    """Write + load both LaunchAgents. Returns paths keyed by label."""
    uv = _resolve_uv()

    from mlb_model.config import settings

    morning_plist = _build_plist(
        MORNING_LABEL,
        args=[uv, "run", "mlb-model", "morning-sync"],
        calendar_interval={
            "Hour": int(settings.morning_sync_hour),
            "Minute": int(settings.morning_sync_minute),
        },
    )
    weekly_plist = _build_plist(
        WEEKLY_LABEL,
        args=[uv, "run", "mlb-model", "weekly-train"],
        # Weekday: 0 or 7 = Sunday in launchd.
        calendar_interval={"Weekday": 0, "Hour": 3, "Minute": 0},
    )

    paths: dict[str, Path] = {}
    for label, payload in [(MORNING_LABEL, morning_plist), (WEEKLY_LABEL, weekly_plist)]:
        path = _write_plist(label, payload)
        # Unload any prior version then load the fresh one.
        _launchctl("unload", str(path))
        _launchctl("load", "-w", str(path))
        log.info("scheduler.installed", label=label, path=str(path))
        paths[label] = path
    return paths


def uninstall() -> None:
    """Unload and delete both LaunchAgents."""
    for label in (MORNING_LABEL, WEEKLY_LABEL):
        path = LAUNCH_AGENTS_DIR / f"{label}.plist"
        if path.exists():
            _launchctl("unload", str(path))
            path.unlink()
            log.info("scheduler.removed", label=label)
