"""Launch the app in a native macOS window via pywebview.

This module gives the user a "click an icon to use the app" experience
with no visible terminal, no separate browser tab, and no awareness of
the embedded HTTP server. Internally:

  1) A free local port is reserved on 127.0.0.1.
  2) Uvicorn starts in a daemon thread, serving the existing FastAPI
     app. We poll ``/healthz`` until it responds (max 20 s).
  3) A pywebview window opens pointed at the local URL. macOS uses its
     native WebKit; no Chromium is downloaded.
  4) When the user closes the window pywebview returns and the
     interpreter exits, terminating the daemon thread.

The HTTP listener stays bound to ``127.0.0.1`` only; firewalls and
network listeners on this machine are unaffected.
"""

from __future__ import annotations

import socket
import threading
import time
import urllib.request
from typing import Final

import uvicorn

from mlb_model.logging import get_logger

log = get_logger("app.desktop")

WINDOW_TITLE: Final[str] = "MLB Forecast"
DEFAULT_WIDTH: Final[int] = 1280
DEFAULT_HEIGHT: Final[int] = 860


def _find_free_port() -> int:
    """Ask the OS for an unused localhost port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _UvicornThread(threading.Thread):
    """Uvicorn server running on a daemon thread.

    We hold a reference to the underlying ``uvicorn.Server`` so the
    pywebview ``closed`` callback can signal a graceful shutdown
    (``should_exit = True``) rather than hard-killing the thread.
    """

    def __init__(self, host: str, port: int) -> None:
        super().__init__(daemon=True, name="mlb-model-uvicorn")
        config = uvicorn.Config(
            "mlb_model.app.main:app",
            host=host,
            port=port,
            log_level="warning",
            access_log=False,
            loop="asyncio",
        )
        self.server = uvicorn.Server(config)

    def run(self) -> None:  # noqa: D401 -- thread entrypoint
        try:
            self.server.run()
        except Exception:  # noqa: BLE001 -- log only; main thread proceeds
            log.exception("desktop.uvicorn.crashed")

    def stop(self) -> None:
        self.server.should_exit = True


def _wait_for_health(url: str, timeout_s: float = 20.0) -> bool:
    """Poll ``url`` until it returns HTTP 200 or we hit the timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:  # noqa: S310 -- localhost
                if resp.status == 200:
                    return True
        except Exception:  # noqa: BLE001 -- expected before server warms up
            pass
        time.sleep(0.1)
    return False


def _maybe_run_automation_lazy() -> None:
    """Run morning sync / weekly retrain in a background thread if due.

    This is the "no LaunchAgent required" backstop: even users who skip
    ``mlb-model install-schedule`` get fresh data the first time they
    open the app each day.
    """
    from mlb_model.automation import morning_sync, weekly_train

    def _worker() -> None:
        try:
            if morning_sync.needs_run():
                log.info("desktop.lazy_automation.morning_sync.start")
                morning_sync.run_morning_sync()
        except Exception:  # noqa: BLE001
            log.exception("desktop.lazy_automation.morning_sync.failed")

        try:
            if weekly_train.needs_run():
                log.info("desktop.lazy_automation.weekly_train.start")
                weekly_train.run_weekly_train()
        except Exception:  # noqa: BLE001
            log.exception("desktop.lazy_automation.weekly_train.failed")

    threading.Thread(target=_worker, daemon=True, name="mlb-model-lazy-sync").start()


def launch_native_window(
    *,
    host: str = "127.0.0.1",
    port: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> None:
    """Open the app in a native window. Blocks until the user closes it."""
    if host not in {"127.0.0.1", "localhost"}:
        raise ValueError(f"Refusing to bind native window to non-local host {host!r}.")

    # pywebview is imported lazily so the rest of the package stays usable
    # in environments without a display (CI / docker / SSH sessions).
    import webview

    port = port or _find_free_port()
    url = f"http://{host}:{port}"
    health_url = f"{url}/healthz"

    server = _UvicornThread(host, port)
    server.start()

    if not _wait_for_health(health_url):
        server.stop()
        raise RuntimeError(
            f"Local prediction server did not come up at {url} within 20 s. "
            f"Run `mlb-model serve` directly to see startup logs."
        )

    _maybe_run_automation_lazy()

    log.info("desktop.window.open", url=url)
    window = webview.create_window(
        title=WINDOW_TITLE,
        url=url,
        width=width,
        height=height,
        min_size=(960, 640),
        resizable=True,
        text_select=True,
    )

    def _on_closed() -> None:
        log.info("desktop.window.closed")
        server.stop()

    window.events.closed += _on_closed

    # Default backend on macOS is Cocoa/WebKit. We don't request a
    # specific backend so it works on Linux (gtk/qt) too if someone
    # tries it.
    webview.start()

    # Just in case the window event fires before the server thread sees it.
    server.stop()
