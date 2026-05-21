"""FastAPI factory for the local desktop app.

We expose a single ``create_app()`` function. The ``mlb-model serve``
CLI command launches uvicorn on this factory and opens the user's
browser. The app is **local-only**: it binds to 127.0.0.1 and never
accepts external traffic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from mlb_model.app.routes import router
from mlb_model.logging import get_logger

log = get_logger("app.main")

APP_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Run startup/shutdown hooks for the FastAPI app.

    On startup we eagerly ensure the DuckDB schema exists. This used to
    happen lazily inside ``mlb-model init`` only -- a first launch from
    the desktop app would crash on the first SELECT because the tables
    didn't exist. Initializing at app boot keeps every request path
    schema-safe (and is essentially free after the first call -- a
    process-local flag short-circuits subsequent invocations).
    """
    from mlb_model.data.warehouse import init_schema

    log.info("app.start")
    try:
        init_schema()
    except Exception:  # noqa: BLE001 -- never let DB init crash app boot
        log.exception("app.start.schema_init_failed")
    yield
    log.info("app.stop")


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLB Forecast",
        description="Local desktop forecasting UI for the MLB betting model.",
        version="1.0.0",
        # Local-only app -- no public exposure, no auth needed, but we keep
        # docs off so the surface area is minimal.
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=_lifespan,
    )

    app.mount(
        "/static",
        StaticFiles(directory=APP_DIR / "static"),
        name="static",
    )

    app.include_router(router)
    return app


app = create_app()
