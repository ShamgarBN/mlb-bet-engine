"""Entry point so ``python -m mlb_model`` works the same as ``mlb-model``."""

from mlb_model.cli import app


if __name__ == "__main__":
    app()
