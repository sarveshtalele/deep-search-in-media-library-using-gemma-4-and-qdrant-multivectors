"""FastAPI backend exposing the Deep Search pipeline to the Next.js frontend."""

from deepsearch.api.main import app, run

__all__ = ["app", "run"]
