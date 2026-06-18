"""Centralised logging built on loguru, plus an in-memory ring buffer so the UI
can surface recent log lines in a debug panel."""

from __future__ import annotations

import sys
from collections import deque
from functools import lru_cache

from loguru import logger

from deepsearch.config import get_settings

# Recent log lines, newest last. Read by the Streamlit debug panel.
_LOG_BUFFER: deque[str] = deque(maxlen=400)


def _buffer_sink(message) -> None:
    _LOG_BUFFER.append(message.rstrip("\n"))


def get_recent_logs(limit: int = 200) -> list[str]:
    """Return the most recent log lines (oldest first)."""
    lines = list(_LOG_BUFFER)
    return lines[-limit:]


def clear_logs() -> None:
    _LOG_BUFFER.clear()


@lru_cache(maxsize=1)
def _configure() -> None:
    logger.remove()
    level = get_settings().runtime.log_level.upper()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
        "<cyan>{name}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt, colorize=True)
    # Plain-text mirror into the ring buffer for the UI (no ANSI colour codes).
    logger.add(
        _buffer_sink,
        level=level,
        format="{time:HH:mm:ss} | {level: <7} | {name} - {message}",
        colorize=False,
    )


def get_logger(name: str):
    """Return a module-scoped logger with global config applied once."""
    _configure()
    return logger.bind(name=name)
