"""
Structured logging configuration for AvicennaGuard.

Usage:
    from avicennaguard.utils import setup_logging
    setup_logging(level="INFO")
"""

from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure structured logging with consistent timestamp and level formatting.

    Args:
        level: Logging level string (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )
    # Suppress noisy third-party logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("ollama").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
