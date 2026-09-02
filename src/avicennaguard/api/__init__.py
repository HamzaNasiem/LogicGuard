"""
FastAPI REST interface package for AvicennaGuard hallucination validation.

Exports the FastAPI application instance for running with ASGI servers:
    from avicennaguard.api import app
"""

from avicennaguard.api.main import app

__all__ = [
    "app",
]
