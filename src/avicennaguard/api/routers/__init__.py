"""
API routers package for AvicennaGuard REST endpoints.

Exports individual sub-routers for health, knowledge base, and validation operations.
"""

from avicennaguard.api.routers import health, kb, validate

__all__ = [
    "health",
    "kb",
    "validate",
]
