"""
Dependency injection for AvicennaGuard API.

Separate module to avoid circular imports between main.py and routers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.pipeline.avicennaguard import AvicennaGuard

# Shared state — set by lifespan in main.py
_lg: Optional[AvicennaGuard] = None


def set_avicennaguard(instance: AvicennaGuard) -> None:
    """
    Set the active global AvicennaGuard pipeline instance.

    Args:
        instance: Initialized AvicennaGuard pipeline instance.
    """
    global _lg
    _lg = instance


def get_avicennaguard() -> AvicennaGuard:
    """
    Retrieve the active global AvicennaGuard pipeline instance.

    Returns:
        Active AvicennaGuard instance.

    Raises:
        HTTPException: If pipeline has not been initialized (503 Service Unavailable).
    """
    if _lg is None:
        raise HTTPException(status_code=503, detail="AvicennaGuard not initialized")
    return _lg


def reload_avicennaguard(kb_path: Path | str, model: str) -> AvicennaGuard:
    """
    Hot-reload KB after dynamic upload.

    Args:
        kb_path: Filesystem path to new KB JSON file.
        model: Model identifier.

    Returns:
        Newly initialized AvicennaGuard pipeline instance.
    """
    global _lg
    kb = KnowledgeBase(kb_path)
    _lg = AvicennaGuard(kb, model=model)
    return _lg


# Backward compatibility aliases
set_logicguard = set_avicennaguard
get_logicguard = get_avicennaguard
reload_logicguard = reload_avicennaguard
