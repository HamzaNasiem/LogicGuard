"""
Dependency injection for AvicennaGuard API.

Separate module to avoid circular imports between main.py and routers.
"""

from pathlib import Path

from fastapi import HTTPException

from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.pipeline.avicennaguard import AvicennaGuard

# Shared state — set by lifespan in main.py
_lg: AvicennaGuard | None = None


def set_avicennaguard(instance: AvicennaGuard) -> None:
    global _lg
    _lg = instance


def get_avicennaguard() -> AvicennaGuard:
    if _lg is None:
        raise HTTPException(status_code=503, detail="AvicennaGuard not initialized")
    return _lg


def reload_avicennaguard(kb_path: Path, model: str) -> AvicennaGuard:
    """Hot-reload KB after upload."""
    global _lg
    kb = KnowledgeBase(kb_path)
    _lg = AvicennaGuard(kb, model=model)
    return _lg


# Backward compatibility aliases
set_logicguard = set_avicennaguard
get_logicguard = get_avicennaguard
reload_logicguard = reload_avicennaguard
