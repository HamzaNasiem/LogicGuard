"""
Dependency injection for LogicGuard API.

Separate module to avoid circular imports between main.py and routers.
"""

from pathlib import Path

from fastapi import HTTPException

from logicguard.kb.loader import KnowledgeBase
from logicguard.pipeline.logicguard import LogicGuard

# Shared state — set by lifespan in main.py
_lg: LogicGuard | None = None


def set_logicguard(instance: LogicGuard) -> None:
    global _lg
    _lg = instance


def get_logicguard() -> LogicGuard:
    if _lg is None:
        raise HTTPException(status_code=503, detail="LogicGuard not initialized")
    return _lg


def reload_logicguard(kb_path: Path, model: str) -> LogicGuard:
    """Hot-reload KB after upload."""
    global _lg
    kb = KnowledgeBase(kb_path)
    _lg = LogicGuard(kb, model=model)
    return _lg
