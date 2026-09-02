"""Health check and KB statistics endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from avicennaguard.api.dependencies import get_avicennaguard
from avicennaguard.pipeline.avicennaguard import AvicennaGuard

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response payload schema."""

    status: str
    version: str
    kb_stats: Dict[str, Any]
    model: str


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health(lg: AvicennaGuard = Depends(get_avicennaguard)) -> HealthResponse:
    """Returns API status, knowledge base statistics, and active model."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        kb_stats=lg.kb.stats,
        model=lg._parser.model,
    )
