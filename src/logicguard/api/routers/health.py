"""Health check and KB stats endpoints."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from logicguard.api.dependencies import get_logicguard
from logicguard.pipeline.logicguard import LogicGuard

router = APIRouter()


class HealthResponse(BaseModel):
    status:    str
    version:   str
    kb_stats:  dict
    model:     str


@router.get("/health", response_model=HealthResponse, summary="API health check")
async def health(lg: LogicGuard = Depends(get_logicguard)) -> HealthResponse:
    """Returns API status, knowledge base statistics, and active model."""
    return HealthResponse(
        status="ok",
        version="2.0.0",
        kb_stats=lg.kb.stats,
        model=lg._parser.model,
    )
