"""
/validate and /batch API endpoints.

POST /api/v1/validate  — Single query validation
POST /api/v1/batch     — Batch validation (max 50 queries)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from avicennaguard.api.dependencies import get_avicennaguard
from avicennaguard.pipeline.avicennaguard import AvicennaGuard

router = APIRouter()


class ValidateRequest(BaseModel):
    """Payload schema for single query validation."""

    question: str = Field(..., description="Natural language question to validate", example="Are all dogs mammals?")
    llm_answer: Optional[str] = Field(None, description="LLM's raw answer (yes/no) before AvicennaGuard override", example="yes")
    model: str = Field("llama3.2:3b", description="Ollama model for Stage 1 parsing")


class ValidateResponse(BaseModel):
    """Response schema for single query validation."""

    question: str
    epistemic_state: str = Field(..., description="YAQEEN | WAHM | SHAKK | ZANN")
    graph_answer: Optional[bool] = Field(None, description="BFS result. None if KB doesn't cover this query")
    llm_answer: str
    final_answer: str = Field(..., description="Answer returned to user (may override LLM answer)")
    covered: bool = Field(..., description="Whether KB covers this entity/relation")
    intercepted: bool = Field(..., description="True if a hallucination was caught")
    query_type: str = Field(..., description="taxonomic | categorical | hypothetical | non-logical")
    subject: str
    predicate: str
    path: List[str] = Field(..., description="BFS traversal path (audit trail)")
    latency_ms: Dict[str, Any]


class BatchRequest(BaseModel):
    """Payload schema for batch query validation."""

    queries: List[ValidateRequest] = Field(..., max_length=50)


class BatchResponse(BaseModel):
    """Response schema for batch query validation."""

    total: int
    intercepted: int
    results: List[ValidateResponse]


@router.post("/validate", response_model=ValidateResponse, summary="Validate a single question")
async def validate_single(
    req: ValidateRequest,
    lg: AvicennaGuard = Depends(get_avicennaguard),
) -> ValidateResponse:
    """
    Run a single natural language question through the AvicennaGuard two-stage pipeline.

    **Stage 1 (Neural):** Constrained LLM extracts logical form (T=0.0, JSON-only)

    **Stage 2 (Symbolic):** BFS graph traversal on knowledge base — fully deterministic

    Returns the Avicennian epistemic state, BFS audit path, and whether a hallucination was intercepted.
    """
    result = lg.validate(req.question, llm_answer=req.llm_answer)
    d = result.to_dict()
    return ValidateResponse(question=req.question, **d)


@router.post("/batch", response_model=BatchResponse, summary="Validate a batch of questions")
async def validate_batch(
    req: BatchRequest,
    lg: AvicennaGuard = Depends(get_avicennaguard),
) -> BatchResponse:
    """
    Validate a batch of queries (maximum 50 per request).

    Returns aggregate interception count plus per-query results.
    """
    results: List[ValidateResponse] = []
    intercepted_count = 0

    for q in req.queries:
        result = lg.validate(q.question, llm_answer=q.llm_answer)
        d = result.to_dict()
        results.append(ValidateResponse(question=q.question, **d))
        if result.intercepted:
            intercepted_count += 1

    return BatchResponse(
        total=len(results),
        intercepted=intercepted_count,
        results=results,
    )
