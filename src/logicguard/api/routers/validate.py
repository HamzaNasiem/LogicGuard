"""
/validate and /batch API endpoints.

POST /api/v1/validate  — Single query validation
POST /api/v1/batch     — Batch validation (max 50 queries)
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from logicguard.api.dependencies import get_logicguard
from logicguard.pipeline.logicguard import LogicGuard

router = APIRouter()


class ValidateRequest(BaseModel):
    question:   str = Field(..., description="Natural language question to validate", example="Are all dogs mammals?")
    llm_answer: str | None = Field(None, description="LLM's raw answer (yes/no) before LogicGuard override", example="yes")
    model:      str = Field("llama3.2:3b", description="Ollama model for Stage 1 parsing")


class ValidateResponse(BaseModel):
    question:        str
    epistemic_state: str = Field(..., description="YAQEEN | WAHM | SHAKK | ZANN")
    graph_answer:    bool | None = Field(None, description="BFS result. None if KB doesn't cover this query")
    llm_answer:      str
    final_answer:    str = Field(..., description="Answer returned to user (may override LLM answer)")
    covered:         bool = Field(..., description="Whether KB covers this entity/relation")
    intercepted:     bool = Field(..., description="True if a hallucination was caught")
    query_type:      str = Field(..., description="taxonomic | categorical | hypothetical | non-logical")
    subject:         str
    predicate:       str
    path:            list[str] = Field(..., description="BFS traversal path (audit trail)")
    latency_ms:      dict


class BatchRequest(BaseModel):
    queries: list[ValidateRequest] = Field(..., max_length=50)


class BatchResponse(BaseModel):
    total:       int
    intercepted: int
    results:     list[ValidateResponse]


@router.post("/validate", response_model=ValidateResponse, summary="Validate a single question")
async def validate_single(
    req: ValidateRequest,
    lg: LogicGuard = Depends(get_logicguard),
) -> ValidateResponse:
    """
    Run a single natural language question through the LogicGuard two-stage pipeline.

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
    lg: LogicGuard = Depends(get_logicguard),
) -> BatchResponse:
    """
    Validate a batch of queries (maximum 50 per request).

    Returns aggregate interception count plus per-query results.
    """
    results = []
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
