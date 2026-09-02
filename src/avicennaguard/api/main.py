"""
AvicennaGuard FastAPI Application.

Exposes AvicennaGuard as a production REST API with:
    POST /api/v1/validate  — Single query validation
    POST /api/v1/batch     — Batch query validation (up to 50 queries)
    GET  /api/v1/health    — Health check + KB stats
    GET  /docs             — Auto-generated OpenAPI documentation

Run:
    cd D:/research/AvicennaGuard
    PYTHONPATH=src uvicorn avicennaguard.api.main:app --host 0.0.0.0 --port 8000 --reload

Or with the helper script:
    python run_api.py
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from avicennaguard.api import dependencies
from avicennaguard.api.routers import health, kb, validate
from avicennaguard.kb.loader import KnowledgeBase
from avicennaguard.pipeline.avicennaguard import AvicennaGuard

logger = logging.getLogger(__name__)

# KB path — configurable via env var or defaults to extended KB
KB_PATH      = Path(os.environ.get("AVICENNAGUARD_KB", os.environ.get("LOGICGUARD_KB", "knowledge_base_1200.json")))
DEFAULT_MODEL = os.environ.get("AVICENNAGUARD_MODEL", os.environ.get("LOGICGUARD_MODEL", "llama3.2:3b"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load KB and initialize AvicennaGuard pipeline on startup."""
    logger.info("Loading AvicennaGuard knowledge base from %s", KB_PATH)
    kb = KnowledgeBase(KB_PATH)
    lg = AvicennaGuard(kb, model=DEFAULT_MODEL)
    dependencies.set_avicennaguard(lg)
    logger.info("AvicennaGuard ready — KB: %s | Model: %s", KB_PATH.name, DEFAULT_MODEL)
    logger.info("KB stats: %s", kb.stats)
    yield
    logger.info("AvicennaGuard API shutting down.")


app = FastAPI(
    title="AvicennaGuard API",
    description=(
        "Neuro-symbolic middleware for deterministic hallucination interception "
        "in Large Language Models using Avicennian syllogistic frameworks.\n\n"
        "**Epistemic States:** YAQEEN (certain) · WAHM (intercepted) · SHAKK (unknown) · ZANN (probable)\n\n"
        "**Architecture:** Stage 1 (LLM semantic parser, T=0.0) → Stage 2 (BFS graph validator, deterministic)"
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.include_router(validate.router, prefix="/api/v1", tags=["Validation"])
app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(kb.router,       prefix="/api/v1", tags=["Knowledge Base"])
