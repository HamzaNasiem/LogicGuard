"""Knowledge base upload and reload endpoints."""

import json
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel, Field

from logicguard.api.dependencies import get_logicguard, reload_logicguard
from logicguard.kb.loader import KnowledgeBase
from logicguard.pipeline.logicguard import LogicGuard

router = APIRouter()


class KBStatsResponse(BaseModel):
    path: str
    taxonomies: int
    properties: int
    conditionals: int


class KBUploadResponse(BaseModel):
    message: str
    stats: KBStatsResponse


def _validate_kb_schema(data: dict) -> None:
    for key in ("taxonomies", "properties", "conditionals"):
        if key not in data or not isinstance(data[key], dict):
            raise HTTPException(status_code=400, detail=f"Invalid KB: missing or invalid '{key}'")


@router.get("/kb/stats", response_model=KBStatsResponse, summary="Current KB statistics")
async def kb_stats(lg: LogicGuard = Depends(get_logicguard)) -> KBStatsResponse:
    stats = lg.kb.stats
    return KBStatsResponse(
        path=str(lg.kb.kb_path),
        taxonomies=stats.get("taxonomy_nodes", 0),
        properties=stats.get("property_entities", 0),
        conditionals=stats.get("conditional_rules", 0),
    )


@router.post("/kb/upload", response_model=KBUploadResponse, summary="Upload domain KB JSON")
async def kb_upload(file: UploadFile = File(...)) -> KBUploadResponse:
    """
    Upload a domain knowledge base JSON file. Validates schema and hot-reloads the pipeline.
    """
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Upload must be a .json file")

    raw = await file.read()
    try:
        data = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}") from e

    _validate_kb_schema(data)

    upload_dir = Path("data/knowledge_bases/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, suffix=".json") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)

    # Validate load before committing
    try:
        kb = KnowledgeBase(tmp_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"KB load failed: {e}") from e

    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp_path.unlink(missing_ok=True)

    model = lg_model_from_env()
    reload_logicguard(dest, model)
    stats = kb.stats

    return KBUploadResponse(
        message=f"KB loaded from {file.filename}",
        stats=KBStatsResponse(
            path=str(dest),
            taxonomies=stats.get("taxonomy_nodes", 0),
            properties=stats.get("property_entities", 0),
            conditionals=stats.get("conditional_rules", 0),
        ),
    )


def lg_model_from_env() -> str:
    import os
    return os.environ.get("LOGICGUARD_MODEL", "llama3.2:3b")
