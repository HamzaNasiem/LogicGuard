#!/usr/bin/env python3
"""
AvicennaGuard Production API — Launcher
=====================================
Starts the FastAPI server on port 8000.

Usage:
    python run_api.py
    python run_api.py --port 8001
    python run_api.py --kb knowledge_base_1200.json --port 8000

Endpoints (after startup):
    GET  http://localhost:8000/docs          → Swagger UI
    GET  http://localhost:8000/api/v1/health → Health check
    POST http://localhost:8000/api/v1/validate
    POST http://localhost:8000/api/v1/batch
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="AvicennaGuard FastAPI server")
parser.add_argument("--host",  default="0.0.0.0",         help="Bind host (default: 0.0.0.0)")
parser.add_argument("--port",  type=int, default=8000,     help="Port (default: 8000)")
parser.add_argument("--kb",    default="knowledge_base_1200.json",
                    help="Knowledge base JSON path (default: extended KB)")
parser.add_argument("--model", default="llama3.2:3b",      help="Ollama model for Stage 1 parser")
parser.add_argument("--reload",action="store_true",         help="Enable hot reload (dev mode)")
args = parser.parse_args()

# Inject KB path and model via env vars (picked up by main.py lifespan)
os.environ["AVICENNAGUARD_KB"] = args.kb
os.environ["LOGICGUARD_KB"] = args.kb
os.environ["AVICENNAGUARD_MODEL"] = args.model
os.environ["LOGICGUARD_MODEL"] = args.model

# Ensure src/ is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import uvicorn

print("=" * 60)
print("  AvicennaGuard Production API")
print("=" * 60)
print(f"  KB     : {args.kb}")
print(f"  Model  : {args.model}")
print(f"  URL    : http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
print(f"  Docs   : http://localhost:{args.port}/docs")
print("=" * 60)

uvicorn.run(
    "avicennaguard.api.main:app",
    host=args.host,
    port=args.port,
    reload=args.reload,
    log_level="info",
)
