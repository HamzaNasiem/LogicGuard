"""Console entry point for AvicennaGuard API server."""

from __future__ import annotations

import argparse
import os


def main() -> None:
    """Parse command line arguments and launch the AvicennaGuard FastAPI server."""
    parser = argparse.ArgumentParser(description="AvicennaGuard FastAPI server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding network interface host")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument(
        "--kb",
        default="knowledge_base_1200.json",
        help="Path or name of knowledge base JSON file",
    )
    parser.add_argument("--model", default="llama3.2:3b", help="Default Stage 1 LLM parser model tag")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    os.environ["AVICENNAGUARD_KB"] = args.kb
    os.environ["AVICENNAGUARD_MODEL"] = args.model

    import uvicorn

    uvicorn.run(
        "avicennaguard.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
