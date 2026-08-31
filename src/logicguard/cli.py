"""Console entry point for LogicGuard API server."""

import argparse
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="LogicGuard FastAPI server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--kb",
        default="knowledge_base_1200.json",
    )
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    os.environ["LOGICGUARD_KB"] = args.kb
    os.environ["LOGICGUARD_MODEL"] = args.model

    import uvicorn

    uvicorn.run(
        "logicguard.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
