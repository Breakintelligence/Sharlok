"""Sharlok FastAPI Server Launcher.

This script runs the Sharlok threat modeling web server.
Access the interface at http://localhost:8000

Before starting, it makes sure Ollama is running and the required models are
present (starting `ollama serve` and pulling missing models automatically).

Use --build-index to create or update the FAISS knowledge store and exit.
"""

import argparse
import logging
import os
import sys

import requests

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))


def main() -> None:
    parser = argparse.ArgumentParser(description="Sharlok server launcher.")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Create or update the FAISS store from PDF/, Repos/, Tools/ and webpages.yml, then exit",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from ready import ensure_ollama_ready

    try:
        ensure_ollama_ready()
    except (RuntimeError, requests.RequestException) as exc:
        sys.exit(f"Startup check failed: {exc}")

    if args.build_index:
        from knowledge_base import build_index

        print(f"Indexed {build_index()} new chunk(s).")
        return

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
