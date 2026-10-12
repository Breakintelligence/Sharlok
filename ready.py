"""Prerequisite checks that run before the Sharlok server starts.

Mirrors Captain's ready.py: makes sure the Ollama server is reachable
(starting `ollama serve` when it is not) and that every model Sharlok uses
is available locally (pulling any that are missing).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

import requests

log = logging.getLogger("Sharlok.ready")

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
GENERATION_MODEL = os.environ.get("Sharlok_MODEL", "gemma4:latest")
VISION_MODEL = os.environ.get("Sharlok_VISION_MODEL", "gemma4:latest")
EMBED_MODEL = os.environ.get("Sharlok_EMBED_MODEL", "nomic-embed-text")
REQUIRED_MODELS = tuple(dict.fromkeys([GENERATION_MODEL, VISION_MODEL, EMBED_MODEL]))

_REQUEST_TIMEOUT_S = 3
_START_TIMEOUT_S = 30
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def _with_tag(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def _is_local_host() -> bool:
    return (urlparse(OLLAMA_HOST).hostname or "") in _LOCAL_HOSTS


def _ollama_up() -> bool:
    try:
        return requests.get(f"{OLLAMA_HOST}/api/tags", timeout=_REQUEST_TIMEOUT_S).status_code == 200
    except requests.RequestException:
        return False


def _installed_models() -> set[str]:
    resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=_REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    return {m["name"] for m in resp.json().get("models", [])}


def _start_server() -> None:
    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama CLI not found. Install it with: make install-ollama")

    log.warning("Ollama is not running. Starting `ollama serve` in the background...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.monotonic() + _START_TIMEOUT_S
    while time.monotonic() < deadline:
        if _ollama_up():
            return
        time.sleep(1)
    raise RuntimeError(f"Ollama did not respond at {OLLAMA_HOST} within {_START_TIMEOUT_S}s.")


def _pull(model: str) -> None:
    log.info("Pulling Ollama model %s (this can take a while)...", model)
    try:
        subprocess.run(["ollama", "pull", model], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to pull Ollama model {model}: {exc}") from exc


def ensure_ollama_ready() -> None:
    """Start Ollama if needed and pull missing models. Raises RuntimeError on failure."""
    if not _ollama_up():
        if not _is_local_host():
            raise RuntimeError(f"Ollama is not reachable at {OLLAMA_HOST} (remote hosts are not started automatically).")
        _start_server()
    log.info("Ollama is running at %s", OLLAMA_HOST)

    installed = _installed_models()
    for model in REQUIRED_MODELS:
        if _with_tag(model) in installed:
            log.info("Model ready: %s", model)
        else:
            _pull(model)
