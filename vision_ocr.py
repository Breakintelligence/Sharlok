"""Ollama vision (image → text) helper for Sharlok's diagram-extraction step.

Vision stays on Ollama: the local gemma4 blob uses Ollama's own native
multimodal engine and is NOT loadable by vanilla llama.cpp (confirmed directly:
llama-server errors with "wrong number of tensors; expected 2131, got 720" when
pointed at the Ollama gemma4 GGUF blob). All other Sharlok text generation
(STRIDE writeup, recommendations, chat) runs on llama.cpp instead — see
``llama_cpp_utils.py`` and ``threat_modeling.py``.

Environment:
  OLLAMA_HOST            — Ollama API host (default ``http://localhost:11434``)
  Sharlok_VISION_MODEL  — vision model tag (default ``gemma4:latest``)
"""

from __future__ import annotations

import base64
import logging
import mimetypes
import os
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger("Sharlok.vision")

_OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_VISION_MODEL = os.environ.get("Sharlok_VISION_MODEL", "gemma4:latest")
_USER_AGENT = os.environ.get("USER_AGENT", "Sharlok/1.0 (threat-modeling-agent)")


def _ensure_ollama_running() -> None:
    """Start `ollama serve` in the background if the API is unreachable."""
    health = _OLLAMA.rstrip("/") + "/api/tags"
    try:
        requests.get(health, timeout=2)
        return  # already up
    except Exception:
        pass
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(15):
        time.sleep(1)
        try:
            requests.get(health, timeout=2)
            return
        except Exception:
            continue
    raise RuntimeError("Started `ollama serve` but it never became ready. Check that ollama is installed.")


def _image_to_base64(path: str) -> tuple[str, str]:
    """Convert image file to base64 with MIME type detection."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    mime, _ = mimetypes.guess_type(str(p))
    if mime not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
        mime = mime or "image/png"
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return b64, mime


def ollama_vision_analyze(
    image_path: str,
    *,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 300,
) -> str:
    """Send one image to Ollama (gemma4 by default); return assistant text.

    Args:
        image_path: Path to image file
        system_prompt: System context
        user_prompt: User query about the image
        model: Ollama model name (defaults to Sharlok_VISION_MODEL)
        timeout: Request timeout in seconds

    Returns:
        Assistant's response text

    Raises:
        FileNotFoundError: If image file doesn't exist
        RuntimeError: If Ollama is unreachable or model returns empty content
    """
    _ensure_ollama_running()

    b64, _mime = _image_to_base64(image_path)
    model = model or _VISION_MODEL

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {
            "role": "user",
            "content": user_prompt or "Analyze this architecture diagram.",
            "images": [b64],
        }
    )

    url = _OLLAMA.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2, "num_ctx": 8192},
    }
    headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}

    log.info("Vision request: model=%s image=%s", model, image_path)
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    msg = data.get("message") or {}
    content = msg.get("content") or ""
    if not content.strip():
        raise RuntimeError(
            f"Vision model returned empty content. Is `{model}` pulled? Try: ollama pull {model.split(':')[0]}"
        )
    return content.strip()
