"""Security architecture review pipeline, mirroring brodit's proven Ollama setup.

Sharlok is a full AI security architecture reviewer — STRIDE is one section
of the review, not the entire scope (also covers identity/secrets, network and
cloud exposure, supply chain, observability, and AI/LLM-specific risks).

Pipeline (same local-Ollama pattern as brodit's agents.py):
1. Vision extraction (Ollama gemma4) — one short call to describe the
   diagram's components and connections in plain text.
2. Knowledge-base retrieval with reranking (Ollama embeddings, see
   ``knowledge_base.py``).
3. Full review writing, recommendations, and chat — all via Ollama's
   ``/api/generate``, exactly like brodit's ``_ollama()``.
"""

import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import yaml
from typing_extensions import TypedDict

log = logging.getLogger("Sharlok.threat_modeling")

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_MODEL = os.environ.get("Sharlok_MODEL", "gemma4:latest")
_USER_AGENT = os.environ.get("USER_AGENT", "Sharlok/1.0 (threat-modeling-agent)")

_PROMPTS_CONFIG_PATH = Path(__file__).parent / "prompts.yaml"
_prompts_cache: dict = {}


def load_prompts() -> dict:
    """Load prompts from prompts.yaml config file. Cached after first load."""
    global _prompts_cache
    if _prompts_cache:
        return _prompts_cache
    if not _PROMPTS_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Prompts config not found: {_PROMPTS_CONFIG_PATH}")
    with open(_PROMPTS_CONFIG_PATH, "r", encoding="utf-8") as f:
        _prompts_cache = yaml.safe_load(f)
    return _prompts_cache


def get_prompt(name: str) -> str:
    """Get a specific prompt by name from the config."""
    prompts = load_prompts()
    if name not in prompts:
        raise KeyError(f"Prompt '{name}' not found in {_PROMPTS_CONFIG_PATH}")
    return prompts[name]


def reload_prompts() -> dict:
    """Force reload prompts from config file (useful for hot-reloading)."""
    global _prompts_cache
    _prompts_cache = {}
    return load_prompts()


def _ensure_ollama_running() -> None:
    """Start `ollama serve` in the background if the API is unreachable. Same as brodit."""
    health = _OLLAMA_HOST.rstrip("/") + "/api/tags"
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


def _ollama(prompt: str, model: str = _MODEL) -> str:
    """Call Ollama HTTP streaming API and return assembled text. Same as brodit."""
    url = _OLLAMA_HOST.rstrip("/") + "/api/generate"
    headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    payload = {"model": model, "prompt": prompt, "stream": True,
               "options": {"num_ctx": 32768}}
    parts: List[str] = []
    try:
        with requests.post(url, json=payload, headers=headers, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                token = obj.get("response", "")
                if token:
                    parts.append(token)
                if obj.get("done"):
                    break
    except requests.exceptions.ReadTimeout as exc:
        raise RuntimeError(
            "Ollama timed out after 600s. Try a shorter query or smaller source set."
        ) from exc
    return "".join(parts)


def _markdown_macos_local_image_url(abs_path: str) -> str:
    """Build a markdown image URL from absolute path."""
    from urllib.parse import quote
    posix = Path(abs_path).expanduser().resolve().as_posix()
    return quote(posix, safe="/")


def _inject_architecture_image_markdown(blog: str, image_path: str) -> str:
    """Place image reference immediately under the first heading."""
    url = _markdown_macos_local_image_url(image_path)
    posix = Path(image_path).expanduser().resolve().as_posix()
    img_line = f"![Source architecture diagram]({url})"
    lines = blog.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+\S", line):
            body = "\n".join(lines[i + 1 : i + 5])
            if url in body or posix in body:
                return blog
            out = lines[: i + 1] + ["", img_line, ""] + lines[i + 1 :]
            return "\n".join(out).rstrip() + "\n"
    return f"# Security Architecture Review\n\n{img_line}\n\n{blog.strip()}\n"


def _first_markdown_heading_text(markdown: str) -> Optional[str]:
    """Return text of the first markdown ATX heading."""
    for line in markdown.splitlines():
        m = re.match(r"^#{1,6}\s+(.+?)\s*$", line.strip())
        if m:
            return m.group(1).strip()
    return None


def analyze_architecture(
    image_path: str,
    use_knowledge_base: bool = True,
    use_web_search: bool = False,
) -> str:
    """Analyze an uploaded architecture artifact and generate a full security review.

    Accepts any format irrespective of domain: images (png/jpg/webp/gif), PDF,
    PowerPoint (pptx), or Mermaid/plain-text diagrams — see ``file_ingest.py``.
    STRIDE is one section of the output, not the entire review — see
    ``threat_model_system_prompt`` / ``threat_model_user_prompt`` in prompts.yaml
    for the full scope (identity/secrets, network/cloud, supply chain,
    observability, AI/LLM-specific risks when applicable).

    Two-call pipeline (same "fetch then write" shape as brodit's blog
    pipeline, not a third redundant generation call):
      1. Description extraction — vision (Ollama gemma4) for images, direct
         text extraction (pypdf/python-pptx/plain read) for everything else.
      2. Context gathering (FAISS hybrid retrieval always; live web search
         only when explicitly enabled, both grounded on the extracted
         description) + ONE final Ollama generation call that writes the
         review using that context. Web search adds real latency (page
         fetch + larger prompt), so it's opt-in via the frontend toggle —
         off by default means fast generation from the knowledge base alone.

    Args:
        image_path: Path to the uploaded architecture artifact
        use_knowledge_base: Whether to ground the review in the FAISS store
        use_web_search: Whether to also fetch live web context (slower)

    Returns:
        Security architecture review markdown text

    Raises:
        FileNotFoundError: If the file isn't found
        ValueError: If the file type is unsupported
        RuntimeError: If extraction or Ollama fails
    """
    from pathlib import Path as _Path
    from vision_ocr import ollama_vision_analyze
    from file_ingest import is_image, extract_architecture_description
    from knowledge_base import get_threat_modeling_context, ensure_index_exists
    from web_search import fetch_web_context

    log.info("Analyzing architecture artifact: %s", image_path)

    # Stage 1: architecture description, dispatched by file type
    ext = _Path(image_path).suffix.lower()
    if is_image(ext):
        log.info("Extracting architecture description from image (Ollama gemma4)...")
        arch_description = ollama_vision_analyze(
            image_path,
            system_prompt="Extract the architecture components, technologies named on the diagram, and connections. Be specific and literal about labels shown.",
            user_prompt="List the components, any named products/services shown, and how they connect.",
        )
    else:
        log.info("Extracting architecture description from %s file...", ext)
        arch_description = extract_architecture_description(image_path)

    # Stage 2a: knowledge base context, hybrid semantic + BM25 (not generic vector-only search)
    kb_context = ""
    if use_knowledge_base:
        try:
            if ensure_index_exists():
                log.info("Retrieving knowledge base context (hybrid semantic + BM25)...")
                kb_context = get_threat_modeling_context(arch_description)
        except Exception as e:
            log.warning("Knowledge base retrieval failed: %s", e)

    # Stage 2b: live web search, only when the frontend toggle is enabled
    web_context, web_url_count = "", 0
    if use_web_search:
        try:
            log.info("Fetching live web context for grounding...")
            web_context, web_url_count = fetch_web_context(arch_description, top_k=6)
            if web_url_count:
                log.info("Web context: %d source page(s) fetched", web_url_count)
        except Exception as e:
            log.warning("Web search failed: %s", e)
    else:
        log.info("Web search disabled — using knowledge base only for fast generation")

    # Stage 3: single final review generation via Ollama (same _ollama() pattern as brodit)
    _ensure_ollama_running()

    system_prompt = get_prompt("threat_model_system_prompt")
    user_prompt = get_prompt("threat_model_user_prompt")

    context_block = f"Architecture description (extracted from uploaded artifact):\n{arch_description}\n"
    if kb_context:
        context_block += (
            f"\nKNOWLEDGE BASE CONTEXT (grounded findings — cite specifics from here, not generic advice):\n{kb_context}\n"
        )
    if web_context:
        context_block += (
            f"\nLIVE WEB CONTEXT (current guidance — cite specifics from here too):\n{web_context}\n"
        )
    context_block += (
        "\nGround every finding in the above context where relevant — name the actual "
        "components, services, and techniques mentioned rather than generic categories.\n"
    )

    prompt = f"{system_prompt}\n\n{context_block}\n{user_prompt}"

    log.info("Generating security architecture review via Ollama (%s)...", _MODEL)
    threat_model = _ollama(prompt)

    return _strip_preamble(threat_model.strip())


def _strip_preamble(markdown: str) -> str:
    """Drop any meta-commentary paragraph the model wrote before its first heading.

    Some models ignore the "no preamble" instruction and open with a sentence
    like "As a senior security architecture reviewer, I have analyzed...".
    Cut everything before the first markdown heading so output always starts
    with structured content.
    """
    lines = markdown.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#{1,6}\s+\S", line):
            return "\n".join(lines[i:]).strip()
    return markdown


def save_threat_model(
    threat_model: str,
    image_path: Optional[str] = None,
    folder: str = "threat_models"
) -> str:
    """Save the security review markdown and return the file path.

    Args:
        threat_model: Review markdown content
        image_path: Optional path to the source artifact; only embedded inline
            when it's an actual image (PDF/PPTX/text sources aren't embeddable)
        folder: Folder to save in (default: threat_models)

    Returns:
        Path to saved file
    """
    from file_ingest import is_image

    os.makedirs(folder, exist_ok=True)
    
    # Extract title from threat model
    heading = _first_markdown_heading_text(threat_model)
    title = heading if heading else "threat-model"
    
    # Create safe filename
    safe_title = re.sub(r"[^\w\s-]", "", title)
    safe_title = re.sub(r"[-\s]+", "-", safe_title)
    filename = f"{safe_title}.md"
    filepath = os.path.join(folder, filename)
    
    # Embed inline only for actual source images (PDF/PPTX/text aren't embeddable)
    content = threat_model
    if image_path and os.path.isfile(image_path) and is_image(Path(image_path).suffix.lower()):
        content = _inject_architecture_image_markdown(content, image_path)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    log.info("Threat model saved: %s", filepath)
    return filepath


def generate_recommendations(threat_model: str, use_knowledge_base: bool = True) -> str:
    """Generate prioritized security recommendations based on the review.

    Args:
        threat_model: Security architecture review markdown
        use_knowledge_base: Whether to ground recommendations in the KB

    Returns:
        Recommendations markdown
    """
    _ensure_ollama_running()

    kb_context = ""
    if use_knowledge_base:
        try:
            from knowledge_base import retrieve_context, ensure_index_exists
            ensure_index_exists()
            log.info("Retrieving knowledge base context for recommendations...")
            kb_context = retrieve_context(threat_model[:1000], top_k=8, use_rerank=True)
        except Exception as e:
            log.warning("Knowledge base retrieval failed: %s", e)

    rec_prompt_template = get_prompt("recommendation_prompt")
    prompt = rec_prompt_template.format(threat_model=threat_model)
    if kb_context:
        prompt += (
            f"\n\nKNOWLEDGE BASE CONTEXT (relevant security research and best practices):\n{kb_context}\n"
            "\nGround your recommendations in the above context where relevant.\n"
        )

    log.info("Generating security recommendations")
    recommendations = _ollama(prompt)
    
    return recommendations


class ThreatModelState(TypedDict):
    """State for threat modeling conversation."""
    image_path: str
    threat_model: str
    recommendations: str
    saved_path: str


class ChatMessage(TypedDict):
    """Chat message structure."""
    role: str  # "user" or "assistant"
    content: str


def threat_model_chat(
    threat_model: str,
    user_input: str,
    chat_history: List[ChatMessage]
) -> str:
    """Continue a chat conversation about the threat model.
    
    Args:
        threat_model: The STRIDE threat model
        user_input: User's question or comment
        chat_history: Previous messages in conversation
        
    Returns:
        Assistant's response
    """
    _ensure_ollama_running()
    
    system_prompt = get_prompt("chat_system_prompt")

    # Truncate the review before embedding it in the prompt — the full text
    # (with tables, multiple sections) can be 8k+ chars, and re-sending it in
    # full on every turn makes each reply as slow as the original analysis.
    # 3000 chars keeps the headings/findings the model needs without paying
    # that cost every turn.
    _MAX_REVIEW_CHARS = 3000
    truncated_review = threat_model[:_MAX_REVIEW_CHARS]
    if len(threat_model) > _MAX_REVIEW_CHARS:
        truncated_review += "\n[...truncated for chat context...]"

    # Build context from threat model
    context = f"""
Threat Model for Reference:

{truncated_review}

---

Please answer questions about this threat model and provide security guidance.
"""
    
    # Build chat history
    messages = f"System: {system_prompt}\n\n{context}\n\nConversation:\n"
    for msg in chat_history[-5:]:  # Keep last 5 messages for context
        messages += f"{msg['role'].title()}: {msg['content']}\n"
    
    messages += f"\nUser: {user_input}\nAssistant:"
    
    log.info("Chat response for threat model Q&A")
    response = _ollama(messages)
    
    return response.strip()


def _strip_markdown(text: str) -> str:
    """Convert markdown to clean plain text for display."""
    text = re.sub(r"```[a-z]*\n?", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}([^_]+)_{1,3}", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^\s*[-*+]\s+", "  - ", text, flags=re.MULTILINE)
    text = re.sub(r"^[-*_]{3,}\s*$", "─" * 50, text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
