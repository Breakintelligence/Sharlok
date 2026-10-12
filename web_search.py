"""Live web search for Sharlok, ported from brodit's proven fetch pipeline.

Reads topic-keyed URL lists from ``webpages.yml`` (instead of a URL/ folder of
.txt files) and fetches live context by default during analysis, so findings
are grounded in current guidance rather than generic model knowledge alone.

Fetch strategy: aiohttp (fast, concurrent) with automatic fallback to
``requests`` (honors HTTPS_PROXY / REQUESTS_CA_BUNDLE) if aiohttp returns
nothing — same behavior as brodit's app.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Tuple

import requests
import yaml

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    import aiohttp
except ImportError:
    aiohttp = None

log = logging.getLogger("Sharlok.web_search")

_WEBPAGES_PATH = Path(__file__).parent / "webpages.yml"
_HEADERS = {
    "User-Agent": os.environ.get("USER_AGENT", "Sharlok/1.0 (security-architecture-reviewer)"),
    "Accept": "text/html,application/xhtml+xml",
}

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "as", "if", "than", "then", "that",
    "this", "these", "those", "what", "which", "who", "how", "when",
    "where", "why", "architecture", "diagram", "system", "review",
}

_webpages_cache: Optional[dict] = None


def _load_webpages() -> dict:
    global _webpages_cache
    if _webpages_cache is not None:
        return _webpages_cache
    if not _WEBPAGES_PATH.exists():
        log.warning("webpages.yml not found at %s", _WEBPAGES_PATH)
        _webpages_cache = {}
        return _webpages_cache
    with open(_WEBPAGES_PATH, "r", encoding="utf-8") as f:
        _webpages_cache = yaml.safe_load(f) or {}
    return _webpages_cache


def resolve_urls_for_query(query: str) -> List[str]:
    """Pick the right topic URL list(s) from webpages.yml based on query keywords.

    Same matching rule as brodit: topic key checked as a case-insensitive word
    match against the query; "other" is the fallback when nothing matches.
    """
    topics = _load_webpages()
    if not topics:
        return []

    query_lower = query.lower()
    matched: List[str] = []
    for topic, urls in topics.items():
        if topic == "other":
            continue
        if re.search(rf"\b{re.escape(topic)}\b", query_lower):
            matched.extend(urls or [])
            log.info("Web source matched: %s", topic)

    if not matched:
        other = topics.get("other")
        if other:
            log.info("No specific topic matched query; using 'other'.")
            matched.extend(other)
        else:
            for urls in topics.values():
                matched.extend(urls or [])

    seen, deduped = set(), []
    for u in matched:
        if u not in seen:
            seen.add(u)
            deduped.append(u)
    return deduped


def _parse_html(html: str) -> str:
    if BeautifulSoup is None:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


async def _fetch_one_aio(session: "aiohttp.ClientSession", url: str, timeout: int = 15) -> Optional[SimpleNamespace]:
    try:
        async with session.get(url, headers=_HEADERS, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            resp.raise_for_status()
            html = await resp.text()
            return SimpleNamespace(page_content=_parse_html(html), metadata={"source": url})
    except Exception as exc:
        log.warning("aiohttp fetch failed for %s: %s", url, exc)
        return None


async def _fetch_all_aio(url_list: List[str]) -> List[SimpleNamespace]:
    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(_fetch_one_aio(session, u)) for u in url_list]
        results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r is not None]


def _sync_fetch_one(url: str, session: requests.Session, timeout: int = 15) -> Optional[SimpleNamespace]:
    try:
        resp = session.get(url, headers=_HEADERS, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        return SimpleNamespace(page_content=_parse_html(resp.text), metadata={"source": url})
    except Exception as exc:
        log.warning("requests fetch failed for %s: %s", url, exc)
        return None


def _sync_fetch(url_list: List[str]) -> List[SimpleNamespace]:
    if not url_list:
        return []
    docs: List[SimpleNamespace] = []
    workers = min(10, len(url_list))
    with requests.Session() as session:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_sync_fetch_one, u, session): u for u in url_list}
            for fut in concurrent.futures.as_completed(futures):
                res = fut.result()
                if res is not None:
                    docs.append(res)
    return docs


def fetch_urls(url_list: List[str], timeout_s: int = 20) -> List[SimpleNamespace]:
    """Fetch a list of URLs with aiohttp, falling back to requests on total failure."""
    if not url_list:
        return []

    if aiohttp is not None:
        try:
            docs = asyncio.run(asyncio.wait_for(_fetch_all_aio(url_list), timeout=timeout_s))
        except Exception as exc:
            log.warning("aiohttp run failed (%s) — falling back to requests", exc)
            docs = []
        if docs:
            log.info("Web fetch: %d/%d URLs succeeded via aiohttp", len(docs), len(url_list))
            return docs
        log.warning("aiohttp fetched 0/%d URLs — retrying with requests", len(url_list))

    return _sync_fetch(url_list)


def _chunk(docs: List[SimpleNamespace], size: int = 3000, overlap: int = 400) -> List[SimpleNamespace]:
    chunks: List[SimpleNamespace] = []
    step = size - overlap
    for doc in docs:
        text = doc.page_content or ""
        src = getattr(doc, "metadata", {}).get("source", "")
        for i in range(0, max(1, len(text)), step):
            chunk_text = text[i: i + size]
            if chunk_text.strip():
                chunks.append(SimpleNamespace(page_content=chunk_text, metadata={"source": src}))
    return chunks


def _query_tokens(query: str) -> set:
    raw_tokens = set(re.findall(r"\w+", query.lower()))
    tokens = raw_tokens - _STOP_WORDS
    tokens = {t for t in tokens if not re.fullmatch(r"[\dv]+", t)}
    return tokens or raw_tokens


def _bm25_score(tokens: set, text: str) -> float:
    words = re.findall(r"\w+", text.lower())
    if not words:
        return 0.0
    hit = sum(1 for w in words if w in tokens)
    return hit / (1 + len(words) / 500)


def fetch_web_context(query: str, top_k: int = 6, timeout_s: int = 20) -> Tuple[str, int]:
    """Fetch live web context for *query*, ranked by keyword overlap.

    Returns ``(context_string, url_count)``. Non-fatal on any fetch failure —
    returns an empty context string so callers can proceed with FAISS-only
    context instead of blocking the whole review.
    """
    url_list = resolve_urls_for_query(query)
    if not url_list:
        return "", 0

    try:
        docs = fetch_urls(url_list, timeout_s=timeout_s)
    except Exception as exc:
        log.warning("Web context fetch failed entirely: %s", exc)
        return "", 0

    if not docs:
        return "", 0

    chunks = _chunk(docs)
    tokens = _query_tokens(query)
    scored = [(c, _bm25_score(tokens, c.page_content)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [c for c, s in scored[:top_k] if s > 0] or [c for c, _ in scored[:top_k]]

    parts = []
    for i, c in enumerate(top, 1):
        source = c.metadata.get("source", "unknown")
        parts.append(f"[Web source {i}: {source}]\n{c.page_content[:1200]}")

    return "\n\n---\n\n".join(parts), len(docs)
