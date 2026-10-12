"""Knowledge base management for Sharlok.

Provides:
- FAISS index loading and management (cached in-process after first load)
- Hybrid semantic + BM25 keyword retrieval (same blended strategy as
  brodit's ``retrieve_blog_posts`` — pure semantic search alone tends to
  drift toward generic, topically-adjacent chunks and miss exact tool/
  technique names; BM25 catches those verbatim matches)
- Cross-encoder reranking on the blended candidate set
- Embedding generation using Ollama (/api/embed), same as brodit's app.py

Retrieval reads ``Knowledgebase/.faiss_store``. ``build_index()`` creates or
incrementally updates it from Knowledgebase/PDF, Knowledgebase/Repos,
Knowledgebase/Tools and the URLs in webpages.yml.
"""

import hashlib
import logging
import os
import pickle
import re
import threading
from types import SimpleNamespace
from typing import List, Optional, Tuple

# faiss-cpu and PyTorch/sentence-transformers each bundle libomp.dylib on macOS,
# which causes OMP err #15 (duplicate runtime). Must be set before either
# library is imported (same fix as brodit's app.py).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import numpy as np
except ImportError:
    np = None

import requests

from file_ingest import _extract_pdf_text
from web_search import _chunk, _load_webpages, fetch_urls

log = logging.getLogger("Sharlok.knowledge_base")

_KB_ROOT = os.path.join(os.path.dirname(__file__), "Knowledgebase")
_FAISS_DIR = os.path.join(_KB_ROOT, ".faiss_store")
_FAISS_INDEX = os.path.join(_FAISS_DIR, "index.faiss")
_FAISS_META = os.path.join(_FAISS_DIR, "metadata.pkl")

_PDF_DIR = os.path.join(_KB_ROOT, "PDF")
_CODE_DIRS = [os.path.join(_KB_ROOT, "Repos"), os.path.join(_KB_ROOT, "Tools")]
_CODE_EXTS = {
    ".py", ".js", ".ts", ".go", ".java", ".cs", ".cpp", ".c", ".sh",
    ".tf", ".bicep", ".yaml", ".yml", ".json", ".md", ".txt",
}
_EMBED_BATCH = 32
_build_lock = threading.Lock()

_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
_EMBED_MODEL = os.environ.get("Sharlok_EMBED_MODEL", "nomic-embed-text")
_RERANK_MODEL = os.environ.get("Sharlok_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "as", "if", "than", "then", "that",
    "this", "these", "those", "what", "which", "who", "how", "when",
    "where", "why", "architecture", "diagram", "system", "review",
}

# Cached in-process so retrieval does not re-read the 190MB+ store per call;
# build_index() refreshes these after writing the store.
_reranker = None
_reranker_lock = threading.Lock()
_index_cache = None
_metadata_cache: Optional[dict] = None
_index_load_attempted = False


def _get_reranker():
    """Lazily initialize CrossEncoder for reranking."""
    global _reranker
    if _reranker is not None:
        return _reranker

    with _reranker_lock:
        if _reranker is not None:
            return _reranker
        try:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(_RERANK_MODEL)
            log.info("Reranker loaded: %s", _RERANK_MODEL)
        except Exception as exc:
            log.warning("Reranker unavailable (%s). Falling back to semantic score.", exc)

    return _reranker


def generate_embedding(text: str) -> Optional["np.ndarray"]:
    """Generate embedding for text using Ollama's /api/embed (same as brodit)."""
    if not text or not text.strip():
        return None

    try:
        url = _OLLAMA_HOST.rstrip("/") + "/api/embed"
        payload = {"model": _EMBED_MODEL, "input": text}
        resp = requests.post(url, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        if embeddings:
            if np is None:
                return embeddings[0]
            return np.array(embeddings[0], dtype=np.float32)
    except Exception as exc:
        log.error("Embedding generation failed: %s", exc)

    return None


def load_faiss_index() -> Tuple[Optional[object], Optional[dict]]:
    """Load FAISS index and metadata from disk, cached after the first call.

    The on-disk metadata schema is a dict of parallel arrays:
      metadata["texts"][i]   -> chunk text for FAISS vector i
      metadata["sources"][i] -> source file/URL for FAISS vector i
    """
    global _index_cache, _metadata_cache, _index_load_attempted

    if _index_load_attempted:
        return _index_cache, _metadata_cache
    _index_load_attempted = True

    if not os.path.exists(_FAISS_INDEX) or not os.path.exists(_FAISS_META):
        log.warning(
            "FAISS index not found at %s. Run `main.py --build-index` to create it "
            "and enable knowledge-base-grounded reviews.",
            _FAISS_DIR,
        )
        return None, None

    try:
        import faiss
        _index_cache = faiss.read_index(_FAISS_INDEX)
        with open(_FAISS_META, "rb") as f:
            _metadata_cache = pickle.load(f)
        log.info("Loaded FAISS index with %d vectors (cached in-process)", _index_cache.ntotal)
        return _index_cache, _metadata_cache
    except Exception as exc:
        log.error("Failed to load FAISS index: %s", exc)
        return None, None


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


def retrieve_context(
    query: str,
    top_k: int = 10,
    use_rerank: bool = True
) -> str:
    """Retrieve architecture-review context using hybrid semantic + BM25 retrieval.

    Semantic search alone tends to surface generic, topically-adjacent chunks
    and can miss exact tool/technique names (e.g. "mTLS", "IMDSv2",
    "AgentCore"). This blends FAISS semantic candidates with a BM25 keyword
    pass over the same corpus, then optionally reranks the union with a
    cross-encoder — same strategy as brodit's ``retrieve_blog_posts``.

    Args:
        query: Security topic or architecture aspect
        top_k: Number of top results to retrieve
        use_rerank: Whether to use cross-encoder reranking

    Returns:
        Formatted context string for review generation
    """
    index, metadata = load_faiss_index()

    if index is None or metadata is None:
        log.warning("No knowledge base available for query: %s", query)
        return ""

    texts = metadata.get("texts") or []
    sources = metadata.get("sources") or []
    if not texts:
        log.warning("FAISS metadata has no 'texts' entries")
        return ""

    over_fetch = max(top_k * 4, 24)

    # (1) Semantic candidates via FAISS
    semantic_candidates: dict = {}
    query_embedding = generate_embedding(query)
    if query_embedding is not None:
        qvec = query_embedding.reshape(1, -1).astype(np.float32)
        k = min(over_fetch, index.ntotal)
        distances, indices = index.search(qvec, k)
        n = len(indices[0])
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0])):
            if 0 <= idx < len(texts):
                semantic_candidates[idx] = (n - rank) / n if n else 0.0
    else:
        log.warning("Failed to generate query embedding; falling back to BM25 only")

    # (2) BM25 keyword candidates over the same corpus
    tokens = _query_tokens(query)
    bm25_scored = [(i, _bm25_score(tokens, t)) for i, t in enumerate(texts)] if tokens else []
    bm25_scored.sort(key=lambda x: x[1], reverse=True)
    bm25_top = {i: s for i, s in bm25_scored[:over_fetch] if s > 0}

    # (3) Union, blended score (BM25 weighted higher — it catches exact terms
    # semantic search paraphrases away)
    all_idx = set(semantic_candidates) | set(bm25_top)
    if not all_idx:
        return ""

    blended = []
    for idx in all_idx:
        sem = semantic_candidates.get(idx, 0.0)
        bm25 = bm25_top.get(idx, 0.0) or _bm25_score(tokens, texts[idx])
        blended.append((idx, sem + 2.0 * bm25))
    blended.sort(key=lambda x: x[1], reverse=True)

    candidates = [
        {
            "content": texts[idx],
            "source": sources[idx] if idx < len(sources) else "unknown",
            "score": float(score),
        }
        for idx, score in blended[:over_fetch]
    ]

    if use_rerank and candidates:
        candidates = rerank_candidates(query, candidates, top_k)
    else:
        candidates = candidates[:top_k]

    if not candidates:
        return ""

    context_parts = [f"[Query focus: {query}]\n"]
    for i, cand in enumerate(candidates, 1):
        source = cand.get("source", "unknown")
        content = cand.get("content", "")[:600]
        context_parts.append(f"[Source {i}: {source}]\n{content}\n")

    return "\n---\n".join(context_parts)


def rerank_candidates(query: str, candidates: List[dict], top_k: int) -> List[dict]:
    """Rerank candidates using cross-encoder."""
    ranker = _get_reranker()

    if ranker is None or not candidates:
        return candidates[:top_k]

    try:
        pairs = [(query, cand["content"][:512]) for cand in candidates]
        scores = ranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        log.info("Reranked: %d → top_%d selected", len(candidates), top_k)
        return [cand for cand, _ in reranked[:top_k]]
    except Exception as exc:
        log.warning("Reranking failed (%s); using blended score", exc)
        return candidates[:top_k]


def get_threat_modeling_context(architecture_description: str) -> str:
    """Get security context from the knowledge base for a given architecture.

    Not STRIDE-specific or AI-specific — pulls whatever is most relevant to
    the given architecture (cloud, on-prem, mobile, AI/ML, etc.) using hybrid
    semantic + BM25 retrieval so exact tool/technique names in the corpus are
    not lost to generic paraphrase matches.
    """
    query = f"security review and mitigations for: {architecture_description[:500]}"
    return retrieve_context(query, top_k=8, use_rerank=True)


def ensure_index_exists() -> bool:
    """Report whether the FAISS store exists. Does not build it (see build_index_if_missing)."""
    if os.path.exists(_FAISS_INDEX) and os.path.exists(_FAISS_META):
        log.info("Existing FAISS index found, reusing it as-is.")
        return True
    log.warning(
        "No FAISS index at %s — knowledge-base grounding will be skipped.",
        _FAISS_DIR,
    )
    return False


# ── Index build (``main.py --build-index`` and auto-build when the store is missing) ──

def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _collect_files(folders: List[str], extensions: set) -> List[str]:
    paths: List[str] = []
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for root, _, files in os.walk(folder):
            paths.extend(
                os.path.join(root, name)
                for name in files
                if os.path.splitext(name)[1].lower() in extensions
            )
    return sorted(paths)


def _load_file_doc(path: str) -> Optional[Tuple[SimpleNamespace, str]]:
    """Return (doc, sha256 fingerprint) for a PDF or text/code file, or None if unusable."""
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
        if path.lower().endswith(".pdf"):
            text = _extract_pdf_text(path)
        else:
            text = raw.decode("utf-8", errors="ignore")
    except Exception as exc:
        log.warning("Skipping %s: %s", path, exc)
        return None
    if not text.strip():
        return None
    return SimpleNamespace(page_content=text, metadata={"source": path}), hashlib.sha256(raw).hexdigest()


def _webpage_urls() -> List[str]:
    urls = [url for topic_urls in _load_webpages().values() for url in (topic_urls or [])]
    return list(dict.fromkeys(urls))


def _collect_new_docs(state: dict) -> Tuple[List[SimpleNamespace], dict]:
    """Return docs for new/changed files and not-yet-indexed URLs, plus new file fingerprints."""
    docs: List[SimpleNamespace] = []
    fingerprints: dict = {}
    for path in _collect_files([_PDF_DIR], {".pdf"}) + _collect_files(_CODE_DIRS, _CODE_EXTS):
        loaded = _load_file_doc(path)
        if loaded is None:
            continue
        doc, fingerprint = loaded
        if state["source_fingerprints"].get(path) != fingerprint:
            docs.append(doc)
            fingerprints[path] = fingerprint
    known_sources = set(state["sources"])
    new_urls = [url for url in _webpage_urls() if url not in known_sources]
    if new_urls:
        docs.extend(fetch_urls(new_urls))
    return docs, fingerprints


def _load_store() -> Tuple[Optional[object], dict]:
    """Return (index, state) from disk, or (None, empty state) when no store exists."""
    empty = {"texts": [], "sources": [], "content_hashes": set(), "source_fingerprints": {}}
    if not (os.path.exists(_FAISS_INDEX) and os.path.exists(_FAISS_META)):
        return None, empty

    import faiss
    with open(_FAISS_META, "rb") as fh:
        meta = pickle.load(fh)
    index = faiss.read_index(_FAISS_INDEX)
    texts = list(meta.get("texts", []))
    if index.ntotal != len(texts):
        raise RuntimeError(
            f"FAISS index has {index.ntotal} vectors but metadata has {len(texts)} texts. "
            f"Delete {_FAISS_DIR} and re-run the build."
        )
    if "content_hashes" in meta:
        content_hashes = set(meta["content_hashes"])
    else:
        content_hashes = {_text_hash(text) for text in texts}
    state = {
        "texts": texts,
        "sources": list(meta.get("sources", [])),
        "content_hashes": content_hashes,
        "source_fingerprints": dict(meta.get("source_fingerprints", {})),
    }
    return index, state


def _request_embeddings(batch: List[str]) -> List[list]:
    url = _OLLAMA_HOST.rstrip("/") + "/api/embed"
    resp = requests.post(url, json={"model": _EMBED_MODEL, "input": batch}, timeout=120)
    resp.raise_for_status()
    return resp.json()["embeddings"]


def _embed_texts(texts: List[str]) -> List[Optional[list]]:
    """Embed texts in batches; items that fail even individually come back as None."""
    vectors: List[Optional[list]] = []
    for start in range(0, len(texts), _EMBED_BATCH):
        batch = texts[start:start + _EMBED_BATCH]
        try:
            vectors.extend(_request_embeddings(batch))
        except Exception as exc:
            log.warning("Embedding batch failed (%s); retrying items individually.", exc)
            for text in batch:
                try:
                    vectors.extend(_request_embeddings([text]))
                except Exception as item_exc:
                    log.warning("Skipping chunk that failed to embed (%s)", item_exc)
                    vectors.append(None)
    return vectors


def _normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def _unseen_chunks(chunks: List[SimpleNamespace], seen: set) -> List[SimpleNamespace]:
    unique: List[SimpleNamespace] = []
    batch_hashes: set = set()
    for chunk in chunks:
        digest = _text_hash(chunk.page_content)
        if digest in seen or digest in batch_hashes:
            continue
        batch_hashes.add(digest)
        unique.append(chunk)
    return unique


def _save_store(index, state: dict) -> None:
    import faiss
    os.makedirs(_FAISS_DIR, exist_ok=True)
    faiss.write_index(index, _FAISS_INDEX + ".tmp")
    os.replace(_FAISS_INDEX + ".tmp", _FAISS_INDEX)
    with open(_FAISS_META + ".tmp", "wb") as fh:
        pickle.dump(
            {
                "texts": state["texts"],
                "sources": state["sources"],
                "content_hashes": list(state["content_hashes"]),
                "source_fingerprints": state["source_fingerprints"],
            },
            fh,
        )
    os.replace(_FAISS_META + ".tmp", _FAISS_META)


def _refresh_cache(index, state: dict) -> None:
    global _index_cache, _metadata_cache, _index_load_attempted
    _index_cache = index
    _metadata_cache = {"texts": state["texts"], "sources": state["sources"]}
    _index_load_attempted = True


def build_index() -> int:
    """Create or incrementally update the FAISS store. Returns the number of chunks added.

    Changed files and unseen URLs are embedded; everything else is skipped.
    """
    import faiss

    with _build_lock:
        index, state = _load_store()
        docs, fingerprints = _collect_new_docs(state)
        chunks = _unseen_chunks(_chunk(docs), state["content_hashes"])
        vectors = _embed_texts([chunk.page_content for chunk in chunks])
        kept = [(chunk, vector) for chunk, vector in zip(chunks, vectors) if vector is not None]

        if chunks and not kept:
            raise RuntimeError(f"Embedding failed for every chunk. Is Ollama running with {_EMBED_MODEL}?")
        if not kept and not fingerprints:
            log.info("Knowledge store is up to date (%d chunks).", len(state["texts"]))
            return 0

        if kept:
            matrix = _normalize(np.array([vector for _, vector in kept], dtype="float32"))
            if index is None:
                index = faiss.IndexFlatIP(matrix.shape[1])
            index.add(matrix)
            state["texts"].extend(chunk.page_content for chunk, _ in kept)
            state["sources"].extend(chunk.metadata["source"] for chunk, _ in kept)
            state["content_hashes"].update(_text_hash(chunk.page_content) for chunk, _ in kept)

        state["source_fingerprints"].update(fingerprints)
        _save_store(index, state)
        _refresh_cache(index, state)
        log.info("Indexed %d new chunk(s); store now holds %d.", len(kept), len(state["texts"]))
        return len(kept)


def build_index_if_missing() -> bool:
    """Build the store only when it is absent. Returns True if a store exists afterwards."""
    if os.path.exists(_FAISS_INDEX) and os.path.exists(_FAISS_META):
        return True
    log.warning("No FAISS index at %s. Building it from knowledge sources.", _FAISS_DIR)
    build_index()
    return os.path.exists(_FAISS_INDEX)
