# rag/retriever.py
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Dict, List, Optional

from rag.store import VectorStoreConfig, get_collection
from search.dense_source import DenseSource
from search.meta_search import meta_search
from search.reranker import Reranker
from search.sparse_source import SparseSource


@dataclass(frozen=True)
class Chunk:
    """
    A retrieved chunk from the 2-stage hybrid retriever.
    """
    id: str
    text: str
    source: Optional[str]
    chunk_index: Optional[int]
    distance: Optional[float]
    metadata: Dict[str, Any]


_LOCK = Lock()
_DENSE: Optional[DenseSource] = None
_SPARSE: Optional[SparseSource] = None
_SPARSE_BUILT_FOR_COUNT: int = -1
_RERANKER: Optional[Reranker] = None


def _get_hybrid_sources(config: VectorStoreConfig):
    """
    Build dense + sparse sources once and reuse across calls.

    BM25 is built in-memory from the Chroma collection at construction time;
    rebuilt only when the collection size changes (e.g. after ingest).
    """
    global _DENSE, _SPARSE, _SPARSE_BUILT_FOR_COUNT

    with _LOCK:
        if _DENSE is None:
            _DENSE = DenseSource(config=config)

        current_count = get_collection(config, create_if_missing=True).count()
        if _SPARSE is None or current_count != _SPARSE_BUILT_FOR_COUNT:
            _SPARSE = SparseSource(config=config)
            _SPARSE_BUILT_FOR_COUNT = current_count

        return [_DENSE, _SPARSE]


def _get_reranker() -> Reranker:
    """
    Cross-encoder is heavy to instantiate (downloads weights). Build once.
    """
    global _RERANKER
    with _LOCK:
        if _RERANKER is None:
            _RERANKER = Reranker()
        return _RERANKER


def retrieve(
    query: str,
    *,
    top_k: int = 4,
    candidate_pool: int = 20,
    config: VectorStoreConfig = VectorStoreConfig(),
) -> List[Chunk]:
    """
    Two-stage hybrid retrieval:
      1. Dense (Chroma) + Sparse (BM25) fused via RRF -> candidate_pool results
      2. Cross-encoder rerank -> top_k

    `candidate_pool` should be >= top_k. Larger pools give the reranker more to
    work with at the cost of latency.
    """
    if not query or not query.strip():
        return []

    pool_size = max(candidate_pool, top_k)

    sources = _get_hybrid_sources(config)
    fused = meta_search(
        query,
        sources=sources,
        per_source_top_k=max(20, pool_size * 2),
        final_top_k=pool_size,
    )

    if not fused:
        return []

    reranker = _get_reranker()
    reranked = reranker.rerank(query, fused, top_k=top_k)

    chunks: List[Chunk] = []
    for r in reranked:
        meta = dict(r.metadata or {})
        chunks.append(
            Chunk(
                id=str(r.id),
                text=str(r.text),
                source=meta.get("source"),
                chunk_index=meta.get("chunk_index"),
                distance=float(r.score) if r.score is not None else None,
                metadata=meta,
            )
        )

    return chunks


def format_context(chunks: List[Chunk]) -> str:
    """
    Formats retrieved chunks into a context block with stable numeric citations [1], [2], ...
    """
    if not chunks:
        return "No relevant context found."

    lines: List[str] = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(f"[{i}] {ch.text}")
    return "\n\n".join(lines)
