from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Sequence

from search.base import SearchResult, SearchSource
from search.fusion import RRF_K, reciprocal_rank_fusion


def meta_search(
    query: str,
    sources: Sequence[SearchSource],
    *,
    per_source_top_k: int = 20,
    final_top_k: int = 5,
    rrf_k: int = RRF_K,
) -> List[SearchResult]:
    """
    Meta-search across multiple heterogeneous sources, fused via RRF.

    Sources are queried in parallel with a thread pool (sources are I/O-bound
    in our case: Chroma network call, in-memory BM25). This bounds total
    latency to ~max(source_latency) instead of sum.
    """
    if not query.strip() or not sources:
        return []

    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        result_lists = list(
            executor.map(lambda s: s.search(query, top_k=per_source_top_k), sources)
        )

    return reciprocal_rank_fusion(result_lists, k=rrf_k, top_k=final_top_k)


def default_sources() -> List[SearchSource]:
    """
    Convenience: instantiate the standard hybrid sources (dense + sparse).
    Heavy to construct (BM25 build), so cache and reuse in production.
    """
    from search.dense_source import DenseSource
    from search.sparse_source import SparseSource

    return [DenseSource(), SparseSource()]
