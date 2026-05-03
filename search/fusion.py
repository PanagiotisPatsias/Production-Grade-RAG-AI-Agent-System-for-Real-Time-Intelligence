from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Sequence

from search.base import SearchResult


RRF_K = 60  # Original paper's recommended constant


def reciprocal_rank_fusion(
    result_lists: Sequence[List[SearchResult]],
    *,
    k: int = RRF_K,
    top_k: int = 10,
) -> List[SearchResult]:
    """
    Fuse multiple ranked result lists into one ranking via Reciprocal Rank Fusion.

        RRF(d) = sum over each list: 1 / (k + rank_in_list)

    RRF is score-agnostic: it ignores raw scores and uses only ranks. This avoids
    the calibration problem of combining heterogeneous score scales (cosine
    distance, BM25, binary matches).

    When the same document id appears in multiple lists, we keep the SearchResult
    from the first list it appeared in (for text/metadata) and record all source
    names that surfaced it in metadata['fused_from'].
    """
    if not result_lists:
        return []

    rrf_scores: Dict[str, float] = defaultdict(float)
    first_seen: Dict[str, SearchResult] = {}
    fused_from: Dict[str, List[str]] = defaultdict(list)

    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            rrf_scores[r.id] += 1.0 / (k + rank)
            fused_from[r.id].append(r.source_name)
            if r.id not in first_seen:
                first_seen[r.id] = r

    ranked_ids = sorted(rrf_scores.keys(), key=lambda i: rrf_scores[i], reverse=True)

    out: List[SearchResult] = []
    for _id in ranked_ids[:top_k]:
        original = first_seen[_id]
        merged_meta = dict(original.metadata)
        merged_meta["fused_from"] = fused_from[_id]
        merged_meta["rrf_score"] = rrf_scores[_id]
        out.append(
            SearchResult(
                id=original.id,
                text=original.text,
                score=rrf_scores[_id],
                source_name="fused",
                metadata=merged_meta,
            )
        )
    return out
