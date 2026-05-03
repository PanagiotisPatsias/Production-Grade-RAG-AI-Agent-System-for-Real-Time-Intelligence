from __future__ import annotations

from typing import List, Sequence

from sentence_transformers import CrossEncoder

from search.base import SearchResult


DEFAULT_MODEL = "BAAI/bge-reranker-base"


class Reranker:
    """
    Second-stage reranker over candidates from first-stage retrieval.

    Why a cross-encoder:
    - Bi-encoders (used in dense retrieval) embed query and doc independently,
      then compare with cosine. Fast but loses query/doc interaction signal.
    - Cross-encoders feed (query, doc) pairs jointly through the model and
      produce a single relevance score. Slower but much more accurate.

    Standard pattern: retrieve top-N (e.g. 20) cheaply, then rerank to top-K
    (e.g. 5) with the cross-encoder. Bounds the expensive computation.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        candidates: Sequence[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda pair: pair[1],
            reverse=True,
        )[:top_k]

        out: List[SearchResult] = []
        for c, score in ranked:
            merged_meta = dict(c.metadata)
            merged_meta["rerank_score"] = float(score)
            merged_meta["pre_rerank_source"] = c.source_name
            out.append(
                SearchResult(
                    id=c.id,
                    text=c.text,
                    score=float(score),
                    source_name="reranked",
                    metadata=merged_meta,
                )
            )
        return out
