from __future__ import annotations

import re
from typing import List

from rank_bm25 import BM25Okapi

from rag.store import VectorStoreConfig, get_collection
from search.base import SearchResult, SearchSource


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class SparseSource(SearchSource):
    """
    Lexical statistical search via BM25.

    Strength: rare terms, acronyms (TCFD, GDPR), exact word overlap.
    Weakness: paraphrases, synonyms.

    Index is built in-memory at construction time by pulling all chunks from Chroma.
    Suitable for the project's scale; in production, you would persist the index.
    """

    name = "sparse"

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()
        self._build_index()

    def _build_index(self) -> None:
        collection = get_collection(self.config, create_if_missing=True)
        raw = collection.get(include=["documents", "metadatas"])

        self._ids: List[str] = list(raw.get("ids") or [])
        self._docs: List[str] = [str(d) for d in (raw.get("documents") or [])]
        self._metas = [dict(m or {}) for m in (raw.get("metadatas") or [])]

        tokenized = [_tokenize(d) for d in self._docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        if not query.strip() or self._bm25 is None or not self._docs:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                id=self._ids[i],
                text=self._docs[i],
                score=float(scores[i]),
                source_name=self.name,
                metadata=self._metas[i],
            )
            for i in ranked
            if scores[i] > 0
        ]
