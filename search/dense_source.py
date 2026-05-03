from __future__ import annotations

from typing import List

from rag.store import VectorStoreConfig, get_collection
from search.base import SearchResult, SearchSource


class DenseSource(SearchSource):
    """
    Semantic search via Chroma + embeddings.

    Strength: paraphrases, semantic similarity.
    Weakness: rare acronyms, exact identifiers.
    """

    name = "dense"

    def __init__(self, config: VectorStoreConfig | None = None):
        self.config = config or VectorStoreConfig()

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        if not query.strip():
            return []

        collection = get_collection(self.config, create_if_missing=True)
        raw = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (raw.get("ids") or [[]])[0]
        docs = (raw.get("documents") or [[]])[0]
        metas = (raw.get("metadatas") or [[]])[0]
        dists = (raw.get("distances") or [[]])[0]

        return [
            SearchResult(
                id=str(_id),
                text=str(doc),
                score=float(dist) if dist is not None else 0.0,
                source_name=self.name,
                metadata=dict(meta or {}),
            )
            for _id, doc, meta, dist in zip(ids, docs, metas, dists)
        ]
