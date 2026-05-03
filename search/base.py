from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class SearchResult:
    """
    A single result from any search source.

    `score` is source-specific (cosine distance for dense, BM25 score for sparse,
    binary 1.0 for metadata matches). It is NOT comparable across sources -
    fusion is rank-based via RRF.
    """
    id: str
    text: str
    score: float
    source_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class SearchSource(ABC):
    """
    Abstract pluggable search source.

    Every source receives the same query string and returns a ranked list of
    SearchResult, sorted best-first. The orchestrator fuses results via RRF.
    """

    name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        ...
