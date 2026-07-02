"""
Smoke test for the meta-search module.

Run from the repo root:
    python -m search.demo "your query here"

Compares per-source rankings vs RRF-fused ranking. Useful for inspecting
where each source shines and where fusion helps.
"""
from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from search.dense_source import DenseSource
from search.fusion import reciprocal_rank_fusion
from search.meta_search import meta_search
from search.reranker import Reranker
from search.sparse_source import SparseSource


def _print_results(title: str, results, limit: int = 5) -> None:
    print(f"\n=== {title} ===")
    if not results:
        print("  (no results)")
        return
    for rank, r in enumerate(results[:limit], start=1):
        snippet = r.text[:120].replace("\n", " ")
        fused = r.metadata.get("fused_from")
        suffix = f" [fused_from={fused}]" if fused else ""
        print(f"  {rank}. id={r.id}  score={r.score:.4f}{suffix}")
        print(f"     {snippet}...")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="Search query")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    args = parser.parse_args()

    print(f"Query: {args.query!r}")

    dense = DenseSource()
    sparse = SparseSource()

    dense_results = dense.search(args.query, top_k=args.top_k)
    sparse_results = sparse.search(args.query, top_k=args.top_k)

    _print_results("Dense (Chroma)", dense_results, args.top_k)
    _print_results("Sparse (BM25)", sparse_results, args.top_k)

    fused_inline = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        top_k=args.top_k,
    )
    _print_results("RRF Fused (offline)", fused_inline, args.top_k)

    fused_orchestrated = meta_search(
        args.query,
        sources=[dense, sparse],
        per_source_top_k=20,
        final_top_k=20 if args.rerank else args.top_k,
    )
    _print_results("meta_search() (parallel + RRF)", fused_orchestrated, args.top_k)

    if args.rerank:
        reranker = Reranker()
        reranked = reranker.rerank(args.query, fused_orchestrated, top_k=args.top_k)
        _print_results("Reranked (cross-encoder)", reranked, args.top_k)

    return 0


if __name__ == "__main__":
    sys.exit(main())
