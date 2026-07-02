# evaluation/ablation_runner.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from dotenv import load_dotenv
from evaluation.judge import judge_answer
from rag.generator import answer_question
from rag.ingest import ingest_pdf_dir
from rag.llm_config import JUDGE_MODEL
from rag.store import VectorStoreConfig


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def main():
    p = argparse.ArgumentParser(description="Run ablations for RAG parameters.")
    p.add_argument("--dataset", required=True, help="Dataset JSON with questions (use ci_golden.json or nightly suite)")
    p.add_argument("--pdf-dir", default="data", help="Directory of PDFs to ingest")
    p.add_argument("--topk", default="2,4,8", help="Comma-separated top_k values")
    p.add_argument("--chunk-sizes", default="800,1000,1500", help="Comma-separated chunk_size values")
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--output", default="evaluation/artifacts/ablation_results.json")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    args = p.parse_args()
    load_dotenv()
    dataset = load_json_list(Path(args.dataset))
    questions = [ex["question"] for ex in dataset]

    topk_values = [int(x.strip()) for x in args.topk.split(",") if x.strip()]
    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",") if x.strip()]

    results: List[Dict[str, Any]] = []

    cfg = VectorStoreConfig()

    # Ablation: chunk_size (rebuild index each time), keep top_k fixed at 4
    for cs in chunk_sizes:
        ingest_pdf_dir(Path(args.pdf_dir), reset=True, chunk_size=cs, chunk_overlap=args.chunk_overlap, config=cfg)

        scores = []
        for q in questions:
            rag = answer_question(q, top_k=4)
            jr = judge_answer(
                question=q,
                answer=rag.answer,
                context="\n\n".join([f"[{i}] {c.text}" for i, c in enumerate(rag.chunks, start=1)]),
                ideal_answer=None,  # could use ideal if present; keep None for general benchmark
                model=args.judge_model,
            )
            scores.append(jr.overall)

        results.append(
            {
                "ablation": "chunk_size",
                "chunk_size": cs,
                "chunk_overlap": args.chunk_overlap,
                "top_k": 4,
                "mean_overall": sum(scores) / len(scores),
                "n": len(scores),
            }
        )
        print(f"[chunk_size={cs}] mean_overall={results[-1]['mean_overall']:.3f}")

    # Ablation: top_k (no rebuild needed), use chunk_size=1000 baseline
    ingest_pdf_dir(Path(args.pdf_dir), reset=True, chunk_size=1000, chunk_overlap=args.chunk_overlap, config=cfg)

    for k in topk_values:
        scores = []
        for q in questions:
            rag = answer_question(q, top_k=k)
            jr = judge_answer(
                question=q,
                answer=rag.answer,
                context="\n\n".join([f"[{i}] {c.text}" for i, c in enumerate(rag.chunks, start=1)]),
                ideal_answer=None,
                model=args.judge_model,
            )
            scores.append(jr.overall)

        results.append(
            {
                "ablation": "top_k",
                "chunk_size": 1000,
                "chunk_overlap": args.chunk_overlap,
                "top_k": k,
                "mean_overall": sum(scores) / len(scores),
                "n": len(scores),
            }
        )
        print(f"[top_k={k}] mean_overall={results[-1]['mean_overall']:.3f}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved ablation results to: {out}")


if __name__ == "__main__":
    main()
