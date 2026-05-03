# evaluation/eval_runner.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evaluation.judge import judge_answer
from rag.llm_config import JUDGE_MODEL
from rag.retriever import format_context, Chunk
from rag.generator import answer_question


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser(description="Run evaluation suite.")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON")
    p.add_argument("--output", default="evaluation/artifacts/results.jsonl", help="Output JSONL path")
    p.add_argument("--mode", choices=["ci", "nightly"], default="ci",
                   help="ci: frozen-context / nightly: end-to-end RAG")
    p.add_argument("--top-k", type=int, default=4, help="Top-k retrieval for nightly mode")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    args = p.parse_args()

    dataset_path = Path(args.dataset)
    out_path = Path(args.output)

    data = load_json_list(dataset_path)
    rows: List[Dict[str, Any]] = []

    for i, ex in enumerate(data, start=1):
        print(f"Evaluating example {data}..")
        ex_id = ex.get("id", f"ex{i}")
        q = ex["question"]
        ideal = ex.get("ideal_answer")

        if args.mode == "ci":
            # Frozen-retrieval regression: pass dataset context as a chunk and
            # let the LLM actually generate the answer. This isolates the
            # generator (the only variable here) from retrieval noise.
            context_text = ex.get("context", "")
            frozen_chunks = (
                [
                    Chunk(
                        id=f"{ex_id}-frozen",
                        text=context_text,
                        source="frozen",
                        chunk_index=0,
                        distance=0.0,
                        metadata={"source": "frozen", "chunk_index": 0},
                    )
                ]
                if context_text
                else []
            )
            rag = answer_question(q, chunks_override=frozen_chunks)
            answer = rag.answer
            context = format_context(rag.chunks)
            retrieved_debug = [
                {
                    "id": c.id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "frozen": True,
                }
                for c in rag.chunks
            ]
        else:
            # nightly: run end-to-end RAG (retrieval+generation)
            rag = answer_question(q, top_k=args.top_k)
            answer = rag.answer
            context = format_context(rag.chunks)
            retrieved_debug = [
                {
                    "id": c.id,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                    "distance": c.distance,
                }
                for c in rag.chunks
            ]

        jr = judge_answer(
            question=q,
            answer=answer,
            context=context,
            ideal_answer=ideal,
            model=args.judge_model,
        )

        row = {
            "id": ex_id,
            "question": q,
            "ideal_answer": ideal,
            "answer": answer,
            "mode": args.mode,
            "scores": {
                "relevance": jr.relevance,
                "correctness": jr.correctness,
                "grounding": jr.grounding,
                "completeness": jr.completeness,
                "reasoning_quality": jr.reasoning_quality,
                "overall": jr.overall,
            },
            "explanation": jr.explanation,
            "retrieval": retrieved_debug,
        }
        rows.append(row)
        print(f"[{i}/{len(data)}] id={ex_id} overall={jr.overall:.3f}")

    write_jsonl(out_path, rows)
    mean_overall = sum(r["scores"]["overall"] for r in rows) / max(len(rows), 1)
    print(f"\nSaved: {out_path}")
    print(f"Mean overall: {mean_overall:.3f}")


if __name__ == "__main__":
    main()
