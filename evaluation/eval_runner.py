import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from .judge_llm import evaluate_answer_with_judge_llm


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """
    Load evaluation dataset. Expects a JSON file with a list of items, each like:

    {
      "id": "ex1",
      "question": "...",
      "ideal_answer": "...",
      "context": "...",
      "rag_answer": "...",              # (filled by generate_rag_answers.py)
      "retrieved_chunks": ["...", ...]  # optional, for debugging
    }
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list, got: {type(data)}")

    return data


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-as-a-judge evaluation on a RAG dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSON dataset (e.g. rag_dataset.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.jsonl",
        help="Where to store evaluation results (JSONL).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use as the judge.",
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    items = load_dataset(dataset_path)
    print(f"Loaded {len(items)} examples from {dataset_path}")

    with output_path.open("w", encoding="utf-8") as out_f:
        for i, ex in enumerate(items, start=1):
            q = ex["question"]
            rag_answer = ex.get("rag_answer") or ex.get("golden_rag_answer", "")

            context = ex.get("context", "")
            ideal = ex.get("ideal_answer")

            if not rag_answer:
                print(f"[{i}] WARNING: missing rag_answer for id={ex.get('id')}, skipping.")
                continue

            # You could also choose to pass retrieved_chunks as part of the context if desired.
            scores = evaluate_answer_with_judge_llm(
                question=q,
                rag_answer=rag_answer,
                context=context,
                ideal_answer=ideal,
                model=args.model,
            )

            row = {
                "id": ex.get("id", i),
                "question": q,
                "rag_answer": rag_answer,
                "context": context,
                "ideal_answer": ideal,
                "scores": scores.to_dict(),
            }

            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[{i}/{len(items)}] id={row['id']} "
                f"overall={scores.overall:.3f} "
                f"(rel={scores.relevance:.3f}, corr={scores.correctness:.3f}, "
                f"ground={scores.grounding:.3f})"
            )

    print(f"Evaluation results saved to: {output_path}")


if __name__ == "__main__":
    main()
