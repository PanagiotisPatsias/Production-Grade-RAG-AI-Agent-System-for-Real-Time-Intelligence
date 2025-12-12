import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from .generate_rag_answers import generate_rag_answer


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Expected a JSON list")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Create a small curated CI 'golden' dataset by running the RAG system."
    )
    parser.add_argument("--input", required=True, help="Input JSON dataset (synthetic_dataset.json)")
    parser.add_argument("--output", default="evaluation/ci_golden.json", help="Output golden CI dataset")
    parser.add_argument("--num", type=int, default=5, help="How many examples to take from input")
    parser.add_argument("--top-k", type=int, default=4, help="How many chunks to retrieve")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    data = load_json_list(input_path)
    picked = data[: args.num]

    golden: List[Dict[str, Any]] = []

    for i, ex in enumerate(picked, start=1):
        q = ex["question"]
        ideal = ex.get("ideal_answer")
        ctx = ex.get("context", "")

        rag = generate_rag_answer(question=q, top_k=args.top_k)

        golden.append(
            {
                "id": ex.get("id", f"ci-{i}"),
                "question": q,
                "context": ctx,
                "ideal_answer": ideal,
                # golden baseline output:
                "golden_rag_answer": rag["rag_answer"],
                # keep retrieval for debugging:
                "retrieved_chunks": rag["retrieved_chunks"],
            }
        )

        print(f"[{i}/{len(picked)}] created golden case id={golden[-1]['id']}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(golden, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved golden CI dataset to: {output_path}")


if __name__ == "__main__":
    main()
