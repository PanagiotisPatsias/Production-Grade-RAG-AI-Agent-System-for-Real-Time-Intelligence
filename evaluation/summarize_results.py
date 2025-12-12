import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import statistics


def load_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(json.loads(line))
    return results


def summarize_scores(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean / min / max for each metric across all examples.
    """
    metrics = ["relevance", "correctness", "grounding", "completeness", "reasoning_quality", "overall"]
    values: Dict[str, List[float]] = {m: [] for m in metrics}

    for item in results:
        scores = item.get("scores", {})
        for m in metrics:
            if m in scores and isinstance(scores[m], (int, float)):
                values[m].append(float(scores[m]))

    summary: Dict[str, Dict[str, float]] = {}
    for m in metrics:
        if not values[m]:
            continue
        summary[m] = {
            "mean": statistics.mean(values[m]),
            "min": min(values[m]),
            "max": max(values[m]),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Summarize LLM evaluation results.")
    parser.add_argument(
        "--results",
        type=str,
        default="evaluation_results.jsonl",
        help="Path to JSONL file with evaluation results.",
    )
    args = parser.parse_args()

    path = Path(args.results)
    results = load_results(path)

    print(f"Loaded {len(results)} evaluation records from {path}")
    summary = summarize_scores(results)

    print("\n=== Evaluation Summary ===")
    for metric, stats in summary.items():
        print(
            f"{metric:18s} mean={stats['mean']:.3f}  "
            f"min={stats['min']:.3f}  max={stats['max']:.3f}"
        )

    # Optionally write summary to a JSON file
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
