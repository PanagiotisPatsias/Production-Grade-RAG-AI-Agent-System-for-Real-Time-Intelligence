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

def compute_mean_overall(results: List[Dict[str, Any]]) -> float:
    overall_scores = []
    for item in results:
        scores = item.get("scores", {})
        if "overall" in scores:
            try:
                overall_scores.append(float(scores["overall"]))
            except (TypeError, ValueError):
                continue

    if not overall_scores:
        raise RuntimeError("No valid 'overall' scores found in results.")

    return statistics.mean(overall_scores)


def main():
    parser = argparse.ArgumentParser(description="Check evaluation quality threshold.")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation_results.jsonl file.",
    )
    parser.add_argument(
        "--min-overall",
        type=float,
        default=0.8,
        help="Minimum allowed mean overall score.",
    )
    args = parser.parse_args()

    path = Path(args.results)
    results = load_results(path)
    mean_overall = compute_mean_overall(results)

    print(f"Mean overall score: {mean_overall:.3f}")
    print(f"Required minimum:  {args.min_overall:.3f}")

    if mean_overall < args.min_overall:
        print("❌ Quality gate FAILED: mean overall below threshold.")
        raise SystemExit(1)
    else:
        print("✅ Quality gate PASSED.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()