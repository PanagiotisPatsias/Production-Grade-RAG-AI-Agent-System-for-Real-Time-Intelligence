# evaluation/reliability.py
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from evaluation.judge import judge_answer
from rag.llm_config import JUDGE_MODEL


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def main():
    p = argparse.ArgumentParser(description="Judge reliability via repeated scoring.")
    p.add_argument("--dataset", required=True, help="Path to dataset JSON (CI frozen-context)")
    p.add_argument("--runs", type=int, default=5, help="How many repeated judge runs per example")
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--output", default="evaluation/artifacts/reliability_summary.json")
    args = p.parse_args()

    data = load_json_list(Path(args.dataset))

    per_example = []
    all_overalls = []

    for ex in data:
        q = ex["question"]
        ideal = ex.get("ideal_answer")
        answer = ex.get("golden_rag_answer") or ex.get("golden_answer") or ""
        context_text = ex.get("context", "")
        context = f"[1] {context_text}" if context_text else "No relevant context found."

        overalls = []
        for _ in range(args.runs):
            jr = judge_answer(
                question=q,
                answer=answer,
                context=context,
                ideal_answer=ideal,
                model=args.judge_model,
                temperature=0.0,
            )
            overalls.append(jr.overall)

        m = statistics.mean(overalls)
        s = statistics.pstdev(overalls)  # population std

        all_overalls.extend(overalls)
        per_example.append(
            {
                "id": ex.get("id"),
                "mean_overall": m,
                "std_overall": s,
                "overalls": overalls,
            }
        )
        print(f"id={ex.get('id')} mean={m:.3f} std={s:.3f} overalls={overalls}")

    summary = {
        "runs_per_example": args.runs,
        "dataset_size": len(data),
        "overall_mean": statistics.mean(all_overalls) if all_overalls else 0.0,
        "overall_std": statistics.pstdev(all_overalls) if all_overalls else 0.0,
        "per_example": per_example,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved reliability summary to: {out}")


if __name__ == "__main__":
    main()
