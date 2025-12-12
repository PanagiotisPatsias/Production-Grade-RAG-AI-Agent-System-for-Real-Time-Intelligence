import argparse
from pathlib import Path
import json

from .synthetic_dataset import generate_synthetic_examples


def main():
    parser = argparse.ArgumentParser(
        description="Create a synthetic evaluation dataset for RAG."
    )
    parser.add_argument(
        "--input-text",
        type=str,
        required=True,
        help="Path to a .txt file containing domain context (e.g., text extracted from a PDF).",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="How many synthetic examples to generate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="synthetic_eval_dataset.json",
        help="Filename for saving the dataset (JSON).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4.1-mini",
        help="OpenAI model to use as the generator.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_text)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    domain_context = input_path.read_text(encoding="utf-8")
    print(f"Loaded domain context from: {input_path} (length {len(domain_context)} chars)")

    examples = generate_synthetic_examples(
        domain_context=domain_context,
        num_examples=args.num_examples,
        model=args.model,
    )

    output_path.write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(examples)} examples to: {output_path}")


if __name__ == "__main__":
    main()
