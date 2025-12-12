import json
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions

import os

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY for RAG answer generator.")

client = OpenAI(api_key=OPENAI_API_KEY)

#Load Chroma collection

chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

chroma_collection  = chroma_client.get_or_create_collection(
    name="rag-docs",
    embedding_function=chroma_ef,
)

# -------------------------------
# RAG Prompt
# -------------------------------
def build_rag_prompt(question: str, context: str) -> str:
    return f"""
You are an expert RAG assistant. Answer the question strictly using the provided context.

CONTEXT:
{context}

QUESTION:
{question}

Rules:
- Answer concisely.
- Do not hallucinate. If the context is insufficient, say "The context does not provide this information."
"""


# -------------------------------
# RAG Answer Generation
# -------------------------------
def generate_rag_answer(question: str, top_k: int = 4) -> Dict[str, Any]:
    """
    Retrieves context from ChromaDB and generates an answer using an LLM.
    Returns dict with:
    {
        "retrieved_chunks": [...],
        "rag_answer": "..."
    }
    """

    # Retrieve from Chroma
    results = chroma_collection.query(
        query_texts=[question],
        n_results=top_k,
    )

    docs = results["documents"][0]

    # Merge retrieved chunks into one context
    merged_context = "\n\n".join(docs)

    prompt = build_rag_prompt(question, merged_context)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a helpful RAG assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    return {
        "retrieved_chunks": docs,
        "rag_answer": answer,
    }


# -------------------------------
# Main Batch Processor
# -------------------------------
def process_dataset(input_path: Path, output_path: Path):
    """
    Loads synthetic dataset, generates RAG answers for each example,
    and writes a new dataset with rag_answer included.
    """

    data = json.loads(input_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} synthetic examples.")

    enriched: List[Dict[str, Any]] = []

    for i, ex in enumerate(data, start=1):
        q = ex["question"]

        print(f"[{i}/{len(data)}] Generating RAG answer for: {q[:60]}...")

        rag_result = generate_rag_answer(q)

        new_item = {
            **ex,
            "retrieved_chunks": rag_result["retrieved_chunks"],
            "rag_answer": rag_result["rag_answer"],
        }

        enriched.append(new_item)

    output_path.write_text(
        json.dumps(enriched, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved enriched dataset with RAG answers to: {output_path}")


# -------------------------------
# CLI Entry Point
# -------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate RAG answers for a synthetic dataset.")
    parser.add_argument("--dataset", required=True, help="Path to synthetic JSON dataset.")
    parser.add_argument("--output", default="rag_enriched_dataset.json", help="Output file path.")
    args = parser.parse_args()

    process_dataset(Path(args.dataset), Path(args.output))