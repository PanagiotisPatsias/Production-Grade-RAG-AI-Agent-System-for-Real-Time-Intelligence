import os
import json
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")        
if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")
client = OpenAI(api_key=OPENAI_API_KEY)


GENERATOR_SYSTEM_PROMPT = """
You are an expert at creating evaluation datasets for Retrieval-Augmented Generation (RAG) systems.


Your goal:
- Given a domain description or context text, you will create diverse question-answer pairs that can be used to evaluate RAG systems.


Requirements:
- Output MUST be valid JSON ONLY, with a list of objects.
- Each object must have the following fields:
  - "id": string (unique identifier, e.g. "ex1", "ex2", ...)
  - "question": string
  - "ideal_answer": string
  - "context": string  (a short passage that contains enough info to answer the question)

  Guidelines:
- Use the provided domain context to create questions that are answerable from that context.
- Include a mix of:
  - simple factoid questions,
  - multi-hop reasoning questions,
  - "why/how" questions,
  - slightly tricky or adversarial questions that test hallucination resistance.
- Keep "context" relatively short (1–3 paragraphs) but sufficient to answer the question.
- "ideal_answer" should be concise, correct, and fully grounded in the context text.

Return ONLY a JSON array, e.g.:

[
  {
    "id": "ex1",
    "question": "...",
    "ideal_answer": "...",
    "context": "..."
  }
]
"""

def generate_synthetic_examples(
        domain_context:str,
        num_examples: int = 10,
        model: str = "gpt-4.1-mini",
    ) -> List[Dict[str, Any]]:

    """It makes synthetic Q&A for RAG evaluation based on the given domain context."""

    user_prompt = {
        "domain_context": domain_context,
        "num_examples": num_examples,   
    }

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_prompt, ensure_ascii=False)},
        ],
        temperature=0.7,
    )
    raw_content = response.choices[0].message.content

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError:
        raise RuntimeError(f"LLM generator επέστρεψε μη έγκυρο JSON: {raw_content!r}")

    if not isinstance(data, list):
        raise RuntimeError(f"Περιμέναμε JSON array, πήραμε: {type(data)}")

    # Basic sanity check
    for i, ex in enumerate(data, start=1):
        if "id" not in ex:
            ex["id"] = f"ex{i}"

    return data

