import os 
import json
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

from .metrics import EvaluationScores

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

client = OpenAI(api_key=OPENAI_API_KEY)

JUDGE_SYSTEM_PROMPT = """
Youa are an expert evaluator of produced by a Retrieval-Augmented Generation (RAG) system.

Your task:
- You receive the user question, the RAG answer, the ground-truth / ideal answer (if available) and the retrieved context.
- You must evaluate the RAG answer along several dimensions on 0-1 scale (0 = worst, 1 = best)
- You MUST respond in JSON format, with the exact schema:
{
  "relevance": float,          // 0.0–1.0
  "correctness": float,        // 0.0–1.0
  "grounding": float,          // 0.0–1.0
  "completeness": float,       // 0.0–1.0
  "reasoning_quality": float,  // 0.0–1.0
  "overall": float,            // 0.0–1.0
  "explanation": string
}
Guidelines:
- relevance: Is the answer on-topic and addressing the question?
- correctness: Are the stated facts correct, compared to the ground truth and context?
- grounding: Does the answer rely only on the given context, or does it hallucinate?
- completeness: Does it fully answer the question, covering all key aspects?
- reasoning_quality: Is the reasoning sound, coherent, and logically structured?
- overall: Your overall judgement of the answer quality as a single score.

Return ONLY valid JSON. No extra text.
"""

def evaluate_answer_with_judge_llm(
    question: str,
    rag_answer: str,
    context: str,
    ideal_answer: Optional[str] = None,
    model: str = "gpt-4.1-mini",
) -> EvaluationScores:
    

    user_payload = {
        "question": question,
        "rag_answer": rag_answer,
        "context": context,
        "ideal_answer": ideal_answer,
    }
    
    response = client.chat.completions.create(
        model=model,
        messages = [
            {"role": "system","content": JUDGE_SYSTEM_PROMPT},
            {"role": "user","content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.0)
    
    raw_content = response.choices[0].message.content

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON from judge LLM response: {e}\nRaw content: {raw_content}")
    
    return EvaluationScores(
        relevance=float(data.get("relevance", 0.0)),
        correctness=float(data.get("correctness", 0.0)),
        grounding=float(data.get("grounding", 0.0)),
        completeness=float(data.get("completeness", 0.0)),
        reasoning_quality=float(data.get("reasoning_quality", 0.0)),
        overall=float(data.get("overall", 0.0)),
        explanation=data.get("explanation"),
    )