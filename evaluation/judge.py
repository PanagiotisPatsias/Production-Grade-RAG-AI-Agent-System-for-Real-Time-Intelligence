# evaluation/judge.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI

from rag.llm_config import DETERMINISTIC_SEED, JUDGE_MODEL


JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of answers produced by a Retrieval-Augmented Generation (RAG) system.

You will receive:
- The user question
- The system answer
- The ideal/ground-truth answer (if provided)
- The retrieved context excerpts (numbered)

Score each dimension on a 0-1 scale (0 worst, 1 best).
Return ONLY valid JSON with this exact schema:
{
  "relevance": float,
  "correctness": float,
  "grounding": float,
  "completeness": float,
  "reasoning_quality": float,
  "overall": float,
  "explanation": string
}

Guidelines:
- relevance: Is it on-topic and answering the question?
- correctness: Factually correct vs ideal answer AND context.
- grounding: Uses ONLY the provided context, no hallucinations.
- completeness: Covers key aspects needed.
- reasoning_quality: Coherent, logically structured, cautious when uncertain.
- overall: single holistic score (not necessarily the mean).

If the context is insufficient, a good answer should say so.
"""


@dataclass(frozen=True)
class JudgeResult:
    relevance: float
    correctness: float
    grounding: float
    completeness: float
    reasoning_quality: float
    overall: float
    explanation: str

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "JudgeResult":
        return JudgeResult(
            relevance=float(d["relevance"]),
            correctness=float(d["correctness"]),
            grounding=float(d["grounding"]),
            completeness=float(d["completeness"]),
            reasoning_quality=float(d["reasoning_quality"]),
            overall=float(d["overall"]),
            explanation=str(d.get("explanation", "")),
        )


def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    return OpenAI(api_key=api_key)


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Robust JSON extraction: handles accidental surrounding text.
    """
    text = text.strip()
    # fast path
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    # try to find first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("Model did not return JSON.")


def judge_answer(
    *,
    question: str,
    answer: str,
    context: str,
    ideal_answer: Optional[str] = None,
    model: str = JUDGE_MODEL,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> JudgeResult:
    """
    Calls an LLM judge to score an answer given question/context/ideal answer.
    Returns strict JSON parsed into a JudgeResult.
    """
    user_prompt = f"""Question:
{question}

System answer:
{answer}

Ideal answer (if available):
{ideal_answer or "N/A"}

Retrieved context excerpts:
{context}
"""

    client = _client()
    last_err: Optional[Exception] = None

    for _ in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                seed=DETERMINISTIC_SEED,
            )
            raw = resp.choices[0].message.content or ""
            data = _extract_json(raw)
            return JudgeResult.from_dict(data)
        except Exception as e:
            last_err = e

    raise RuntimeError(f"Judge failed after retries: {last_err}")
