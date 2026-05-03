# rag/generator.py
from __future__ import annotations

import os
import time
import uuid
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from rag.llm_config import DETERMINISTIC_SEED, GENERATOR_MODEL
from rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from rag.retriever import Chunk, retrieve, format_context

from monitoring.metrics import MetricsLogger, make_metric


# Deterministic refusal string (must match your prompt instruction)
REFUSAL_EXACT = "The provided context does not contain enough information to answer this question."

_METRICS = MetricsLogger()



@dataclass(frozen=True)
class RAGAnswer:
    question: str
    answer: str
    chunks: List[Chunk]  # retrieved chunks used


def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY in environment.")
    return OpenAI(api_key=api_key)


# Capture numeric citations like [1], [2], ...
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))



def answer_question(
    question: str,
    *,
    top_k: int = 4,
    model: str = GENERATOR_MODEL,
    temperature: float = 0.2,
    chunks_override: Optional[List[Chunk]] = None,
) -> RAGAnswer:
    """
    End-to-end: retrieve context -> ask LLM -> return answer with citations
    Also logs request-level metrics (latency, retrieval distances, refusal/citations).

    If `chunks_override` is provided, retrieval is skipped and the supplied chunks
    are used as context. Used by CI golden tests for frozen-retrieval regression.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # 1) Retrieve (or use frozen chunks for deterministic tests)
    if chunks_override is not None:
        chunks = chunks_override
    else:
        chunks = retrieve(question, top_k=top_k)


    context = format_context(chunks)


    # 2) Generate
    user_prompt = RAG_USER_PROMPT_TEMPLATE.format(context=context, question=question)

    client = _get_openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        seed=DETERMINISTIC_SEED,
    )
    text = resp.choices[0].message.content or ""

    # 3) Metrics
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    text = (resp.choices[0].message.content or "").strip()

    refusal = text.strip() == REFUSAL_EXACT

    cited = _has_citations(text)
    distances = []
    for c in chunks:
        # Chunk is yours; assume it has .distance (as shown in your earlier prints)
        d = getattr(c, "distance", None)
        if isinstance(d, (int, float)):
            distances.append(float(d))
    
    _METRICS.log(
        make_metric(
            request_id=request_id,
            question=question,
            top_k=top_k,
            distances=distances,
            cited=cited,
            refusal=refusal,
            latency_ms=latency_ms,
            source="rag",
            model=model,
            collection="rag-docs",
            extra={
                "num_chars_answer": len(text),
                "num_tokens_est": None,  # keep None unless you add token counting later
            },
        )
    )






    return RAGAnswer(question=question, answer=text, chunks=chunks)
