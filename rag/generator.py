# rag/generator.py
from __future__ import annotations

import os
import time
import uuid
import re
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from rag.prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE
from rag.retriever import Chunk, retrieve, format_context

from monitoring.metrics import MetricsLogger, make_metric


# Deterministic refusal string (must match your prompt instruction)
REFUSAL_EXACT = "The provided context does not contain enough information to answer this question."

_METRICS = MetricsLogger()


# -----------------------------
# Guardrails (runtime safety)
# -----------------------------
# 1) Enforce: any non-refusal answer must contain citations like [1]
ENFORCE_CITATIONS = os.getenv("RAG_ENFORCE_CITATIONS", "true").lower() in ("1", "true", "yes")

# 2) Enforce: citation indices must be within [1..len(chunks)]
ENFORCE_VALID_CITATIONS = os.getenv("RAG_ENFORCE_VALID_CITATIONS", "true").lower() in ("1", "true", "yes")

# 3) Optional: refuse early if retrieval is weak (based on distance)
# Set env var to enable, e.g. RAG_MAX_DISTANCE="0.60"
_MAX_DISTANCE_ENV = os.getenv("RAG_MAX_DISTANCE", "").strip()
MAX_DISTANCE: Optional[float] = float(_MAX_DISTANCE_ENV) if _MAX_DISTANCE_ENV else None

# 4) For definition/reference questions, require "definition-like evidence" in context
ENFORCE_DEFINITION_EVIDENCE = os.getenv("RAG_ENFORCE_DEFINITION_EVIDENCE", "true").lower() in ("1", "true", "yes")


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


def _extract_citation_numbers(text: str) -> List[int]:
    return [int(m.group(1)) for m in _CITATION_RE.finditer(text or "")]


def _has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text or ""))


def _distances(chunks: List[Chunk]) -> List[float]:
    out: List[float] = []
    for c in chunks:
        d = getattr(c, "distance", None)
        if isinstance(d, (int, float)):
            out.append(float(d))
    return out


def _min_distance(chunks: List[Chunk]) -> Optional[float]:
    ds = _distances(chunks)
    return min(ds) if ds else None


def _is_definition_or_reference_question(q: str) -> bool:
    ql = (q or "").lower()
    triggers = [
        "define",
        "definition",
        "difference between",
        "what is",
        "what does",
        "from when",
        "as stated",
        "according to article",
        "according to",
        "under article",
        "article ",
    ]
    return any(t in ql for t in triggers)


def _has_definition_like_evidence(ctx: str) -> bool:
    """
    Lightweight heuristics to detect that the retrieved context includes something
    likely to contain an explicit definition or reference phrasing.
    This avoids answering definition/reference questions from unrelated recitals.
    """
    cl = (ctx or "").lower()
    patterns = [
        "for the purposes of this regulation",
        "the following definitions apply",
        "definitions apply",
        "means",
        "refers to",
        "in its conclusions of",
        "article 2",
        "definitions",
    ]
    return any(p in cl for p in patterns)


def _log_metrics(
    *,
    request_id: str,
    question: str,
    top_k: int,
    chunks: List[Chunk],
    answer_text: str,
    refusal: bool,
    cited: bool,
    model: str,
    t0: float,
    extra: Optional[dict] = None,
) -> None:
    latency_ms = int((time.perf_counter() - t0) * 1000.0)
    ds = _distances(chunks)

    _METRICS.log(
        make_metric(
            request_id=request_id,
            question=question,
            top_k=top_k,
            distances=ds,
            cited=cited,
            refusal=refusal,
            latency_ms=latency_ms,
            source="rag",
            model=model,
            collection="rag-docs",
            extra={
                "num_chars_answer": len(answer_text or ""),
                "num_tokens_est": None,
                **(extra or {}),
            },
        )
    )


def answer_question(
    question: str,
    *,
    top_k: int = 4,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
) -> RAGAnswer:
    """
    End-to-end: retrieve context -> ask LLM -> return answer with citations/refusals.
    Includes runtime guardrails:
      - weak retrieval -> refuse (optional threshold)
      - definition/reference questions require definition-like evidence in context
      - enforce citations for non-refusal answers
      - enforce citations indices are within retrieved chunk range
    Also logs request-level metrics (latency, retrieval distances, refusal/citations).
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()

    # 1) Retrieve
    chunks = retrieve(question, top_k=top_k)

    # If nothing retrieved, refuse deterministically (no LLM call)
    if not chunks:
        text = REFUSAL_EXACT
        _log_metrics(
            request_id=request_id,
            question=question,
            top_k=top_k,
            chunks=[],
            answer_text=text,
            refusal=True,
            cited=False,
            model=model,
            t0=t0,
            extra={"refusal_reason": "no_chunks"},
        )
        return RAGAnswer(question=question, answer=text, chunks=[])

    # Optional: refuse early if retrieval is weak (distance threshold)
    md = _min_distance(chunks)
    if MAX_DISTANCE is not None and md is not None and md > MAX_DISTANCE:
        text = REFUSAL_EXACT
        _log_metrics(
            request_id=request_id,
            question=question,
            top_k=top_k,
            chunks=chunks,
            answer_text=text,
            refusal=True,
            cited=False,
            model=model,
            t0=t0,
            extra={"refusal_reason": "weak_retrieval", "min_distance": md, "max_distance": MAX_DISTANCE},
        )
        return RAGAnswer(question=question, answer=text, chunks=chunks)

    context = format_context(chunks)

    # Definition/reference questions: require definition-like evidence in retrieved context
    if (
        ENFORCE_DEFINITION_EVIDENCE
        and _is_definition_or_reference_question(question)
        and not _has_definition_like_evidence(context)
    ):
        text = REFUSAL_EXACT
        _log_metrics(
            request_id=request_id,
            question=question,
            top_k=top_k,
            chunks=chunks,
            answer_text=text,
            refusal=True,
            cited=False,
            model=model,
            t0=t0,
            extra={"refusal_reason": "missing_definition_evidence"},
        )
        return RAGAnswer(question=question, answer=text, chunks=chunks)

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
    )

    text = (resp.choices[0].message.content or "").strip()

    # 3) Post-guardrails: enforce refusal/citations
    refusal = text == REFUSAL_EXACT
    cited = _has_citations(text)

    # Guardrail: if not refusal, must have citations
    if ENFORCE_CITATIONS and (not refusal) and (not cited):
        text = REFUSAL_EXACT
        refusal = True
        cited = False

    # Guardrail: citations must be within [1..len(chunks)]
    if ENFORCE_VALID_CITATIONS and (not refusal):
        nums = _extract_citation_numbers(text)
        # If it cites something, validate indices. (If no citations, previous guardrail handles it.)
        if nums and any(n < 1 or n > len(chunks) for n in nums):
            text = REFUSAL_EXACT
            refusal = True
            cited = False

    # 4) Metrics
    _log_metrics(
        request_id=request_id,
        question=question,
        top_k=top_k,
        chunks=chunks,
        answer_text=text,
        refusal=refusal,
        cited=cited,
        model=model,
        t0=t0,
    )

    return RAGAnswer(question=question, answer=text, chunks=chunks)
