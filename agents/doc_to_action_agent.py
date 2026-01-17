# agents/doc_to_action_agent.py 
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from rag.retriever import retrieve
from agents.prompts import AGENT_SYSTEM_PROMPT, AGENT_USER_PROMPT_TEMPLATE

REFUSAL = "The provided context does not contain enough information to answer this question."


@dataclass
class AgentResult:
    request: str
    json: Dict[str, Any]
    markdown: str
    retrieved_chunk_indices: List[int]


def _format_chunks_for_prompt(chunks) -> str:
    # chunks are your rag.retriever.Chunk objects
    lines = []
    for c in chunks:
        # keep it compact but usable
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 900:
            snippet = snippet[:900] + "..."
        lines.append(f"[{c.chunk_index}] {snippet}")
    return "\n".join(lines)


def _call_llm_json(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=os.getenv("AGENT_MODEL", "gpt-4.1-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def _safe_parse_json(text: str) -> Dict[str, Any]:
    t = (text or "").strip()
    if t == REFUSAL:
        return {"refusal": True, "message": REFUSAL}
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        # last-resort: try to find a JSON block
        start = t.find("{")
        end = t.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(t[start : end + 1])
        raise


def _render_markdown(request: str, payload: Dict[str, Any]) -> str:
    if payload.get("refusal"):
        return f"# Doc-to-Action Report\n\n**Request:** {request}\n\n{REFUSAL}\n"

    summary = payload.get("summary", [])
    checklist = payload.get("action_checklist", [])
    risks = payload.get("risks", [])
    open_q = payload.get("open_questions", [])
    cites = payload.get("citations_used", [])

    md = []
    md.append("# Doc-to-Action Report")
    md.append("")
    md.append(f"**Request:** {request}")
    md.append("")
    md.append("## Executive summary")
    for s in summary[:3]:
        md.append(f"- {s}")
    md.append("")
    md.append("## Action checklist")
    for i, item in enumerate(checklist, start=1):
        task = item.get("task", "")
        owner = item.get("owner_role", "")
        prio = item.get("priority", "")
        ev = item.get("evidence", [])
        ev_txt = " ".join([f"[{x}]" for x in ev]) if ev else ""
        md.append(f"{i}. **({prio})** {task} — _Owner:_ {owner} {ev_txt}".rstrip())
    md.append("")
    md.append("## Risks & mitigations")
    for r in risks:
        risk = r.get("risk", "")
        sev = r.get("severity", "")
        mit = r.get("mitigation", "")
        ev = r.get("evidence", [])
        ev_txt = " ".join([f"[{x}]" for x in ev]) if ev else ""
        md.append(f"- **{sev.upper()}**: {risk} — _Mitigation:_ {mit} {ev_txt}".rstrip())
    md.append("")
    md.append("## Open questions")
    for q in open_q:
        md.append(f"- {q}")
    md.append("")
    md.append("## Chunks indexes")
    md.append(", ".join([f"[{c}]" for c in cites]) if cites else "_None_")
    md.append("")
    md.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")

    return "\n".join(md)


def run_doc_to_action_agent(request: str, top_k: int = 8) -> AgentResult:
    chunks = retrieve(request, top_k=top_k)
    chunk_indices = [c.chunk_index for c in chunks]

    chunks_text = _format_chunks_for_prompt(chunks)
    prompt = AGENT_USER_PROMPT_TEMPLATE.format(request=request, chunks=chunks_text)

    raw = _call_llm_json(prompt)
    payload = _safe_parse_json(raw)

    md = _render_markdown(request, payload)
    return AgentResult(request=request, json=payload, markdown=md, retrieved_chunk_indices=chunk_indices)


def main():
    p = argparse.ArgumentParser(description="Doc-to-Action Agent (RAG + structured output + report).")
    p.add_argument("--request", required=True, help="Client request / objective")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-dir", default="artifacts/agent")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_doc_to_action_agent(args.request, top_k=args.top_k)

    (out_dir / "result.json").write_text(json.dumps(result.json, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "report.md").write_text(result.markdown, encoding="utf-8")

    print("✅ Saved:")
    print(" -", out_dir / "result.json")
    print(" -", out_dir / "report.md")


if __name__ == "__main__":
    main()
