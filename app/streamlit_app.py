# app/streamlit_app.py
import json
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from rag.ingest import ingest_pdf_bytes
from rag.generator import answer_question
from rag.store import collection_count

from agents.doc_to_action_agent import run_doc_to_action_agent

load_dotenv()

st.set_page_config(page_title="RAG Evaluation Demo", page_icon="📚", layout="wide")

st.title("📚 RAG Demo (GDPR) — Retrieval + Chunk indexes")
st.caption("UI-only app: all RAG logic lives under the `rag/` package.")

tab_chat, tab_agent = st.tabs(["💬 RAG Chat", "🧠 Doc-to-Action Agent"])

# -----------------------------
# Shared: Upload & Ingest (πάνω-πάνω)
# -----------------------------
st.subheader("1) Upload a PDF to index")
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    reset_index = st.checkbox("Reset index before ingest", value=False)
with col2:
    st.write(f"Indexed chunks: **{collection_count()}**")

if uploaded_file is not None:
    with st.spinner("Ingesting PDF into Chroma..."):
        n_chunks = ingest_pdf_bytes(
            uploaded_file.getvalue(),
            filename=uploaded_file.name,
            reset=reset_index,
        )
    if n_chunks == 0:
        st.error("No text found in the PDF.")
    else:
        st.success(f"Indexed **{n_chunks}** chunks from `{uploaded_file.name}`.")
        st.write(f"New total chunks: **{collection_count()}**")

st.divider()

# =============================
# TAB 1) CHAT
# =============================
with tab_chat:
    st.subheader("2) Ask questions over the indexed documents")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        if collection_count() == 0:
            msg = "No documents indexed yet. Upload a PDF first."
            with st.chat_message("assistant"):
                st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving context and generating answer..."):
                    result = answer_question(user_input, top_k=4)

                st.markdown(result.answer)

                with st.expander("Retrieved context (debug)"):
                    for i, ch in enumerate(result.chunks, start=1):
                        st.markdown(
                            f"**[{i}]** source=`{ch.source}` chunk_index=`{ch.chunk_index}` "
                            f"distance=`{ch.distance:.4f}` id=`{ch.id}`"
                        )
                        st.write(ch.text)

            st.session_state.messages.append({"role": "assistant", "content": result.answer})

# =============================
# TAB 2) AGENT
# =============================
with tab_agent:
    st.subheader("2) Turn a request into an action checklist + report (Agent)")

    st.write(
        "This agent retrieves evidence from your indexed documents and returns:\n"
        "- structured JSON (checklist/risks/open questions)\n"
        "- a markdown consulting-style report\n"
        "All grounded with chunk citations like [34]."
    )

    req = st.text_area(
        "Client request",
        placeholder="e.g. Summarize GDPR obligations for a SaaS startup and generate an action checklist.",
        height=130,
    )

    colA, colB = st.columns([1, 1])
    with colA:
        top_k = st.slider("Top-K retrieved chunks", min_value=3, max_value=12, value=8)
    with colB:
        st.write(f"Indexed chunks: **{collection_count()}**")

    run = st.button("Run Agent", type="primary", disabled=(collection_count() == 0))

    if run:
        if collection_count() == 0:
            st.error("No documents indexed yet. Upload a PDF first.")
        elif not req.strip():
            st.error("Please enter a request.")
        else:
            with st.spinner("Running agent (retrieve → plan → JSON → report)..."):
                result = run_doc_to_action_agent(req.strip(), top_k=top_k)

            st.success("Done.")

            st.subheader("Report (Markdown)")
            st.markdown(result.markdown)

            st.subheader("Structured output (JSON)")
            st.json(result.json)

            with st.expander("Retrieved chunk indices"):
                st.write(result.retrieved_chunk_indices)

            # ---- Downloads 
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            report_name = f"doc_to_action_report_{ts}.md"
            json_name = f"doc_to_action_result_{ts}.json"

            st.download_button(
                "⬇️ Download report.md",
                data=result.markdown.encode("utf-8"),
                file_name=report_name,
                mime="text/markdown",
            )
            st.download_button(
                "⬇️ Download result.json",
                data=json.dumps(result.json, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=json_name,
                mime="application/json",
            ) 
