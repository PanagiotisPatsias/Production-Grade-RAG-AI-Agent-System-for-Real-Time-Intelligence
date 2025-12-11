import os
from typing import List, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter



# ---------- Load env ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("Λείπει το OPENAI_API_KEY από το .env")

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pdf_text = [page.extract_text() for page in reader.pages if page.extract_text() is not None]

    return "\n".join(pdf_text)


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n",".", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    return chunks

import chromadb
from chromadb.utils import embedding_functions

chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Χρησιμοποιούμε built-in OpenAIEmbeddingFunction του Chroma
chroma_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small",
)

# Συλλογή για τα έγγραφά μας
chroma_collection = chroma_client.get_or_create_collection(
    name="rag-docs",
    embedding_function=chroma_ef,
)

def build_rag_prompt(question: str, retrieved_chunks: List[Tuple[int, str]]) -> str:

    if not retrieved_chunks:
        context_text = "No relevant context was found."
    else:
        context_lines = []
        for idx, chunk in retrieved_chunks:
            context_lines.append(f"[{idx}] {chunk}")
        context_text = "\n\n".join(context_lines)

    prompt = f"""
You are an assistant that answers ONLY based on the context provided below.

Document context:
{context_text}

User question:
{question}

Instructions:
- If the answer is not clearly derived from the context, say that you do not have enough information.
- When you use information from a specific excerpt, add its number in brackets at the end of the point, e.g. [12].
"""

    return prompt



# ---------- Simple chat function ----------
def ask_llm(message: str) -> str:
    """
    Στέλνει ένα μήνυμα στο LLM και επιστρέφει την απάντηση.
    Προς το παρόν ΧΩΡΙΣ RAG, απλά LLM.
    """
    response = client.chat.completions.create(
        model="gpt-4.1-mini",   # ή gpt-4o-mini, ανάλογα τι έχεις διαθέσιμο
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant. Answer in Greek."},
            {"role": "user", "content": message},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content


# ---------- Streamlit UI ----------
# -------------------------------------------
# 4. Streamlit UI
# -------------------------------------------
st.set_page_config(page_title="LLM RAG Chat με Chroma", page_icon="📚")

st.title("📚 LLM RAG Chat με OpenAI + ChromaDB")
st.write(
    """
Demo εφαρμογή RAG:

1. Ανεβάζεις PDF.
2. Τα chunks & embeddings αποθηκεύονται σε ChromaDB.
3. Κάνεις ερωτήσεις και γίνεται retrieval + LLM answer.
"""
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- Upload & ingest ----------
st.subheader("1️⃣ Ανέβασε PDF για indexing")

uploaded_file = st.file_uploader("Επίλεξε ένα PDF έγγραφο", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Διαβάζω το PDF και φτιάχνω chunks..."):
        full_text = extract_text_from_pdf(uploaded_file)
        chunks = chunk_text(full_text)

        if not chunks:
            st.error("Δεν βρέθηκε κείμενο στο PDF.")
        else:
            st.write(f"Βρέθηκαν **{len(chunks)}** chunks. Τα προσθέτω στη Chroma συλλογή...")

            ids = []
            metadatas = []
            for i in range(len(chunks)):
                ids.append(f"{uploaded_file.name}-chunk-{i}")
                metadatas.append(
                    {
                        "source": uploaded_file.name,
                        "chunk_index": i,
                    }
                )

            chroma_collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas,
            )

            st.success("Το έγγραφο μπήκε στο RAG index!")
            with st.expander("Δείγμα από τα πρώτα chunks"):
                for i, ch in enumerate(chunks[:3]):
                    st.markdown(f"**Chunk {i}:**")
                    st.write(ch)

# ---------- Chat ----------
st.subheader("2️⃣ Κάνε ερωτήσεις πάνω στα indexed έγγραφα")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Γράψε την ερώτησή σου...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Έλεγχος αν υπάρχουν καθόλου docs στη συλλογή
    if chroma_collection.count() == 0:
        msg = "Δεν υπάρχουν documents στη Chroma. Ανέβασε πρώτα ένα PDF."
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        with st.spinner("Ψάχνω σχετικό context στη Chroma..."):
            results = chroma_collection.query(
                query_texts=[user_input],
                n_results=4,
            )

        docs = results["documents"][0]
        metas = results["metadatas"][0]

        retrieved = []
        for doc, meta in zip(docs, metas):
            idx = meta.get("chunk_index", -1)
            retrieved.append((idx, doc))

        rag_prompt = build_rag_prompt(user_input, retrieved)

        with st.chat_message("assistant"):
            with st.spinner("Σκέφτομαι με βάση τα έγγραφά σου..."):
                answer = ask_llm(rag_prompt)
                st.markdown(answer)

            if retrieved:
                with st.expander("Context που χρησιμοποιήθηκε από Chroma"):
                    for idx, chunk in retrieved:
                        st.markdown(f"**Chunk [{idx}]:**")
                        st.write(chunk)

        st.session_state.messages.append({"role": "assistant", "content": answer})