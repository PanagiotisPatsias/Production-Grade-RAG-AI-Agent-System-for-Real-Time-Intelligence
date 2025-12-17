# 📚 Production-Grade RAG & AI Agent System for Real-Time Intelligence

## Overview

This project implements a **production-grade Retrieval-Augmented Generation (RAG) system** for legal and regulatory documents, with a **strong focus on evaluation, reliability and safety**.

The system is designed to bridge the gap between **LLM prototyping and production deployment**, emphasizing:

- automated LLM evaluation
- hallucination detection
- deterministic refusals
- grounding guarantees
- CI/CD quality gates
- reproducible experimentation

The project aligns closely with real-world requirements in **legal, compliance and enterprise AI systems** and is directly relevant to roles in **Applied AI, LLM Evaluation and MLOps**.

---

## 🎯 Motivation

Large Language Models are powerful, but **unreliable without systematic evaluation**.

In regulated domains (law, compliance, audit, risk), AI systems must:

- provide grounded answers
- refuse when context is insufficient
- avoid hallucinations
- behave consistently across runs
- be monitored continuously after deployment

This repository demonstrates how to build a **trustworthy RAG system** that can be measured, tested and safely deployed.

---

## 🧠 System Architecture

```text
PDF Documents (GDPR)
        │
        ▼
Ingestion Pipeline
(chunking + embeddings)
        │
        ▼
Vector Store (ChromaDB)
        │
        ▼
Retriever (top-k semantic search)
        │
        ▼
Generator (LLM with strict grounding & refusal rules)
        │
        ▼
Answer + Citations
```


The RAG core is implemented as a **reusable Python library**, independent of the UI layer.

---

## 🧩 Core Components

### RAG Library (Engineering Hygiene)
``` text
rag/
├── store.py        # Vector store configuration & access
├── retriever.py    # Deterministic semantic retrieval
├── generator.py    # Grounded answer generation + citations
├── prompts.py     # Centralized prompt templates
├── schemas.py     # Structured outputs

```
This separation enables:

- unit and integration testing without UI
- future agent integration
- evaluation reuse
- scalable system evolution

---

## Monitoring (stdout-only)

Each RAG request emits structured JSON metrics to stdout, including:

- request_id
- latency_ms
- retrieval distances
- refusal flag
- citation flag
- model name
- collection name
- Designed to work seamlessly on Cloud Run without external monitoring tools.

Includes:

- simulated drift detection

- simple alert rules (logged)

---

## 📊 Evaluation Framework (Key Contribution)

A major focus of this project is **automated LLM evaluation**, inspired by real-world evaluation systems used in production AI teams.

### Supported Evaluation Modes

#### 1️⃣ CI Evaluation (Quality Gate)
- Fixed golden dataset
- Blocks deployment if quality drops
- Enforced directly in CI/CD

#### 2️⃣ Nightly Evaluation
- More permissive configuration
- Tracks long-term performance trends

#### 3️⃣ Reliability Testing
- Repeated runs per query
- Measures output variance and stability

#### 4️⃣ Ablation Studies
- chunk size vs answer quality
- retrieval top-k sensitivity analysis

#### 5️⃣ Hallucination & Refusal Testing
- Explicit unanswerable questions
- Deterministic refusals
- Measurable hallucination rate

---

## 🧪 Evaluation Metrics

Each answer is scored using **LLM-as-a-Judge** with a strict JSON schema:

- relevance
- correctness
- grounding
- completeness
- reasoning quality
- overall score

Evaluation is reproducible and fully automated.

---
## Doc-to-Action Agent

A business-facing AI agent that transforms retrieved evidence into
structured consulting deliverables.

Capabilities:

- RAG-based evidence retrieval
- strict JSON output
- executive summary
- action checklist
- risks & mitigations
- citations per action

---
## 🛑 Safety & Trust Mechanisms

### Deterministic Refusals

If the retrieved context is insufficient, the model must respond with:

```
The provided context does not contain enough information to answer this question.
```


This enables:

- measurable refusal rates
- hallucination detection
- compliance-friendly behavior
- deterministic evaluation

---

## 🚦 Automated Quality Gates

The deployment pipeline enforces:

- minimum answer quality
- maximum hallucination rate
- minimum refusal rate for unanswerable queries
- output stability constraints

Deployment is **automatically blocked** if any gate fails.

---

## ⚙️ CI/CD & Deployment

- Dockerized application
- GitHub Actions pipeline
- Automated evaluation before deployment
- Cloud Run deployment (GCP)
- Environment-based configuration

This reflects **modern MLOps best practices** for deploying AI systems safely.

---

## 🧠 Why This Project Matters

### Relevance to Applied LLM & Evaluation Roles

This project demonstrates hands-on experience with:

- LLM evaluation pipelines
- LLM-as-a-judge methodologies
- grounding and hallucination detection
- automated benchmarking
- trustworthy AI system design

---

### Relevance to AI Engineering & MLOps Roles

This project showcases:

- end-to-end AI solution development
- production-ready RAG architectures
- CI/CD for AI workflows
- monitoring, evaluation, and safety
- clean, modular, reusable engineering practices
- lightweight monitoring and drift simulation (stdout-based)


---

## 📌 Technologies Used

- Python
- OpenAI APIs
- ChromaDB
- Streamlit (UI layer only)
- Docker
- GitHub Actions
- Google Cloud Run

---

