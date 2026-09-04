# 🏛️ Enterprise Financial & Regulatory RAG Engine

[![Live Web Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-orange?style=for-the-badge)](https://huggingface.co/spaces/prakhardewangan/financial-rag-engine)
[![Live Swagger API](https://img.shields.io/badge/Render-Live%20API-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://financial-rag-engine.onrender.com/docs)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Recall@5](https://img.shields.io/badge/Recall%405-100%25-brightgreen.svg?style=for-the-badge)]()
[![MRR](https://img.shields.io/badge/MRR-1.000-success.svg?style=for-the-badge)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

An enterprise-grade, zero-framework Retrieval-Augmented Generation (RAG) engine built from first principles for querying complex financial statements, regulatory circulars (RBI & SEBI), multi-column balance sheets, and statutory audit reports.

---

## 🔗 Live Deployments

* **Interactive Web App (Gradio):** [huggingface.co/spaces/prakhardewangan/financial-rag-engine](https://huggingface.co/spaces/prakhardewangan/financial-rag-engine)
* **REST API & Swagger Docs (FastAPI):** [financial-rag-engine.onrender.com/docs](https://financial-rag-engine.onrender.com/docs)

---

## 🏛️ System Architecture

```text
                               ┌───────────────────────────┐
                               │  Client / Web Interface   │
                               │  (Hugging Face Gradio UI) │
                               └─────────────┬─────────────┘
                                             │ HTTP REST / SSE
                                             ▼
                               ┌───────────────────────────┐
                               │  FastAPI Production Engine│
                               │  (Deployed on Render)     │
                               │  - Request Profiling      │
                               │  - LRU/TTL Response Cache │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Query Transformation     │
                               │  - Acronym Expansion (NII,│
                               │    PAT, DLG, FEMA, CRAR)  │
                               │  - HyDE Passage Synthesis │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
        ┌───────────────────────────┐                 ┌───────────────────────────┐
        │  Dense Semantic Retrieval │                 │   Sparse Exact Retrieval  │
        │  - Vectorized Cosine Sim  │                 │   - Okapi BM25 Engine     │
        │  - L2-Normalized BLAS GEMM│                 │   - Numeric/Ticker Aware  │
        └─────────────┬─────────────┘                 └─────────────┬─────────────┘
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                               ┌───────────────────────────┐
                               │  Reciprocal Rank Fusion   │
                               │  (RRF Hybrid Aggregator)  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │  Cross-Encoder Reranker   │
                               │  - Token Alignment        │
                               │  - Numerical Match Bonus  │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                               ┌───────────────────────────┐
                               │ Context-Grounded Generator│
                               │  - Zero Hallucination     │
                               │  - Verifiable Citations   │
                               └───────────────────────────┘
```

---

## 📊 Evaluation Benchmarks (40 Golden Q&A Pairs)

Evaluated across 40 complex regulatory and financial test cases against 25 real financial filings, RBI master directions, and audited statements:

| Architecture Configuration | Recall@5 | Mean Reciprocal Rank (MRR) | Hit Rate | Avg Latency (p95) | Cost / Query |
|---|---|---|---|---|---|
| **Dense Vector Only** | 95.0% | 0.888 | 95.0% | ~3.2 ms | $0.00 |
| **Sparse BM25 Only** | 100.0% | 1.000 | 100.0% | ~1.8 ms | $0.00 |
| **Hybrid Search (Dense + BM25 RRF)** | 100.0% | 0.965 | 100.0% | ~4.5 ms | $0.00 |
| **Full Production Pipeline** | **100.0%** | **1.000** | **100.0%** | **~4.3 ms** | **$0.00** |

---

## 🎯 Key Engineering Features

1. **Zero External Framework Lock-In:** Implemented from scratch using pure Python, NumPy, and FastAPI without LangChain or LlamaIndex wrappers.
2. **Table & Footnote-Aware Parser:** Parses complex 2D financial tables and preserves footnotes without severing rows or mixing columns.
3. **Hybrid Retrieval (RRF):** Fuses dense semantic vector search with custom financial-tokenized Okapi BM25 to achieve high recall on exact circular codes and numbers.
4. **Sub-5ms In-Memory Latency:** Vectorized NumPy matrix operations execute dense similarity calculations in milliseconds without requiring expensive third-party vector databases.
5. **Deterministic Citations:** Every synthesized answer provides direct source attribution, including exact document names and relevance confidence scores.

---

## 📁 Repository Structure

```text
financial-rag-engine/
├── app/
│   ├── api/
│   │   └── routes.py           # REST endpoints (/query, /search, /ingest, /evals, /health)
│   ├── core/
│   │   ├── embeddings.py       # Dense embedding engine & vectorized cosine similarity
│   │   ├── bm25.py             # Okapi BM25 sparse keyword retriever from scratch
│   │   ├── chunking.py         # Table-preserving smart document chunker
│   │   ├── parser.py           # PDF and TXT multi-format document parser
│   │   ├── vector_store.py     # High-speed local embedded vector & document store
│   │   ├── hybrid.py           # Hybrid RRF search engine
│   │   ├── reranker.py         # Cross-encoder reranking engine
│   │   ├── query_transform.py  # HyDE, synonym & acronym query expansion
│   │   ├── generator.py        # Grounded response generator with citations
│   │   ├── cache.py            # In-memory LRU/TTL query response cache
│   │   └── evals.py            # Evaluation suite (Recall@k, MRR, Hit Rate)
│   ├── config.py               # Pydantic settings & environment configuration
│   ├── models.py               # Request and response data contracts
│   └── main.py                 # FastAPI entrypoint & middleware
├── data/                       # 25 Financial filings & regulatory circulars
├── evals/
│   └── golden_dataset.json     # 40 Ground-truth Q&A test cases
├── scripts/
│   ├── ingest_data.py          # Corpus ingestion & index generation script
│   └── run_evals.py            # Automated benchmark evaluation runner
├── Dockerfile                  # Production container definition
├── docker-compose.yml          # Container orchestration specification
├── render.yaml                 # 1-Click deployment blueprint for Render
├── requirements.txt            # Pinned dependencies
└── README.md                   # Project documentation
```

---

## ⚡ Local Setup & Quickstart

### 1. Clone & Install
```bash
git clone [https://github.com/prakhardewangan2005-hash/financial-rag-engine.git](https://github.com/prakhardewangan2005-hash/financial-rag-engine.git)
cd financial-rag-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Ingest Dataset & Run Server
```bash
# Ingest and index documents
python scripts/ingest_data.py

# Start the API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive Swagger UI will be available at: **http://localhost:8000/docs**

### 3. Run Benchmark Evaluations
```bash
python scripts/run_evals.py
```

---

## 🐳 Run with Docker

```bash
docker build -t financial-rag-engine .
docker run -p 8000:8000 financial-rag-engine
```

---

## 📡 API Reference & cURL Example

### Query the RAG Pipeline
```bash
curl -X POST "[https://financial-rag-engine.onrender.com/api/v1/query](https://financial-rag-engine.onrender.com/api/v1/query)" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "What is the maximum permissible Default Loss Guarantee percentage under RBI digital lending directives?",
       "top_k": 3,
       "method": "hybrid",
       "use_query_transform": true,
       "use_reranker": true
     }'
```

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for more information.
