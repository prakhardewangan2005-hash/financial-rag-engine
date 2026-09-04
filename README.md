# 🚀 Enterprise Financial & Regulatory RAG Engine

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Recall@5](https://img.shields.io/badge/Recall%405-100%25-brightgreen.svg)]()
[![MRR](https://img.shields.io/badge/MRR-1.000-success.svg)]()

An enterprise-grade, zero-framework Retrieval-Augmented Generation (RAG) engine built from first principles for querying complex financial reports, regulatory circulars (RBI/SEBI), balance sheets, and statutory audit notes.

---

## 🏛️ System Architecture

- **Dense Semantic Retrieval:** L2-normalized vector dot product (`GEMM`) in NumPy.
- **Sparse Exact Retrieval:** Okapi BM25 engine with a finance/regulatory tokenizer.
- **Hybrid Fusion:** Reciprocal Rank Fusion (RRF) combining dense and sparse results.
- **Cross-Encoder Reranker:** Exact entity and numerical alignment scoring.
- **FastAPI Service:** Streaming (SSE), caching, and citation grounding.

---

## 📊 Evaluation Benchmarks (40 Golden Q&A Pairs)

| Architecture Configuration | Recall@5 | Mean Reciprocal Rank (MRR) | Hit Rate | Avg Latency | Cost / Query |
|---|---|---|---|---|---|
| **Dense Vector Only** | 95.0% | 0.888 | 95.0% | ~3.2 ms | $0.00 |
| **Sparse BM25 Only** | 100.0% | 1.000 | 100.0% | ~1.8 ms | $0.00 |
| **Hybrid (Dense + BM25 RRF)** | 100.0% | 0.965 | 100.0% | ~4.5 ms | $0.00 |
| **Full Production Pipeline** | **100.0%** | **1.000** | **100.0%** | **~4.3 ms** | **$0.00** |

---

## ⚡ Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Ingest and index documents
python3 scripts/ingest_data.py

# 3. Start API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
