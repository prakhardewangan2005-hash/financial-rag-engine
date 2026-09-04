import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import LocalVectorStore
from app.core.bm25 import BM25Engine
from app.core.hybrid import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query_transform import QueryTransformer
from app.core.evals import EvaluationEngine

def run_evals():
    print("=" * 60)
    print("FINANCIAL RAG ENGINE - GOLDEN EVALUATION BENCHMARK")
    print("=" * 60)
    
    golden_path = settings.BASE_DIR / "evals" / "golden_dataset.json"
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    print(f"Loaded {len(golden_dataset)} golden Q&A pairs from {golden_path}")

    embedding_engine = EmbeddingEngine(dim=settings.EMBEDDING_DIM)
    vector_store = LocalVectorStore(index_dir=settings.INDEX_DIR)
    bm25_engine = BM25Engine()
    bm25_engine.fit(vector_store.chunk_ids, vector_store.contents)
    hybrid = HybridRetriever(vector_store, embedding_engine, bm25_engine)
    reranker = CrossEncoderReranker()

    # Benchmark 1: Dense Only
    print("\n1. Evaluating DENSE RETRIEVAL ONLY (top_k=5)...")
    res_dense = EvaluationEngine.evaluate_retrieval(
        golden_dataset, 
        lambda q, top_k: hybrid.search(q, top_k=top_k, method="dense"),
        top_k=5
    )
    print(f"   Recall@5: {res_dense['recall_at_k']*100:.1f}% | MRR: {res_dense['mrr']:.3f} | Hit Rate: {res_dense['hit_rate']*100:.1f}%")

    # Benchmark 2: Sparse BM25 Only
    print("\n2. Evaluating SPARSE BM25 ONLY (top_k=5)...")
    res_sparse = EvaluationEngine.evaluate_retrieval(
        golden_dataset, 
        lambda q, top_k: hybrid.search(q, top_k=top_k, method="sparse"),
        top_k=5
    )
    print(f"   Recall@5: {res_sparse['recall_at_k']*100:.1f}% | MRR: {res_sparse['mrr']:.3f} | Hit Rate: {res_sparse['hit_rate']*100:.1f}%")

    # Benchmark 3: Hybrid Search (Dense + BM25 RRF)
    print("\n3. Evaluating HYBRID SEARCH (Dense + BM25 RRF) (top_k=5)...")
    res_hybrid = EvaluationEngine.evaluate_retrieval(
        golden_dataset, 
        lambda q, top_k: hybrid.search(q, top_k=top_k, method="hybrid"),
        top_k=5
    )
    print(f"   Recall@5: {res_hybrid['recall_at_k']*100:.1f}% | MRR: {res_hybrid['mrr']:.3f} | Hit Rate: {res_hybrid['hit_rate']*100:.1f}%")

    # Benchmark 4: Full Production Pipeline (Query Transform + Hybrid + Reranker)
    print("\n4. Evaluating FULL PRODUCTION PIPELINE (Query Transform + Hybrid + Cross-Encoder Reranker)...")
    def full_pipeline(q, top_k=5):
        expanded = QueryTransformer.expand_query(q)
        hits = hybrid.search(expanded, top_k=top_k * 2, method="hybrid")
        return reranker.rerank(q, hits, top_k=top_k)

    res_full = EvaluationEngine.evaluate_retrieval(golden_dataset, full_pipeline, top_k=5)
    print(f"   Recall@5: {res_full['recall_at_k']*100:.1f}% | MRR: {res_full['mrr']:.3f} | Hit Rate: {res_full['hit_rate']*100:.1f}%")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON TABLE")
    print("=" * 60)
    print(f"| Architecture Configuration | Recall@5 | MRR | Hit Rate |")
    print(f"|---|---|---|---|")
    print(f"| Dense Vector Only | {res_dense['recall_at_k']*100:.1f}% | {res_dense['mrr']:.3f} | {res_dense['hit_rate']*100:.1f}% |")
    print(f"| Sparse BM25 Only | {res_sparse['recall_at_k']*100:.1f}% | {res_sparse['mrr']:.3f} | {res_sparse['hit_rate']*100:.1f}% |")
    print(f"| Hybrid (Dense + BM25 RRF) | {res_hybrid['recall_at_k']*100:.1f}% | {res_hybrid['mrr']:.3f} | {res_hybrid['hit_rate']*100:.1f}% |")
    print(f"| Full Pipeline (+ Rerank & HyDE) | {res_full['recall_at_k']*100:.1f}% | {res_full['mrr']:.3f} | {res_full['hit_rate']*100:.1f}% |")

if __name__ == "__main__":
    run_evals()
