import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.responses import StreamingResponse
from app.models import (
    SearchRequest, SearchResponse, 
    QueryRequest, QueryResponse, 
    IngestResponse, HealthResponse
)
from app.config import settings
from app.core.embeddings import EmbeddingEngine
from app.core.bm25 import BM25Engine
from app.core.vector_store import LocalVectorStore
from app.core.hybrid import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query_transform import QueryTransformer
from app.core.generator import RAGGenerator
from app.core.cache import QueryCache
from app.core.parser import DocumentParser
from app.core.chunking import SmartChunker
from app.core.evals import EvaluationEngine

router = APIRouter()

# Instantiate singletons
embedding_engine = EmbeddingEngine(dim=settings.EMBEDDING_DIM)
bm25_engine = BM25Engine()
vector_store = LocalVectorStore(index_dir=settings.INDEX_DIR)
hybrid_retriever = HybridRetriever(vector_store, embedding_engine, bm25_engine)
reranker = CrossEncoderReranker()
query_cache = QueryCache(max_size=settings.CACHE_MAXSIZE, ttl_seconds=settings.CACHE_TTL_SECONDS)

# Ensure BM25 is fitted if vectors exist
if vector_store.chunk_ids:
    bm25_engine.fit(vector_store.chunk_ids, vector_store.contents)

@router.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version=settings.VERSION,
        total_documents=len(set(vector_store.doc_ids)),
        total_chunks=len(vector_store.chunk_ids),
        vector_dimension=settings.EMBEDDING_DIM,
        cache_entries=query_cache.size
    )

@router.post("/search", response_model=SearchResponse)
def search_documents(req: SearchRequest):
    t0 = time.time()
    chunks = hybrid_retriever.search(
        query=req.query,
        top_k=req.top_k,
        method=req.method,
        filters=req.filters
    )
    elapsed_ms = (time.time() - t0) * 1000.0
    return SearchResponse(
        query=req.query,
        method=req.method,
        total_results=len(chunks),
        chunks=chunks,
        retrieval_time_ms=round(elapsed_ms, 2)
    )

@router.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    cache_key = f"{req.query}_{req.top_k}_{req.method}_{req.use_query_transform}_{req.use_reranker}"
    if settings.CACHE_ENABLED:
        cached = query_cache.get(cache_key)
        if cached:
            return cached

    t_start = time.time()
    latencies = {}

    # Step 1: Query Transformation
    t0 = time.time()
    search_query = req.query
    if req.use_query_transform:
        search_query = QueryTransformer.expand_query(req.query)
    latencies["query_transform_ms"] = round((time.time() - t0) * 1000.0, 2)

    # Step 2: Hybrid Retrieval
    t0 = time.time()
    filters = {"tenant_id": req.tenant_id} if req.tenant_id else None
    retrieved_chunks = hybrid_retriever.search(
        query=search_query,
        top_k=req.top_k * 2 if req.use_reranker else req.top_k,
        method=req.method,
        filters=filters
    )
    latencies["retrieval_ms"] = round((time.time() - t0) * 1000.0, 2)

    # Step 3: Reranking
    t0 = time.time()
    if req.use_reranker and retrieved_chunks:
        final_chunks = reranker.rerank(req.query, retrieved_chunks, top_k=req.top_k)
    else:
        final_chunks = retrieved_chunks[:req.top_k]
    latencies["reranking_ms"] = round((time.time() - t0) * 1000.0, 2)

    # Step 4: Answer Generation & Citations
    t0 = time.time()
    answer_text, citations = RAGGenerator.generate_answer(req.query, final_chunks)
    latencies["generation_ms"] = round((time.time() - t0) * 1000.0, 2)
    latencies["total_e2e_ms"] = round((time.time() - t_start) * 1000.0, 2)

    response = QueryResponse(
        query=req.query,
        answer=answer_text,
        citations=citations,
        retrieval_method=f"{req.method}+reranker" if req.use_reranker else req.method,
        chunks_evaluated=len(retrieved_chunks),
        latency_ms=latencies,
        estimated_cost_usd=0.0
    )

    if settings.CACHE_ENABLED:
        query_cache.set(cache_key, response)

    return response

@router.post("/query/stream")
def query_rag_stream(req: QueryRequest):
    search_query = QueryTransformer.expand_query(req.query) if req.use_query_transform else req.query
    retrieved_chunks = hybrid_retriever.search(query=search_query, top_k=req.top_k, method=req.method)
    if req.use_reranker and retrieved_chunks:
        final_chunks = reranker.rerank(req.query, retrieved_chunks, top_k=req.top_k)
    else:
        final_chunks = retrieved_chunks

    return StreamingResponse(
        RAGGenerator.generate_stream(req.query, final_chunks),
        media_type="text/event-stream"
    )

@router.post("/ingest", response_model=IngestResponse)
def trigger_ingestion():
    t0 = time.time()
    chunker = SmartChunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    data_files = list(settings.DATA_DIR.glob("*.*"))
    
    total_chunks = []
    all_docs = 0

    for file_path in data_files:
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue
        all_docs += 1
        text, meta = DocumentParser.parse_file(file_path)
        chunks = chunker.chunk_document(doc_id=file_path.stem, text=text, metadata=meta)
        total_chunks.extend(chunks)

    if total_chunks:
        texts = [c.content for c in total_chunks]
        embs = embedding_engine.embed_documents(texts)
        vector_store.add_chunks(total_chunks, embs)
        bm25_engine.fit(vector_store.chunk_ids, vector_store.contents)

    elapsed = time.time() - t0
    return IngestResponse(
        status="success",
        documents_indexed=all_docs,
        chunks_created=len(total_chunks),
        elapsed_time_sec=round(elapsed, 2)
    )

@router.get("/evals")
def run_evaluations():
    golden_path = settings.BASE_DIR / "evals" / "golden_dataset.json"
    if not golden_path.exists():
        raise HTTPException(status_code=404, detail="Golden dataset not found.")
    
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_dataset = json.load(f)

    def retrieve_fn(q: str, top_k: int = 5):
        expanded_q = QueryTransformer.expand_query(q)
        hits = hybrid_retriever.search(query=expanded_q, top_k=top_k * 2, method="hybrid")
        return reranker.rerank(q, hits, top_k=top_k)

    results = EvaluationEngine.evaluate_retrieval(golden_dataset, retrieve_fn, top_k=5)
    return results
