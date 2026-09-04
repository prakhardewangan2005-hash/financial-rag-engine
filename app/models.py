from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    retrieval_method: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    method: str = "hybrid"  # "dense", "sparse", "hybrid"
    tenant_id: Optional[str] = None
    filters: Optional[Dict[str, Any]] = None

class SearchResponse(BaseModel):
    query: str
    method: str
    total_results: int
    chunks: List[DocumentChunk]
    retrieval_time_ms: float

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    method: str = "hybrid"
    stream: bool = False
    use_query_transform: bool = True
    use_reranker: bool = True
    tenant_id: Optional[str] = None

class Citation(BaseModel):
    chunk_id: str
    doc_name: str
    source_snippet: str
    relevance_score: float

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    retrieval_method: str
    chunks_evaluated: int
    latency_ms: Dict[str, float]
    estimated_cost_usd: float = 0.0

class IngestResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_created: int
    elapsed_time_sec: float

class HealthResponse(BaseModel):
    status: str
    version: str
    total_documents: int
    total_chunks: int
    vector_dimension: int
    cache_entries: int
