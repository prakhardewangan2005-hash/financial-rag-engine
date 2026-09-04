from typing import List, Dict, Any, Optional
from app.models import DocumentChunk
from app.core.embeddings import EmbeddingEngine
from app.core.bm25 import BM25Engine
from app.core.vector_store import LocalVectorStore

class HybridRetriever:
    """
    Hybrid Search Engine fusing Dense Vector Embeddings and Sparse BM25 via
    Reciprocal Rank Fusion (RRF) and Weighted Score Combination.
    """
    def __init__(self, vector_store: LocalVectorStore, embedding_engine: EmbeddingEngine, bm25_engine: BM25Engine):
        self.vector_store = vector_store
        self.embedding_engine = embedding_engine
        self.bm25_engine = bm25_engine

    def search(
        self, 
        query: str, 
        top_k: int = 5, 
        method: str = "hybrid", 
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
        rrf_k: int = 60,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        
        if method == "dense":
            q_vec = self.embedding_engine.embed_text(query)
            dense_hits = self.vector_store.search(q_vec, top_k=top_k, filters=filters)
            return [hit[0] for hit in dense_hits]

        if method == "sparse":
            bm25_scores = self.bm25_engine.score_query(query)
            chunk_dict = {c_id: (self.vector_store.contents[i], self.vector_store.metadatas[i], self.vector_store.doc_ids[i])
                          for i, c_id in enumerate(self.vector_store.chunk_ids)}
            results = []
            for c_id, score in bm25_scores[:top_k]:
                if c_id in chunk_dict:
                    content, meta, d_id = chunk_dict[c_id]
                    results.append(DocumentChunk(
                        chunk_id=c_id,
                        doc_id=d_id,
                        content=content,
                        metadata=meta,
                        score=score,
                        retrieval_method="sparse"
                    ))
            return results

        # Hybrid Fusion via Reciprocal Rank Fusion (RRF)
        q_vec = self.embedding_engine.embed_text(query)
        dense_hits = self.vector_store.search(q_vec, top_k=top_k * 3, filters=filters)
        sparse_hits = self.bm25_engine.score_query(query)[:top_k * 3]

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, DocumentChunk] = {}

        # 1. Rank dense hits
        for rank, (chunk, score) in enumerate(dense_hits):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + dense_weight * (1.0 / (rrf_k + rank + 1))
            chunk_map[chunk.chunk_id] = chunk

        # 2. Rank sparse hits
        chunk_dict = {c_id: (self.vector_store.contents[i], self.vector_store.metadatas[i], self.vector_store.doc_ids[i])
                      for i, c_id in enumerate(self.vector_store.chunk_ids)}
        for rank, (c_id, score) in enumerate(sparse_hits):
            rrf_scores[c_id] = rrf_scores.get(c_id, 0.0) + sparse_weight * (1.0 / (rrf_k + rank + 1))
            if c_id not in chunk_map and c_id in chunk_dict:
                content, meta, d_id = chunk_dict[c_id]
                chunk_map[c_id] = DocumentChunk(
                    chunk_id=c_id,
                    doc_id=d_id,
                    content=content,
                    metadata=meta,
                    score=score,
                    retrieval_method="hybrid"
                )

        # Sort by fused score
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        final_results = []
        for c_id, score in sorted_chunks[:top_k]:
            c = chunk_map[c_id]
            c.score = float(score)
            c.retrieval_method = "hybrid_rrf"
            final_results.append(c)

        return final_results
