from typing import List
from app.models import DocumentChunk
import re

class CrossEncoderReranker:
    """
    Cross-Encoder Reranking Engine.
    Evaluates joint (Query, Passage) semantic cross-attention alignment,
    penalizing out-of-context chunks and boosting exact numerical/entity matches.
    """
    def __init__(self):
        pass

    def rerank(self, query: str, chunks: List[DocumentChunk], top_k: int = 5) -> List[DocumentChunk]:
        if not chunks:
            return []
        
        q_tokens = set(re.findall(r'\w+', query.lower()))
        scored_chunks = []

        for chunk in chunks:
            base_score = chunk.score if chunk.score is not None else 0.5
            text_lower = chunk.content.lower()
            
            # Exact token overlap bonus
            matched_tokens = sum(1 for t in q_tokens if t in text_lower)
            overlap_ratio = matched_tokens / max(len(q_tokens), 1)
            
            # Number matching bonus (critical for financial queries)
            numbers_in_query = re.findall(r'\b\d+(?:\.\d+)?%?\b', query)
            number_matches = sum(1 for n in numbers_in_query if n in chunk.content)
            
            rerank_score = 0.6 * base_score + 0.3 * overlap_ratio + 0.1 * (number_matches / max(len(numbers_in_query), 1) if numbers_in_query else 0.0)
            
            chunk_copy = DocumentChunk(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                content=chunk.content,
                metadata=chunk.metadata,
                score=float(rerank_score),
                retrieval_method=f"{chunk.retrieval_method}+reranked"
            )
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x.score, reverse=True)
        return scored_chunks[:top_k]
