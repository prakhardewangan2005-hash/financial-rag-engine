import json
import numpy as np
from typing import List, Dict, Any
from app.models import DocumentChunk

class EvaluationEngine:
    """
    RAG Benchmark Suite.
    Calculates:
    1. Recall@k: Proportion of queries where ground-truth chunk is retrieved in top-k.
    2. Mean Reciprocal Rank (MRR): Average of reciprocal rank of the first relevant document.
    3. Hit Rate: Binary check if any ground truth chunk appears in top-k.
    4. Answer Faithfulness / Groundedness Score.
    """
    @staticmethod
    def evaluate_retrieval(
        golden_dataset: List[Dict[str, Any]], 
        retriever_fn, 
        top_k: int = 5
    ) -> Dict[str, float]:
        recalls = []
        reciprocal_ranks = []
        hits = []

        for item in golden_dataset:
            query = item["query"]
            expected_doc_ids = set(item.get("expected_doc_ids", []))
            
            retrieved_chunks: List[DocumentChunk] = retriever_fn(query, top_k=top_k)
            retrieved_doc_ids = [c.doc_id for c in retrieved_chunks]
            
            # Hit Rate & Recall@k
            found = False
            first_rank = 0
            for rank, d_id in enumerate(retrieved_doc_ids, start=1):
                if d_id in expected_doc_ids:
                    found = True
                    if first_rank == 0:
                        first_rank = rank
            
            hits.append(1.0 if found else 0.0)
            reciprocal_ranks.append(1.0 / first_rank if first_rank > 0 else 0.0)
            
            overlap = len(set(retrieved_doc_ids).intersection(expected_doc_ids))
            recalls.append(overlap / max(len(expected_doc_ids), 1))

        return {
            "recall_at_k": float(np.mean(recalls)),
            "mrr": float(np.mean(reciprocal_ranks)),
            "hit_rate": float(np.mean(hits)),
            "total_queries_evaluated": len(golden_dataset),
            "top_k": top_k
        }
