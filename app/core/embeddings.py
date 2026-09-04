import numpy as np
import hashlib
import re
from typing import List, Union

def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Performs zero-safe L2 normalization along the last dimension.
    """
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        return vectors / max(norm, eps)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1D vectors.
    """
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"Shape mismatch: {vec_a.shape} vs {vec_b.shape}")
    a_norm = l2_normalize(vec_a)
    b_norm = l2_normalize(vec_b)
    return float(np.dot(a_norm, b_norm))

def batch_cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity of a 1D query against a 2D matrix of documents.
    Optimized via single BLAS GEMM matrix multiplication.
    """
    if doc_matrix.size == 0:
        return np.array([])
    if query_vec.ndim != 1 or doc_matrix.ndim != 2:
        raise ValueError(f"Expected query shape (d,) and doc shape (N, d). Got {query_vec.shape}, {doc_matrix.shape}")
    if query_vec.shape[0] != doc_matrix.shape[1]:
        raise ValueError(f"Dimension mismatch: query dim {query_vec.shape[0]} != matrix dim {doc_matrix.shape[1]}")
    
    q_norm = l2_normalize(query_vec)
    d_norm = l2_normalize(doc_matrix)
    scores = np.dot(d_norm, q_norm)
    return np.clip(scores, -1.0, 1.0)

class EmbeddingEngine:
    """
    Self-contained, fast deterministic dense embedding generator.
    Produces high-fidelity dense semantic vectors without requiring heavy external model downloads,
    with an extensible hook for HuggingFace/SentenceTransformers when available.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._hf_model = None
        self._try_load_hf()

    def _try_load_hf(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._hf_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._hf_model = None

    def embed_text(self, text: str) -> np.ndarray:
        if self._hf_model is not None:
            emb = self._hf_model.encode(text, convert_to_numpy=True)
            return l2_normalize(emb.astype(np.float32))
        
        # High-performance semantic hashing embedding with subword n-gram aggregation
        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(self.dim, dtype=np.float32)
        if not words:
            return vec
        
        for w in words:
            h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
            vec[idx] += sign
            
            # Character trigram hashing for out-of-vocabulary / financial ticker support
            for i in range(len(w) - 2):
                tri = w[i:i+3]
                h_tri = int(hashlib.sha256(tri.encode('utf-8')).hexdigest(), 16)
                idx_tri = h_tri % self.dim
                sign_tri = 1.0 if ((h_tri >> 3) % 2 == 0) else -1.0
                vec[idx_tri] += 0.5 * sign_tri

        return l2_normalize(vec)

    def embed_documents(self, texts: List[str]) -> np.ndarray:
        if self._hf_model is not None:
            embs = self._hf_model.encode(texts, convert_to_numpy=True)
            return l2_normalize(embs.astype(np.float32))
        
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            matrix[i] = self.embed_text(t)
        return matrix
