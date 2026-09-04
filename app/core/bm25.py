import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter

class BM25Engine:
    """
    Production Okapi BM25 implementation from scratch in pure Python.
    Features:
    - Custom financial/regulatory tokenizer preserving decimal numbers and circular codes.
    - Cached document frequencies, inverse document frequency (IDF) with Robertson-Spärck Jones weighting.
    - Fast batch query scoring against indexed corpus.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: List[int] = []
        self.avgdl: float = 0.0
        self.doc_count: int = 0
        self.doc_ids: List[str] = []
        self.doc_term_freqs: List[Counter] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        # Preserve alphanumeric codes, decimals, and percentages
        tokens = re.findall(r'[a-zA-Z0-9]+(?:[\.\/\-][a-zA-Z0-9]+)*(?:%)?', text.lower())
        return [t for t in tokens if len(t) > 1 or t.isdigit()]

    def fit(self, doc_ids: List[str], documents: List[str]):
        """
        Indexes the corpus and precomputes IDF tables.
        """
        self.doc_ids = doc_ids
        self.doc_count = len(documents)
        self.doc_term_freqs = []
        self.doc_lengths = []
        self.df = {}
        
        total_len = 0
        for doc in documents:
            tokens = self._tokenize(doc)
            length = len(tokens)
            self.doc_lengths.append(length)
            total_len += length
            
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for term in tf.keys():
                self.df[term] = self.df.get(term, 0) + 1

        self.avgdl = total_len / max(self.doc_count, 1)

        # Precompute RSJ IDF
        self.idf = {}
        for term, freq in self.df.items():
            # BM25 standard smoothed IDF
            self.idf[term] = math.log(1.0 + (self.doc_count - freq + 0.5) / (freq + 0.5))

    def score_query(self, query: str) -> List[Tuple[str, float]]:
        """
        Returns list of (doc_id, bm25_score) sorted in descending order.
        """
        query_tokens = self._tokenize(query)
        if not query_tokens or self.doc_count == 0:
            return [(d_id, 0.0) for d_id in self.doc_ids]

        scores = [0.0] * self.doc_count
        for token in query_tokens:
            if token not in self.idf:
                continue
            idf_val = self.idf[token]
            for i in range(self.doc_count):
                tf = self.doc_term_freqs[i].get(token, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[i]
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                scores[i] += idf_val * (numerator / denominator)

        results = [(self.doc_ids[i], scores[i]) for i in range(self.doc_count)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results
