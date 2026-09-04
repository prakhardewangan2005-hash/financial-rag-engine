import time
from typing import Dict, Any, Optional

class QueryCache:
    """
    In-memory LRU/TTL Cache for query results and expensive embedding calculations.
    """
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if time.time() - entry["timestamp"] > self.ttl_seconds:
            del self._cache[key]
            return None
        return entry["value"]

    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_size:
            # Evict oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]
        self._cache[key] = {
            "value": value,
            "timestamp": time.time()
        }

    def clear(self):
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
