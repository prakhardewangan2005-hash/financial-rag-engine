import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    PROJECT_NAME: str = "Financial Regulatory RAG Engine"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    
    # Storage & DB
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = BASE_DIR / "data" / "index"
    
    # Chunking & Retrieval Parameters
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 80
    DEFAULT_TOP_K: int = 5
    DENSE_WEIGHT: float = 0.5
    SPARSE_WEIGHT: float = 0.5
    RRF_K: int = 60
    
    # Embeddings Dimension
    EMBEDDING_DIM: int = 384
    
    # Cache
    CACHE_ENABLED: bool = True
    CACHE_TTL_SECONDS: int = 3600
    CACHE_MAXSIZE: int = 1000

    # LLM Settings (Local Ollama or Standard OpenAI-Compatible)
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "local")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama3")
    LLM_TEMPERATURE: float = 0.1

settings = Settings()
