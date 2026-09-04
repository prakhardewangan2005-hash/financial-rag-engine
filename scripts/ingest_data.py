import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.core.parser import DocumentParser
from app.core.chunking import SmartChunker
from app.core.embeddings import EmbeddingEngine
from app.core.vector_store import LocalVectorStore
from app.core.bm25 import BM25Engine

def run_ingestion():
    print("=" * 60)
    print("FINANCIAL RAG ENGINE - CORPUS INGESTION & INDEXING")
    print("=" * 60)
    
    t0 = time.time()
    chunker = SmartChunker(chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
    embedding_engine = EmbeddingEngine(dim=settings.EMBEDDING_DIM)
    vector_store = LocalVectorStore(index_dir=settings.INDEX_DIR)
    bm25_engine = BM25Engine()

    data_files = sorted(list(settings.DATA_DIR.glob("*.*")))
    print(f"Found {len(data_files)} files in {settings.DATA_DIR}")

    total_chunks = []
    doc_count = 0
    for file_path in data_files:
        if file_path.suffix.lower() not in [".pdf", ".txt", ".md"]:
            continue
        doc_count += 1
        print(f"  -> Parsing {file_path.name}...")
        text, metadata = DocumentParser.parse_file(file_path)
        chunks = chunker.chunk_document(doc_id=file_path.stem, text=text, metadata=metadata)
        total_chunks.extend(chunks)
        print(f"     Generated {len(chunks)} chunks.")

    print(f"\nGenerating dense embeddings for {len(total_chunks)} chunks (dim={settings.EMBEDDING_DIM})...")
    texts = [c.content for c in total_chunks]
    embs = embedding_engine.embed_documents(texts)

    print("Persisting chunks and vectors in LocalVectorStore index...")
    vector_store.add_chunks(total_chunks, embs)

    print("Fitting BM25 inverted index tables...")
    bm25_engine.fit(vector_store.chunk_ids, vector_store.contents)

    elapsed = time.time() - t0
    print(f"\n[SUCCESS] Indexed {doc_count} documents into {len(total_chunks)} chunks in {elapsed:.2f} seconds.")

if __name__ == "__main__":
    run_ingestion()
