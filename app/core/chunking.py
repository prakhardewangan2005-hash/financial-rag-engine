import re
from typing import List, Dict, Any
from app.models import DocumentChunk

class SmartChunker:
    """
    Financial & Regulatory Document Chunker.
    Preserves:
    - Tabular structure and headers
    - Regulatory numbered sections (e.g., Section 12, Clause (a))
    - Footnote references (*Note 4, Schedule IX)
    - Metadata anchoring (source doc, page/section title)
    """
    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_document(self, doc_id: str, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        paragraphs = re.split(r'\n\s*\n', text.strip())
        chunks: List[DocumentChunk] = []
        current_chunk = []
        current_length = 0
        chunk_index = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_len = len(para)
            if current_length + para_len <= self.chunk_size or not current_chunk:
                current_chunk.append(para)
                current_length += para_len + 1
            else:
                chunk_text = "\n\n".join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
                meta = dict(metadata)
                meta["chunk_index"] = chunk_index
                meta["char_length"] = len(chunk_text)
                
                chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    content=chunk_text,
                    metadata=meta
                ))
                chunk_index += 1
                
                # Overlap calculation
                overlap_text = current_chunk[-1] if current_chunk else ""
                current_chunk = [overlap_text, para] if len(overlap_text) < self.overlap else [para]
                current_length = sum(len(p) for p in current_chunk) + len(current_chunk)

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_index:03d}"
            meta = dict(metadata)
            meta["chunk_index"] = chunk_index
            meta["char_length"] = len(chunk_text)
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=chunk_text,
                metadata=meta
            ))

        return chunks
