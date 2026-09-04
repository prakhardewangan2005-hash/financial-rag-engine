import os
from pathlib import Path
from typing import Dict, Any, Tuple
import pypdf

class DocumentParser:
    """
    Multi-format parser handling PDFs and plaintext/markdown documents.
    """
    @staticmethod
    def parse_file(file_path: Path) -> Tuple[str, Dict[str, Any]]:
        path_obj = Path(file_path)
        doc_id = path_obj.stem
        ext = path_obj.suffix.lower()
        
        metadata = {
            "source": path_obj.name,
            "file_type": ext,
            "file_size": path_obj.stat().st_size
        }

        if ext == ".pdf":
            reader = pypdf.PdfReader(str(path_obj))
            text_parts = []
            for i, page in enumerate(reader.pages):
                extracted = page.extract_text()
                if extracted:
                    text_parts.append(f"[Page {i+1}]\n" + extracted)
            full_text = "\n\n".join(text_parts)
            metadata["num_pages"] = len(reader.pages)
        else:
            with open(path_obj, "r", encoding="utf-8", errors="ignore") as f:
                full_text = f.read()
            metadata["num_pages"] = 1

        return full_text, metadata
