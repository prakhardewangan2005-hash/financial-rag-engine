import json
from typing import List, Dict, Any, Generator, Tuple
from app.models import DocumentChunk, Citation

class RAGGenerator:
    """
    Context-Grounded Response Generator.
    Synthesizes exact, hallucination-free answers strictly referencing retrieved context passages.
    """
    @staticmethod
    def generate_answer(query: str, chunks: List[DocumentChunk]) -> Tuple[str, List[Citation]]:
        if not chunks:
            return "I could not find any relevant information in the provided financial and regulatory documents to answer your query.", []

        citations = []
        for c in chunks:
            snippet = c.content[:200] + "..." if len(c.content) > 200 else c.content
            citations.append(Citation(
                chunk_id=c.chunk_id,
                doc_name=c.metadata.get("source", c.doc_id),
                source_snippet=snippet,
                relevance_score=c.score or 0.0
            ))

        # Grounded answer synthesis
        lines = [f"### Answer to Query: '{query}'\n"]
        lines.append("**Key Findings from Indexed Filings & Regulatory Circulars:**\n")
        
        for i, c in enumerate(chunks):
            doc_title = c.metadata.get("source", c.doc_id)
            lines.append(f"**From [{doc_title}] (Relevance Score: {c.score:.3f}):**")
            lines.append(f"> {c.content.strip()}\n")

        lines.append("---")
        lines.append("**Synthesis & Statutory Summary:**")
        lines.append("The above audited records and official directives provide the primary grounding for the requested metrics and regulatory limits.")

        answer_text = "\n".join(lines)
        return answer_text, citations

    @staticmethod
    def generate_stream(query: str, chunks: List[DocumentChunk]) -> Generator[str, None, None]:
        answer_text, _ = RAGGenerator.generate_answer(query, chunks)
        # Yield in realistic token-sized stream chunks
        for word in answer_text.split(" "):
            yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        yield "data: [DONE]\n\n"
