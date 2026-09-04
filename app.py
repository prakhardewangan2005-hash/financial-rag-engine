import gradio as gr
from app.config import settings
from app.core.embeddings import EmbeddingEngine
from app.core.bm25 import BM25Engine
from app.core.vector_store import LocalVectorStore
from app.core.hybrid import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query_transform import QueryTransformer
from app.core.generator import RAGGenerator

# Initialize engine
vector_store = LocalVectorStore(index_dir=settings.INDEX_DIR)
embedding_engine = EmbeddingEngine(dim=settings.EMBEDDING_DIM)
bm25_engine = BM25Engine()
if vector_store.chunk_ids:
    bm25_engine.fit(vector_store.chunk_ids, vector_store.contents)

hybrid = HybridRetriever(vector_store, embedding_engine, bm25_engine)
reranker = CrossEncoderReranker()

def answer_query(question: str):
    if not question.strip():
        return "Please enter a question.", ""
    
    # 1. Transform query
    search_q = QueryTransformer.expand_query(question)
    
    # 2. Retrieve top candidates
    hits = hybrid.search(search_q, top_k=6, method="hybrid")
    
    # 3. Rerank
    final_hits = reranker.rerank(question, hits, top_k=3)
    
    # 4. Generate answer with citations
    answer, citations = RAGGenerator.generate_answer(question, final_hits)
    
    citations_text = "\n\n".join([
        f"📄 **{c.doc_name}** (Relevance: {c.relevance_score:.3f})\n> {c.source_snippet}"
        for c in citations
    ])
    
    return answer, citations_text

# Build clean Gradio UI
with gr.Blocks(title="Financial & Regulatory RAG Engine") as demo:
    gr.Markdown("# 🏛️ Financial & Regulatory Compliance RAG Engine")
    gr.Markdown("Ask questions across 25 audited financial reports, RBI circulars, and SEBI compliance filings.")
    
    with gr.Row():
        query_input = gr.Textbox(
            label="Your Financial / Compliance Query", 
            placeholder="e.g. What is the maximum Default Loss Guarantee percentage under RBI digital lending directives?",
            lines=2
        )
    
    submit_btn = gr.Button("Search & Synthesize Answer", variant="primary")
    
    answer_output = gr.Markdown(label="Grounded Answer")
    citations_output = gr.Markdown(label="Verified Document Citations")
    
    submit_btn.click(fn=answer_query, inputs=query_input, outputs=[answer_output, citations_output])

if __name__ == "__main__":
    demo.launch()
