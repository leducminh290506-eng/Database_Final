"""
api.py - FastAPI backend for the Vietnamese Legal Assistant RAG System.

Endpoints:
  POST /query    - Submit a legal question and get a RAG-generated answer
  POST /compare  - Run same query through Milvus + PostgreSQL and compare
  GET  /health   - Health check endpoint
  GET  /stats    - Index statistics

Features:
  - CORS middleware for frontend integration
  - Input validation via Pydantic models
  - Structured JSON responses
  - Async processing for non-blocking embedding + retrieval
  - Exception handling with informative error messages
  - Request logging
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from loguru import logger

import config
from src.rag_pipeline import RAGPipeline


# ============================================================
# Pydantic Models (Request / Response schemas)
# ============================================================

class QueryRequest(BaseModel):
    """Request schema for the /query endpoint."""
    question: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Câu hỏi pháp lý bằng tiếng Việt (Vietnamese legal question)",
        examples=["Quy định về thuế thu nhập cá nhân là gì?"],
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Number of documents to retrieve (1-50)",
    )


class DocumentResult(BaseModel):
    """Schema for a single retrieved document in the response."""
    chunk_id: str
    doc_id: str
    text: str
    highlighted_text: str
    score: float
    title: str
    doc_number: str
    doc_type: str
    date_issued: str
    authority: str


class QueryResponse(BaseModel):
    """Response schema for the /query endpoint."""
    question: str
    answer: str
    documents: list[DocumentResult]
    is_valid: bool
    error: str = ""
    processing_time_ms: float = 0.0


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str
    index_size: int
    model: str


# ── Compare schemas ────────────────────────────────────────────

class CompareMethodResult(BaseModel):
    """Results from a single search method."""
    method: str
    time_ms: float
    results: list[dict]


class CompareResponse(BaseModel):
    """Response schema for the /compare endpoint."""
    query: str
    top_k: int
    methods: list[CompareMethodResult]
    overlap: dict = {}


# ============================================================
# Application Lifecycle
# ============================================================

# Global pipeline instance (initialized at startup)
rag_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.
    
    - On startup: Load the Milvus index and initialize the RAG pipeline
    - On shutdown: Clean up resources
    """
    global rag_pipeline
    
    logger.info("Starting up Vietnamese Legal Assistant API...")
    try:
        rag_pipeline = RAGPipeline()
        logger.info("RAG pipeline initialized successfully")
    except FileNotFoundError:
        logger.error(
            "Milvus index not found! Please run 'python scripts/build_index.py' first."
        )
        rag_pipeline = None
    except Exception as e:
        logger.error(f"Failed to initialize RAG pipeline: {e}")
        rag_pipeline = None
    
    yield  # Application runs here
    
    logger.info("Shutting down Vietnamese Legal Assistant API...")


# ============================================================
# FastAPI Application
# ============================================================

app = FastAPI(
    title="Vietnamese Legal Assistant API",
    description=(
        "API trợ lý pháp lý Việt Nam sử dụng RAG (Retrieval-Augmented Generation). "
        "Hệ thống truy xuất văn bản pháp luật liên quan và tạo câu trả lời có trích dẫn nguồn."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the API status, index size, and embedding model name.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Milvus index not loaded. Run build_index.py first.",
        )
    
    return HealthResponse(
        status="healthy",
        index_size=rag_pipeline.retriever.vector_store.size,
        model=config.EMBEDDING_MODEL,
    )


@app.get("/stats")
async def get_stats():
    """
    Return index statistics.
    
    Provides information about the vector store, embedding model,
    and LLM configuration.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")
    
    return {
        "index_size": rag_pipeline.retriever.vector_store.size,
        "embedding_model": config.EMBEDDING_MODEL,
        "embedding_dimension": config.EMBEDDING_DIMENSION,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": config.GEMINI_MODEL if config.LLM_PROVIDER == "gemini" else config.OPENAI_MODEL,
        "top_k": config.TOP_K,
        "chunk_size": config.CHUNK_SIZE,
    }


@app.post("/query", response_model=QueryResponse)
async def query_legal(request: QueryRequest):
    """
    Submit a legal question and receive a RAG-generated answer.
    
    The system:
      1. Embeds the question using SentenceTransformers
      2. Retrieves the TOP-K most relevant legal document chunks from Milvus
      3. Passes the retrieved context to Google Gemini with anti-hallucination prompting
      4. Returns the answer with citations and source documents
    
    Args:
        request: QueryRequest with question and optional top_k.
    
    Returns:
        QueryResponse with answer, retrieved documents, and similarity scores.
    """
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Service unavailable: Milvus index not loaded. Run build_index.py first.",
        )
    
    start_time = time.time()
    
    try:
        # Run the RAG pipeline (async for non-blocking)
        result = await rag_pipeline.query_async(
            question=request.question,
            top_k=request.top_k,
        )
        
        processing_time = (time.time() - start_time) * 1000  # ms
        
        # Build response
        documents = [
            DocumentResult(
                chunk_id=doc.get("chunk_id", ""),
                doc_id=doc.get("doc_id", ""),
                text=doc.get("text", ""),
                highlighted_text=doc.get("highlighted_text", ""),
                score=doc.get("score", 0.0),
                title=doc.get("title", ""),
                doc_number=doc.get("doc_number", ""),
                doc_type=doc.get("doc_type", ""),
                date_issued=doc.get("date_issued", ""),
                authority=doc.get("authority", ""),
            )
            for doc in result.get("documents", [])
        ]
        
        logger.info(
            f"Query processed in {processing_time:.0f}ms | "
            f"Question: '{request.question[:50]}...' | "
            f"Docs retrieved: {len(documents)}"
        )
        
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            documents=documents,
            is_valid=result["is_valid"],
            error=result.get("error", ""),
            processing_time_ms=round(processing_time, 2),
        )
    
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}",
        )


@app.post("/compare", response_model=CompareResponse)
async def compare_search(request: QueryRequest):
    """
    Run the same query through Milvus ANN, PostgreSQL ILIKE, and PostgreSQL FTS
    and return structured comparison results for the UI.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="Service unavailable")

    import asyncio
    from src.embedding import embed_query

    query = request.question
    top_k = request.top_k
    methods: list[dict] = []

    # ── 1. Milvus ANN ─────────────────────────────────────────
    try:
        t0 = time.perf_counter()
        q_emb = embed_query(query)
        raw = rag_pipeline.retriever.vector_store.search(q_emb, top_k=top_k)
        milvus_ms = (time.perf_counter() - t0) * 1000
        milvus_results = [
            {
                "chunk_id": r["chunk_id"],
                "title": r["metadata"].get("title", "")[:120],
                "doc_number": r["metadata"].get("doc_number", ""),
                "score": round(r["score"], 4),
                "text_preview": r["text"][:200],
            }
            for r in raw
        ]
        methods.append({"method": "Milvus ANN (IVF_FLAT + Cosine)", "time_ms": round(milvus_ms, 2), "results": milvus_results})
    except Exception as e:
        logger.warning(f"Milvus compare failed: {e}")

    # ── 2. PostgreSQL ILIKE ────────────────────────────────────
    try:
        from postgres.search import search_pg_ilike
        methods.append(search_pg_ilike(query, top_k))
    except Exception as e:
        logger.warning(f"PG ILIKE compare failed: {e}")

    # ── 3. PostgreSQL FTS ──────────────────────────────────────
    try:
        from postgres.search import search_pg_fts
        methods.append(search_pg_fts(query, top_k))
    except Exception as e:
        logger.warning(f"PG FTS compare failed: {e}")

    # ── Overlap analysis ───────────────────────────────────────
    overlap_info: dict = {}
    if len(methods) >= 2:
        milvus_ids = {r["chunk_id"] for r in methods[0].get("results", [])}
        for m in methods[1:]:
            other_ids = {r["chunk_id"] for r in m.get("results", [])}
            key = m["method"]
            overlap_info[key] = {
                "common": len(milvus_ids & other_ids),
                "only_milvus": len(milvus_ids - other_ids),
                "only_pg": len(other_ids - milvus_ids),
            }

    return CompareResponse(
        query=query,
        top_k=top_k,
        methods=[CompareMethodResult(**m) for m in methods],
        overlap=overlap_info,
    )


# ============================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000 --reload
# ============================================================

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level=config.LOG_LEVEL.lower(),
    )
