"""
FastAPI Application for Boeing 737 Manual RAG Service
Provides REST API endpoint for question answering with page citations.
"""
import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model.models import QuestionRequest, RAGResponse
from src.document_processor import VectorStoreManager
from src.rag_service import HybridBoeingRAGService
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


# Global variables for service instances
rag_service: Optional[HybridBoeingRAGService] = None
vectorstore_manager: Optional[VectorStoreManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup: Initialize RAG service
    global rag_service, vectorstore_manager

    try:
        log.info("Starting Boeing RAG Service...")

        # Get PDF path from environment or use default
        pdf_path = os.getenv(
            "BOEING_MANUAL_PATH",
            "data/document_analysis/Boeing B737 Manual-1.pdf"
        )

        # Get index directory from environment or use default
        index_dir = os.getenv("FAISS_INDEX_DIR", "faiss_index")

        # Get retrieval parameters
        top_k = int(os.getenv("TOP_K", "15"))
        rerank_top_k = int(os.getenv("RERANK_TOP_K", "10"))

        log.info("Configuration loaded",
                pdf_path=pdf_path,
                index_dir=index_dir,
                top_k=top_k,
                rerank_top_k=rerank_top_k)

        # Initialize vector store manager
        vectorstore_manager = VectorStoreManager(index_dir=index_dir)

        # Load or create vector store
        vectorstore = vectorstore_manager.get_or_create_vector_store(pdf_path)

        # Initialize Hybrid RAG service with BM25 + Dense retrieval and reranking
        log.info("Initializing Hybrid RAG Service (BM25 + Dense + Reranking)...")
        rag_service = HybridBoeingRAGService(
            vectorstore=vectorstore,
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )

        log.info("Boeing RAG Service started successfully")

    except Exception as e:
        log.error("Failed to start Boeing RAG Service", error=str(e))
        raise

    yield

    # Shutdown: Cleanup
    log.info("Shutting down Boeing RAG Service...")


# Create FastAPI application
app = FastAPI(
    title="Boeing 737 Manual RAG API",
    description="RAG service for querying Boeing 737 technical manual with page citations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API information"""
    return {
        "service": "Boeing 737 Manual RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )

    return {
        "status": "healthy",
        "service": "Boeing 737 Manual RAG",
        "ready": True
    }


@app.post("/query", response_model=RAGResponse)
async def query_manual(request: QuestionRequest) -> RAGResponse:
    """
    Query the Boeing 737 manual with a question.

    Args:
        request: QuestionRequest with the question

    Returns:
        RAGResponse with answer and page citations

    Raises:
        HTTPException: If service not initialized or query processing fails
    """
    try:
        # Check if service is initialized
        if rag_service is None:
            log.error("Query attempted before service initialization")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not initialized"
            )

        # Validate question
        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        log.info("Processing query request", question_preview=question[:100])

        # Process query
        response = rag_service.query(question)

        log.info("Query processed successfully",
                num_pages=len(response.pages),
                answer_preview=response.answer[:150])

        return response

    except HTTPException:
        raise

    except DocumentPortalException as e:
        log.error("DocumentPortalException during query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

    except Exception as e:
        log.error("Unexpected error during query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/retrieval-info")
async def get_retrieval_info(question: str):
    """
    Get detailed retrieval information for a question (for debugging/evaluation).

    Args:
        question: The question to retrieve information for

    Returns:
        Dictionary with retrieval details
    """
    try:
        if rag_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG service not initialized"
            )

        if not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        retrieval_info = rag_service.get_retrieval_score_info(question)
        return retrieval_info

    except HTTPException:
        raise

    except Exception as e:
        log.error("Error getting retrieval info", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


# Error handlers
@app.exception_handler(DocumentPortalException)
async def document_portal_exception_handler(request, exc):
    """Handle DocumentPortalException"""
    log.error("DocumentPortalException", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    log.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn

    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    log.info("Starting API server", host=host, port=port)

    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=False  # Set to True for development
    )
