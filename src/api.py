import os
import sys
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from model.models import QuestionRequest, RAGResponse
from src.document_processor import VectorStoreManager
from src.rag_service import HybridBoeingRAGService
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException


rag_service: Optional[HybridBoeingRAGService] = None
vectorstore_manager: Optional[VectorStoreManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize RAG service on startup"""
    global rag_service, vectorstore_manager

    try:
        log.info("Starting RAG service...")

        # load config from env
        pdf_path = os.getenv("BOEING_MANUAL_PATH", "data/document_analysis/Boeing B737 Manual-1.pdf")
        index_dir = os.getenv("FAISS_INDEX_DIR", "faiss_index")
        top_k = int(os.getenv("TOP_K", "20"))
        rerank_top_k = int(os.getenv("RERANK_TOP_K", "15"))
        process_images = os.getenv("PROCESS_IMAGES", "false").lower() == "true"

        # setup vector store
        vectorstore_manager = VectorStoreManager(index_dir=index_dir, process_images=process_images)
        vectorstore = vectorstore_manager.get_or_create_vector_store(pdf_path)

        # initialize RAG service
        rag_service = HybridBoeingRAGService(
            vectorstore=vectorstore,
            top_k=top_k,
            rerank_top_k=rerank_top_k
        )

        log.info("Service started")
    except Exception as e:
        log.error("Failed to start service", error=str(e))
        raise

    yield
    log.info("Shutting down...")


app = FastAPI(
    title="Boeing 737 Manual RAG API",
    description="RAG service for Boeing 737 technical manual",
    version="1.0.0",
    lifespan=lifespan
)

# enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Boeing 737 Manual RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Check if service is ready"""
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return {"status": "healthy", "ready": True}


@app.post("/query", response_model=RAGResponse)
async def query_manual(request: QuestionRequest) -> RAGResponse:
    """Main query endpoint"""
    try:
        if rag_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )

        question = request.question.strip()
        if not question:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        log.info("Processing query", q=question[:100])
        response = rag_service.query(question)
        log.info("Query done", pages=len(response.pages))

        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
