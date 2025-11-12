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


rag_service: Optional[HybridBoeingRAGService] = None
vectorstore_manager: Optional[VectorStoreManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service, vectorstore_manager

    try:
        log.info("Starting RAG service...")

        pdf_path = os.getenv("BOEING_MANUAL_PATH", "data/document_analysis/Boeing B737 Manual-1.pdf")
        index_dir = os.getenv("FAISS_INDEX_DIR", "faiss_index")
        top_k = int(os.getenv("TOP_K", "15"))
        rerank_top_k = int(os.getenv("RERANK_TOP_K", "10"))

        vectorstore_manager = VectorStoreManager(index_dir=index_dir)
        vectorstore = vectorstore_manager.get_or_create_vector_store(pdf_path)

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "Boeing 737 Manual RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    if rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized"
        )
    return {"status": "healthy", "ready": True}


@app.post("/query", response_model=RAGResponse)
async def query_manual(request: QuestionRequest) -> RAGResponse:
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
    except DocumentPortalException as e:
        log.error("Query failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )
    except Exception as e:
        log.error("Unexpected error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@app.get("/retrieval-info")
async def get_retrieval_info(question: str):
    try:
        if rag_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not initialized"
            )

        if not question.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Question cannot be empty"
            )

        return rag_service.get_retrieval_score_info(question)

    except HTTPException:
        raise
    except Exception as e:
        log.error("Retrieval info failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"
        )


@app.exception_handler(DocumentPortalException)
async def document_portal_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    log.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"}
    )
