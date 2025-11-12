from pydantic import BaseModel, RootModel, Field
from typing import List, Union, Optional




class QuestionRequest(BaseModel):
    """Request model for RAG API"""
    question: str = Field(..., description="The question to ask about the Boeing 737 manual")

class RAGResponse(BaseModel):
    """Response model for RAG API"""
    answer: str = Field(..., description="The answer to the user's question")
    pages: List[int] = Field(..., description="Array of page numbers (1-based index) that support the answer")
