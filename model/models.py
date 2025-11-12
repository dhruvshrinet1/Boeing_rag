from pydantic import BaseModel, Field
from typing import List


class QuestionRequest(BaseModel):
    question: str = Field(..., description="The question to ask about the Boeing 737 manual")


class RAGResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's question")
    pages: List[int] = Field(..., description="Array of page numbers (1-based index) that support the answer")
