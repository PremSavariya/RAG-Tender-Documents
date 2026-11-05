"""Response models for API endpoints."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class Document(BaseModel):
    """Document model for search results."""
    
    id: str = Field(..., description="Document ID")
    content: str = Field(..., description="Document content")
    score: float = Field(..., description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "20046645_page_1",
                "content": "IT-Projekte müssen folgende Anforderungen erfüllen...",
                "score": 0.85,
                "metadata": {
                    "tender_id": "20046645",
                    "page": 1,
                    "source": "tender_document"
                }
            }
        }


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    
    query: str = Field(..., description="Original search query")
    results: List[Document] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    search_mode: str = Field(..., description="Search mode used")
    process_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "IT-Projekte",
                "results": [
                    {
                        "id": "20046645_page_1",
                        "content": "IT-Projekte müssen...",
                        "score": 0.85,
                        "metadata": {"tender_id": "20046645"}
                    }
                ],
                "total_results": 5,
                "search_mode": "hybrid",
                "process_time": 0.234
            }
        }


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    
    query: str = Field(..., description="User question")
    answer: str = Field(..., description="Generated answer")
    sources: List[Document] = Field(..., description="Source documents used")
    confidence: Optional[float] = Field(None, description="Answer confidence score")
    process_time: float = Field(..., description="Processing time in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "Was sind die Anforderungen für IT-Projekte?",
                "answer": "IT-Projekte müssen folgende Anforderungen erfüllen...",
                "sources": [
                    {
                        "id": "20046645_page_1",
                        "content": "IT-Projekte...",
                        "score": 0.85,
                        "metadata": {"tender_id": "20046645"}
                    }
                ],
                "confidence": 0.92,
                "process_time": 1.456
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    
    detail: str = Field(..., description="Error message")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Invalid API key",
                "timestamp": "2024-11-02T10:30:00"
            }
        }
