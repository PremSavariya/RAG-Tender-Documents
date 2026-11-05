"""Request models for API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, Literal


class SearchRequest(BaseModel):
    """Request model for search endpoint."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Search query text",
        example="IT-Projekte"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    
    search_mode: Literal["hybrid", "dense", "bm25"] = Field(
        default="hybrid",
        description="Search mode: hybrid (recommended), dense, or bm25"
    )
    
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum score threshold (None for dynamic)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "IT-Projekte",
                "top_k": 5,
                "search_mode": "hybrid",
                "score_threshold": None
            }
        }


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question",
        example="Was sind die Anforderungen für IT-Projekte?"
    )
    
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    
    search_mode: Literal["hybrid", "dense", "bm25"] = Field(
        default="hybrid",
        description="Search mode for retrieval"
    )
    
    stream: bool = Field(
        default=False,
        description="Enable streaming response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "Was sind die Anforderungen für IT-Projekte?",
                "top_k": 5,
                "search_mode": "hybrid",
                "stream": False
            }
        }
