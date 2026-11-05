"""
Chat API routes - RAG-powered question answering
Combines retrieval with LLM generation
"""

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse
from src.api.models.requests import ChatRequest
from src.api.models.responses import ChatResponse, Document, ErrorResponse
from src.api.security import verify_api_key, check_rate_limit
import time
import logging
import asyncio
import json

logger = logging.getLogger(__name__)

router = APIRouter()

# Global QA engine instance (lazy loaded)
_qa_engine = None


def get_qa_engine():
    """Get or initialize the QA engine (singleton pattern)."""
    global _qa_engine
    
    if _qa_engine is None:
        try:
            # Import QA functions
            from src.core import qa
            
            logger.info("Initializing QA engine...")
            # Test that QA functions are accessible
            assert callable(qa.answer_query)
            assert callable(qa.retrieve_candidates)
            
            _qa_engine = {
                "answer": qa.answer_query,
                "retrieve": qa.retrieve_candidates
            }
            logger.info("QA engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize QA engine: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize QA engine: {str(e)}"
            )
    
    return _qa_engine


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={
        200: {"description": "Answer generated successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Ask a question",
    description="Ask a question and get an AI-generated answer based on tender documents",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def chat(request: ChatRequest):
    """
    Ask a question and get an AI-generated answer.
    
    - **query**: User question (1-1000 characters)
    - **top_k**: Number of documents to retrieve (1-20, default: 5)
    - **search_mode**: hybrid (recommended), dense, or bm25
    - **stream**: Enable streaming response (default: false)
    
    Returns an answer with source documents.
    """
    start_time = time.time()
    
    try:
        # Get QA functions
        qa_engine = get_qa_engine()
        
        logger.info(f"Chat request: query='{request.query}', top_k={request.top_k}")
        
        # Retrieve candidates (sources)
        candidates = qa_engine["retrieve"](request.query, limit=request.top_k)
        
        # Generate answer using LLM
        answer_text = qa_engine["answer"](request.query)
        
        # Convert candidates to Document format
        sources = []
        for hit in candidates:
            # Extract ID from payload (proper Hit object structure)
            doc_id = hit.payload.get("id") or hit.payload.get("doc_id") or "unknown"
            
            sources.append(Document(
                id=str(doc_id),
                content=hit.text,
                score=hit.score,
                metadata=hit.payload
            ))
        
        process_time = time.time() - start_time
        
        logger.info(f"Chat completed: {len(sources)} sources used in {process_time:.3f}s")
        
        return ChatResponse(
            query=request.query,
            answer=answer_text,
            sources=sources,
            confidence=None,
            process_time=process_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@router.get(
    "/chat/health",
    summary="QA engine health check",
    description="Check if QA engine is initialized and healthy"
)
async def chat_health():
    """Check QA engine health."""
    try:
        qa_engine = get_qa_engine()
        return {
            "status": "healthy",
            "engine": "QAEngine",
            "initialized": qa_engine is not None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"QA engine unhealthy: {str(e)}"
        )
