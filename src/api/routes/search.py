"""
Search API routes - Hybrid search endpoint
Integrates BM25 + Dense vector search with RRF fusion
"""

from fastapi import APIRouter, HTTPException, status, Depends
from src.api.models.requests import SearchRequest
from src.api.models.responses import SearchResponse, Document, ErrorResponse
from src.api.security import verify_api_key, check_rate_limit
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# Global search engine instance (lazy loaded)
_search_engine = None


def get_search_engine():
    """Get or initialize the search engine (singleton pattern)."""
    global _search_engine
    
    if _search_engine is None:
        try:
            # Import search functions
            from src.core import search
            from src.core import hybrid_search
            
            logger.info("Initializing search engine...")
            # Test that search functions are accessible
            assert callable(search.search_hybrid_cached)
            assert callable(search.search_dense)
            assert callable(hybrid_search.search_bm25)
            
            _search_engine = {
                "hybrid": search.search_hybrid_cached,  # Use cached version
                "dense": search.search_dense,
                "bm25": hybrid_search.search_bm25
            }
            logger.info("Search engine initialized with caching enabled")
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to initialize search engine: {str(e)}"
            )
    
    return _search_engine


@router.post(
    "/search",
    response_model=SearchResponse,
    responses={
        200: {"description": "Search results returned successfully"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Invalid API key"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"}
    },
    summary="Search documents",
    description="Search tender documents using hybrid search (BM25 + Dense vectors)",
    dependencies=[Depends(verify_api_key), Depends(check_rate_limit)]
)
async def search_documents(request: SearchRequest):
    """
    Search documents using hybrid search.
    
    - **query**: Search query text (1-500 characters)
    - **top_k**: Number of results to return (1-20, default: 5)
    - **search_mode**: hybrid (recommended), dense, or bm25
    - **score_threshold**: Minimum score threshold (optional, dynamic by default)
    
    Returns ranked documents with relevance scores.
    """
    start_time = time.time()
    
    try:
        # Get search functions
        search_engine = get_search_engine()
        
        # Perform search based on mode
        logger.info(f"Search request: query='{request.query}', mode={request.search_mode}, top_k={request.top_k}")
        
        if request.search_mode == "hybrid":
            # Use cached hybrid search - returns list of dicts
            results = search_engine["hybrid"](request.query, limit=request.top_k)
            
            # Convert dict results to Document format
            documents = []
            for result in results:
                documents.append(Document(
                    id=result.get("id", "unknown"),
                    content=result.get("payload", {}).get("text", ""),
                    score=result.get("score", 0.0),
                    metadata={k: v for k, v in result.get("payload", {}).items() if k not in ["text", "doc_id"]}
                ))
                
        elif request.search_mode == "dense":
            results = search_engine["dense"](request.query, limit=request.top_k)
            
            # Convert Qdrant ScoredPoint to Document format
            documents = []
            for result in results:
                documents.append(Document(
                    id=result.id or str(result.payload.get("doc_id", "unknown")),
                    content=result.payload.get("text", ""),
                    score=result.score,
                    metadata={k: v for k, v in result.payload.items() if k not in ["text", "doc_id"]}
                ))
                
        elif request.search_mode == "bm25":
            results = search_engine["bm25"](request.query, top_k=request.top_k)
            
            # Convert BM25 tuples to Document format
            documents = []
            for doc_id, score in results:
                documents.append(Document(
                    id=doc_id,
                    content="",  # BM25 doesn't return content directly
                    score=score,
                    metadata={}
                ))
        else:
            raise ValueError(f"Invalid search mode: {request.search_mode}")
        
        process_time = time.time() - start_time
        
        logger.info(f"Search completed: {len(documents)} results in {process_time:.3f}s")
        
        return SearchResponse(
            query=request.query,
            results=documents,
            total_results=len(documents),
            search_mode=request.search_mode,
            process_time=process_time
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/search/health",
    summary="Search engine health check",
    description="Check if search engine is initialized and healthy"
)
async def search_health():
    """Check search engine health."""
    try:
        search_engine = get_search_engine()
        return {
            "status": "healthy",
            "engine": "HybridSearch",
            "initialized": search_engine is not None
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Search engine unhealthy: {str(e)}"
        )
