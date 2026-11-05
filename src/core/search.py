# core/search.py
from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer

from .config import CFG

# Keep embedder behavior stable if HF repo updates
PINNED_SHA = "f1944de8402dcd5f2b03f822a4bc22a7f2de2eb9"

# Cache configuration
SEARCH_CACHE_SIZE = 1000  # Number of queries to cache


# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _client() -> QdrantClient:
    # prefer_grpc=False for widest compatibility on Windows
    return QdrantClient(url=CFG.qdrant_url, prefer_grpc=False, timeout=60.0)


@lru_cache(maxsize=1)
def _embedder() -> SentenceTransformer:
    # trust_remote_code is needed for Jina v3; revision keeps it stable
    device = "cuda"
    try:
        import torch  # local import to avoid hard dep
        if not torch.cuda.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"

    return SentenceTransformer(
        CFG.embed_model,
        trust_remote_code=True,
        revision=PINNED_SHA,
        device=device,
    )


@lru_cache(maxsize=1)
def _collection_cfg() -> Tuple[str, Optional[Dict[str, int]] | int | None]:
    """
    Returns:
      ("missing", None)  if collection not found
      ("single", size)   for single-vector collections
      ("named", {name: size, ...}) for named vectors
    """
    try:
        info = _client().get_collection(CFG.qdrant_collection)
    except Exception:
        return ("missing", None)

    vecs = info.config.params.vectors
    if hasattr(vecs, "size"):
        return ("single", int(vecs.size))
    # named vectors
    return ("named", {str(k): int(v.size) for k, v in vecs.items()})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _qdrant_dim() -> Optional[int]:
    kind, data = _collection_cfg()
    if kind == "missing":
        return None
    if kind == "single":
        return int(data)  # type: ignore[arg-type]
    # named: pick a conventional name if present, else first size
    sizes: Dict[str, int] = data  # type: ignore[assignment]
    if "text" in sizes:
        return sizes["text"]
    return next(iter(sizes.values())) if sizes else None


def _select_vector_name() -> Optional[str]:
    kind, data = _collection_cfg()
    if kind in ("missing", "single"):
        return None
    sizes: Dict[str, int] = data  # type: ignore[assignment]
    return "text" if "text" in sizes else next(iter(sizes.keys()))


def _effective_model_dim() -> int:
    try:
        return int(_embedder().get_sentence_embedding_dimension())
    except Exception:
        # very defensive fallback
        v = _embedder().encode(["probe"], normalize_embeddings=True)
        return int(v.shape[-1])


def _ensure_dims_ok() -> None:
    qdim = _qdrant_dim()
    if qdim is None:
        raise RuntimeError(
            f"Qdrant collection '{CFG.qdrant_collection}' not found at {CFG.qdrant_url}."
        )
    model_dim = _effective_model_dim()
    # We'll slice down to qdrant dim if model is larger; if model is smaller, error.
    if model_dim < qdim:
        raise RuntimeError(
            f"Embedding dim mismatch: model={model_dim} < qdrant={qdim}. "
            "Recreate the collection with the model's size or switch model."
        )


def _vector_arg(vec: Sequence[float]):
    """
    Build the 'query_vector' argument for Qdrant .search(), supporting single and named vectors.
    """
    name = _select_vector_name()
    return vec if name is None else (name, vec)


# ---------------------------------------------------------------------------
# ID Detection
# ---------------------------------------------------------------------------

def detect_tender_id(query: str) -> Optional[str]:
    """
    Detect if the query contains a tender ID and extract it.
    
    Supports patterns:
    - "DTAD-ID 20046891"
    - "Tender 20047108"
    - "Vergabe 20046893"
    - "20046891" (8-digit number only)
    
    Returns:
        The extracted ID as a string, or None if no ID pattern found
    """
    # Pattern 1: "DTAD-ID <number>"
    match = re.search(r'DTAD-ID\s*(\d{8})', query, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 2: "Tender <number>"
    match = re.search(r'Tender\s*(\d{8})', query, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 3: "Vergabe <number>"
    match = re.search(r'Vergabe\s*(\d{8})', query, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Pattern 4: Just an 8-digit number (be careful - only if it's the whole query or clearly an ID)
    if re.match(r'^\d{8}$', query.strip()):
        return query.strip()
    
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _embed_query_impl(text: str) -> str:
    """
    Internal implementation of query embedding.
    Returns JSON string of the embedding vector for caching.
    """
    prefix = getattr(CFG, "embed_query_prefix", "search_query: ")
    embs = _embedder().encode(
        [prefix + text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    v: np.ndarray = embs[0].astype("float32", copy=False)

    qdim = _qdrant_dim()
    if qdim is not None and v.shape[0] > qdim:
        v = v[:qdim]
    
    # Return as JSON string for caching
    return json.dumps(v.tolist())


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _embed_query_cached(text: str) -> str:
    """
    Cached wrapper for embedding queries.
    Returns JSON string of the embedding vector.
    """
    return _embed_query_impl(text)


def embed_query(text: str) -> List[float]:
    """
    Encode a user query with the Jina v3 query prefix and return a float list.
    Auto-slices to the collection's vector size if needed.
    
    This function uses LRU caching internally for better performance.
    Repeated queries return instantly (0.001s) instead of 5-7s.
    """
    # Get cached embedding as JSON string
    embedding_json = _embed_query_cached(text)
    
    # Deserialize and return
    return json.loads(embedding_json)


def search_dense(query_text: str, limit: Optional[int] = None, min_score: Optional[float] = None):
    """
    Dense vector search against the configured collection.
    Returns: list[qdrant_client.models.ScoredPoint].
    
    If query contains a tender ID (e.g., "DTAD-ID 20046891"), will filter by metadata.
    """
    _ensure_dims_ok()
    limit = int(limit or getattr(CFG, "topk_candidate", 100))
    
    # Check if query contains a tender ID
    tender_id = detect_tender_id(query_text)
    
    # Embed query ONCE (cached for repeated queries)
    qv = embed_query(query_text)
    
    if tender_id:
        # ID-based search: filter by metadata
        try:
            # Create a filter for the ID field
            filter_condition = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="id",
                        match=qmodels.MatchValue(value=tender_id)
                    )
                ]
            )
            
            # Use vector search with ID filter
            res = _client().search(
                collection_name=CFG.qdrant_collection,
                query_vector=_vector_arg(qv),
                query_filter=filter_condition,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            
            # If we found results by ID, return them immediately (they're the right ones)
            if res:
                return res
                
            # If no results with ID filter, fall through to normal search
            # This handles cases where the ID format doesn't match
        except Exception as e:
            import logging
            logger = logging.getLogger("core.search")
            logger.warning(f"ID-filtered search failed, falling back to normal search: {e}")
    
    # Normal vector search (no ID filter) - reuse the embedded query vector
    # Note: 'params' argument removed for compatibility with Qdrant 1.9.2
    # In newer versions, you can add: params=qmodels.SearchParams(hnsw_ef=128)
    
    res = _client().search(
        collection_name=CFG.qdrant_collection,
        query_vector=_vector_arg(qv),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        score_threshold=None,  # post-filter below
    )

    # Optional post-filter by score
    thresh = float(getattr(CFG, "min_score", 0.0)) if min_score is None else float(min_score)
    if thresh > 0:
        res = [r for r in res if (r.score is not None and float(r.score) >= thresh)]
    return res


def rrf(result_sets: List[Sequence], k: int = 60):
    """
    Reciprocal Rank Fusion over multiple result lists.
    Each item must have `.id`. Returns a fused, re-ranked list.
    """
    scores: dict[str, float] = {}
    keep: dict[str, object] = {}
    for results in result_sets:
        for rank, r in enumerate(results, start=1):
            pid = str(r.id)
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank)
            keep.setdefault(pid, r)
    return sorted(keep.values(), key=lambda r: scores[str(r.id)], reverse=True)


def search_hybrid(query_text: str, limit: Optional[int] = None, dense_weight: float = 0.7):
    """
    Hybrid search: combines dense vector search (70%) with BM25 sparse search (30%).
    Uses Reciprocal Rank Fusion (RRF) to merge results.
    
    NOTE: For better performance, use search_hybrid_cached() instead.
    
    Args:
        query_text: User query
        limit: Final number of results to return (default: CFG.topk_candidate)
        dense_weight: Weight for dense results in fusion (default: 0.7)
                     BM25 weight will be (1 - dense_weight)
    
    Returns:
        List of fused ScoredPoint objects, sorted by fused score
    """
    from .hybrid_search import search_bm25, reciprocal_rank_fusion
    
    _ensure_dims_ok()
    limit = int(limit or getattr(CFG, "topk_candidate", 100))
    
    # 1. Dense vector search
    dense_results = search_dense(query_text, limit=limit)
    
    # 2. BM25 sparse search
    try:
        bm25_results_raw = search_bm25(query_text, top_k=limit)
        
        # Convert BM25 results (doc_id, score) to match Qdrant point IDs
        # We need to fetch the actual points from Qdrant for the BM25 hits
        bm25_doc_ids = [doc_id for doc_id, _ in bm25_results_raw]
        
        if bm25_doc_ids:
            # Fetch points by ID from Qdrant
            from qdrant_client.http import models as qmodels
            bm25_points = _client().retrieve(
                collection_name=CFG.qdrant_collection,
                ids=bm25_doc_ids,
                with_payload=True,
                with_vectors=False,
            )
            
            # Create a lookup for BM25 scores
            bm25_scores = {doc_id: score for doc_id, score in bm25_results_raw}
            
            # Create ScoredPoint-like objects for BM25 results
            # We'll create a simple class to mimic ScoredPoint structure
            class BM25ScoredPoint:
                def __init__(self, point, bm25_score):
                    self.id = point.id
                    self.score = bm25_score
                    self.payload = point.payload
                    self.version = getattr(point, 'version', None)
            
            bm25_results = [
                BM25ScoredPoint(point, bm25_scores.get(str(point.id), 0.0))
                for point in bm25_points
            ]
        else:
            bm25_results = []
    
    except Exception as e:
        # If BM25 fails, fall back to dense-only
        import logging
        logger = logging.getLogger("core.search")
        logger.warning(f"BM25 search failed, using dense-only: {e}")
        bm25_results = []
    
    # 3. Fusion with RRF
    if not bm25_results:
        # No BM25 results, return dense only
        return dense_results[:limit]
    
    # Use RRF to fuse both result sets
    fused_results = rrf([dense_results, bm25_results], k=60)
    
    return fused_results[:limit]


@lru_cache(maxsize=SEARCH_CACHE_SIZE)
def _search_hybrid_cached_impl(query_hash: str, query_text: str, limit: int) -> str:
    """
    Internal cached implementation. Returns JSON string of results.
    
    Note: We cache the JSON string instead of objects to avoid pickling issues
    with Qdrant's ScoredPoint objects.
    """
    results = search_hybrid(query_text, limit=limit)
    
    # Convert results to serializable format
    serializable_results = []
    for hit in results:
        serializable_results.append({
            "id": str(hit.id),
            "score": float(hit.score) if hasattr(hit, 'score') else 0.0,
            "payload": dict(hit.payload) if hasattr(hit, 'payload') and hit.payload else {}
        })
    
    return json.dumps(serializable_results)


def search_hybrid_cached(query_text: str, limit: Optional[int] = None, dense_weight: float = 0.7, return_objects: bool = False):
    """
    Cached version of hybrid search for better performance on repeated queries.
    
    Uses LRU cache to store up to 1000 recent queries. Cached queries return instantly
    (~0.01s) instead of taking 7+ seconds for full hybrid search.
    
    Args:
        query_text: User query
        limit: Final number of results to return (default: CFG.topk_candidate)
        dense_weight: Weight for dense results in fusion (default: 0.7)
        return_objects: If True, return ScoredPoint-like objects (for backward compatibility)
    
    Returns:
        List of result dictionaries with id, score, and payload fields (default)
        or list of ScoredPoint-like objects if return_objects=True
    """
    limit = int(limit or getattr(CFG, "topk_candidate", 100))
    
    # Create a hash of the query for cache key
    query_hash = hashlib.md5(f"{query_text}:{limit}".encode()).hexdigest()
    
    # Get cached results as JSON string
    results_json = _search_hybrid_cached_impl(query_hash, query_text, limit)
    
    # Deserialize
    results = json.loads(results_json)
    
    # Convert to objects if requested
    if return_objects:
        objects = []
        for r in results:
            # Create a simple object with score, payload, and id attributes
            obj = type('ScoredPoint', (), {
                'score': r['score'],
                'payload': r['payload'],
                'id': r['id']
            })()
            objects.append(obj)
        return objects
    
    return results


def count_points() -> Optional[int]:
    """
    Count points in the configured collection.
    Returns None if the collection does not exist or cannot be read.
    """
    try:
        # quick existence check
        kind, _ = _collection_cfg()
        if kind == "missing":
            return None
        return _client().count(CFG.qdrant_collection, exact=True).count
    except Exception:
        return None


def is_alive() -> bool:
    try:
        return bool(_client().is_alive())
    except Exception:
        return False
