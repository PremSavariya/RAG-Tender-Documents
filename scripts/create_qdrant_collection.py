#!/usr/bin/env python3
"""Create the Qdrant collection expected by the RAG app (empty, ready for indexing)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from core.config import CFG


def main() -> None:
    client = QdrantClient(url=CFG.qdrant_url, prefer_grpc=False)
    name = CFG.qdrant_collection
    dim = int(CFG.embed_dim)

    try:
        client.get_collection(name)
        print(f"Collection already exists: {name}")
        return
    except Exception:
        pass

    client.create_collection(
        collection_name=name,
        vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        hnsw_config=qmodels.HnswConfigDiff(m=16, ef_construct=64),
        optimizers_config=qmodels.OptimizersConfigDiff(indexing_threshold=20_000),
        on_disk_payload=False,
    )
    print(f"Created collection '{name}' (dim={dim}) at {CFG.qdrant_url}")


if __name__ == "__main__":
    main()
