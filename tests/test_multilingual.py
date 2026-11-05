"""
Test script for multilingual support (Jina v3 embeddings)
"""

from src.core.search import _embedder, search_hybrid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("=" * 70)
print("Testing Jina v3 Multilingual Capability")
print("=" * 70)

# Test queries in different languages (same meaning)
test_pairs = [
    ("IT projects", "IT-Projekte"),  # English-German
    ("Construction tenders", "Bauprojekte"),  # English-German
    ("Medical equipment supply", "Medizinische Geräte"),  # English-German
]

print("\n1. Cross-Lingual Embedding Similarity:")
print("-" * 70)

embedder = _embedder()

for en_query, de_query in test_pairs:
    # Encode both queries
    en_emb = embedder.encode([en_query], normalize_embeddings=True)
    de_emb = embedder.encode([de_query], normalize_embeddings=True)
    
    # Calculate cosine similarity
    similarity = cosine_similarity(en_emb, de_emb)[0][0]
    
    # Assess multilingual capability
    if similarity > 0.7:
        status = "✅ Excellent"
        symbol = "🟢"
    elif similarity > 0.5:
        status = "⚠️  Good"
        symbol = "🟡"
    else:
        status = "❌ Poor"
        symbol = "🔴"
    
    print(f"{symbol} '{en_query}' ↔ '{de_query}'")
    print(f"    Similarity: {similarity:.4f} ({status})")

print("\n2. Search Results Comparison:")
print("-" * 70)

for en_query, de_query in test_pairs:
    print(f"\n  Testing: '{en_query}' vs '{de_query}'")
    
    # Search with English query
    en_results = search_hybrid(en_query, limit=3)
    
    # Search with German query
    de_results = search_hybrid(de_query, limit=3)
    
    # Compare top results
    en_ids = [str(r.id) for r in en_results]
    de_ids = [str(r.id) for r in de_results]
    
    # Calculate overlap
    common_ids = set(en_ids) & set(de_ids)
    overlap_pct = len(common_ids) / max(len(en_ids), 1) * 100
    
    if overlap_pct > 50:
        status = "✅ Good overlap"
        symbol = "🟢"
    elif overlap_pct > 20:
        status = "⚠️  Some overlap"
        symbol = "🟡"
    else:
        status = "❌ Poor overlap"
        symbol = "🔴"
    
    print(f"    {symbol} Overlap: {overlap_pct:.1f}% ({len(common_ids)}/3 docs) - {status}")
    
    if overlap_pct < 50:
        print(f"    EN top doc: {en_ids[0] if en_ids else 'N/A'}")
        print(f"    DE top doc: {de_ids[0] if de_ids else 'N/A'}")

print("\n3. Embedding Model Info:")
print("-" * 70)

model_name = embedder.model_card_data.base_model if hasattr(embedder, 'model_card_data') else "Unknown"
embedding_dim = embedder.get_sentence_embedding_dimension()

print(f"  Model: {embedder._model_card_data.model_name if hasattr(embedder, '_model_card_data') else 'jinaai/jina-embeddings-v3'}")
print(f"  Embedding dimension: {embedding_dim}")
print(f"  Expected multilingual: Yes (Jina v3)")

print("\n" + "=" * 70)
print("Multilingual Assessment")
print("=" * 70)

# Overall assessment
print("\n📊 Summary:")
print("  - Jina v3 is designed for multilingual support")
print("  - Similarity threshold: >0.7 = Excellent, >0.5 = Good, <0.5 = Poor")
print("  - Search overlap: >50% = Good, >20% = Acceptable, <20% = Needs improvement")

print("\n💡 Recommendations:")
print("  1. If similarity < 0.5: Add query translation layer")
print("  2. If overlap < 20%: Consider re-embedding with multilingual prefix")
print("  3. If similarity > 0.7 but overlap < 50%: Documents may be language-specific")

print("\n" + "=" * 70)
