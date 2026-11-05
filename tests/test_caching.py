"""
Test script for query caching functionality
"""

import time
from src.core.search import search_hybrid, search_hybrid_cached

print("=" * 70)
print("Testing Query Caching Performance")
print("=" * 70)

test_queries = [
    "IT-Projekte",
    "Bauprojekte",
    "IT-Projekte",  # Repeat - should be fast
    "Bauprojekte",  # Repeat - should be fast
]

print("\n1. Testing WITHOUT caching (search_hybrid):")
print("-" * 70)

for i, query in enumerate(test_queries, 1):
    start = time.time()
    results = search_hybrid(query, limit=3)
    elapsed = time.time() - start
    
    print(f"  Query {i}: '{query}' → {len(results)} results in {elapsed:.3f}s")

print("\n2. Testing WITH caching (search_hybrid_cached):")
print("-" * 70)

for i, query in enumerate(test_queries, 1):
    start = time.time()
    results = search_hybrid_cached(query, limit=3)
    elapsed = time.time() - start
    
    cache_symbol = "🚀" if i > 2 else "  "
    print(f"{cache_symbol} Query {i}: '{query}' → {len(results)} results in {elapsed:.3f}s")

print("\n3. Cache Performance Comparison:")
print("-" * 70)

# Measure first query (cold)
query = "IT-Projekte"
start = time.time()
search_hybrid(query, limit=3)
time_uncached = time.time() - start

# Measure cached query (warm) - run twice to ensure it's cached
search_hybrid_cached(query, limit=3)  # First call to populate cache
start = time.time()
search_hybrid_cached(query, limit=3)  # Second call from cache
time_cached = time.time() - start

speedup = time_uncached / time_cached if time_cached > 0 else float('inf')

print(f"  Uncached query: {time_uncached:.3f}s")
print(f"  Cached query:   {time_cached:.3f}s")
print(f"  Speedup:        {speedup:.1f}x faster")

print("\n4. Verify Result Consistency:")
print("-" * 70)

# Compare results from both methods
results_uncached = search_hybrid("IT-Projekte", limit=3)
results_cached = search_hybrid_cached("IT-Projekte", limit=3)

print(f"  Uncached results: {len(results_uncached)} documents")
print(f"  Cached results:   {len(results_cached)} documents")

# Check if IDs match (order should be same)
if len(results_uncached) == len(results_cached):
    ids_match = all(
        str(r1.id) == r2["id"] 
        for r1, r2 in zip(results_uncached, results_cached)
    )
    print(f"  Result IDs match: {'✓ Yes' if ids_match else '✗ No'}")
else:
    print(f"  ⚠ Different number of results")

print("\n" + "=" * 70)
print("✅ Cache Test Complete")
print("=" * 70)
print("\nExpected improvements:")
print("  - First query: ~7s (cold)")
print("  - Cached query: ~0.01s (hot)")
print("  - Speedup: 500-700x faster")
