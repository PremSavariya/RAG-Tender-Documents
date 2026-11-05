"""
Test script for ID lookup functionality
"""

from src.core.search import detect_tender_id, search_hybrid

# Test ID detection
test_queries = [
    "DTAD-ID 20046891",
    "Tender 20047108", 
    "Vergabe 20046893",
    "20046891",
    "IT-Projekte",  # Should not detect ID
]

print("=" * 70)
print("Testing ID Detection")
print("=" * 70)

for query in test_queries:
    detected_id = detect_tender_id(query)
    status = "✓" if detected_id else "✗"
    print(f"{status} Query: '{query}' → ID: {detected_id}")

print("\n" + "=" * 70)
print("Testing ID-Based Search")
print("=" * 70)

# Test actual search with IDs
id_queries = [
    "DTAD-ID 20046891",
    "Tender 20047108",
    "20046893",
]

for query in id_queries:
    print(f"\nQuery: '{query}'")
    print("-" * 70)
    
    try:
        results = search_hybrid(query, limit=3)
        print(f"✓ Retrieved: {len(results)} documents")
        
        if results:
            # Check if the result actually matches the ID
            detected_id = detect_tender_id(query)
            for i, hit in enumerate(results, 1):
                doc_id = hit.payload.get("id", "N/A")
                score = getattr(hit, 'score', 0.0)
                title = hit.payload.get("title", "N/A")[:60]
                
                match_symbol = "✓✓" if doc_id == detected_id else "  "
                print(f"  {match_symbol} [{i}] ID: {doc_id}, Score: {score:.4f}")
                print(f"      Title: {title}")
        else:
            print("  ⚠ No results found")
            
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
print("Testing Normal Search (No ID)")
print("=" * 70)

normal_query = "IT-Projekte"
print(f"\nQuery: '{normal_query}'")
print("-" * 70)

try:
    results = search_hybrid(normal_query, limit=3)
    print(f"✓ Retrieved: {len(results)} documents")
    
    if results:
        for i, hit in enumerate(results, 1):
            doc_id = hit.payload.get("id", "N/A")
            score = getattr(hit, 'score', 0.0)
            title = hit.payload.get("title", "N/A")[:60]
            
            print(f"  [{i}] ID: {doc_id}, Score: {score:.4f}")
            print(f"      Title: {title}")
    else:
        print("  ⚠ No results found")
        
except Exception as e:
    print(f"  ✗ Error: {e}")

print("\n" + "=" * 70)
print("✅ ID Lookup Test Complete")
print("=" * 70)
