"""
Manual testing script for the RAG system
Run this after starting the FastAPI server: python -m src.api.main
"""
import requests
import time
import json

API_URL = "http://localhost:8000/api/v1/search"
API_KEY = "dev-key-12345"

def test_query(query: str, top_k: int = 3, query_num: int = 1):
    """Test a single query and measure performance"""
    print(f"\n{'='*80}")
    print(f"🔍 Query #{query_num}: '{query}'")
    print(f"{'='*80}")
    
    headers = {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json"
    }
    
    data = {
        "query": query,
        "top_k": top_k
    }
    
    # Measure time
    start_time = time.time()
    
    try:
        response = requests.post(API_URL, headers=headers, json=data)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"\n✅ Success!")
            print(f"⏱️  Response Time: {elapsed:.3f}s ({elapsed*1000:.0f}ms)")
            print(f"📊 Results Found: {result.get('total_results', 0)}")
            print(f"🔧 Search Mode: {result.get('mode', 'N/A')}")
            
            if 'processing_time_ms' in result:
                print(f"⚙️  Server Processing: {result['processing_time_ms']:.0f}ms")
            
            # Show first result
            if result.get('results'):
                first = result['results'][0]
                print(f"\n📄 Top Result:")
                print(f"   ID: {first.get('id', 'N/A')}")
                print(f"   Score: {first.get('score', 0):.4f}")
                print(f"   Content: {first.get('content', '')[:100]}...")
                
                if first.get('metadata'):
                    meta = first['metadata']
                    if 'source_path' in meta:
                        print(f"   Source: {meta['source_path']}")
                
            return elapsed
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Connection Error!")
        print(f"Make sure FastAPI server is running:")
        print(f"   python -m src.api.main")
        return None
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None


def main():
    print("\n" + "="*80)
    print("🚀 MULTILINGUAL RAG SYSTEM - MANUAL TESTING")
    print("="*80)
    
    # Check server health
    try:
        health = requests.get("http://localhost:8000/api/health", timeout=2)
        if health.status_code == 200:
            info = health.json()
            print(f"\n✅ Server is running!")
            print(f"   Service: {info.get('service', 'N/A')}")
            print(f"   Status: {info.get('status', 'N/A')}")
            print(f"   Version: {info.get('version', 'N/A')}")
        else:
            print(f"\n⚠️  Server responded with status: {health.status_code}")
    except:
        print(f"\n❌ Cannot connect to server!")
        print(f"\nPlease start the server first:")
        print(f"   python -m src.api.main")
        print(f"\nThen run this script again:")
        print(f"   python manual_test.py")
        return
    
    # Test queries
    test_cases = [
        ("IT-Projekte", "First query - COLD (no cache)"),
        ("IT-Projekte", "Second query - CACHED (should be fast!)"),
        ("Bauprojekte", "Different query - COLD"),
        ("Was ist DTAD-ID 20046891?", "ID lookup query"),
    ]
    
    results = []
    
    for idx, (query, description) in enumerate(test_cases, 1):
        print(f"\n{'-'*80}")
        print(f"📝 Test {idx}/4: {description}")
        print(f"{'-'*80}")
        
        elapsed = test_query(query, top_k=3, query_num=idx)
        if elapsed:
            results.append((query, description, elapsed))
        
        # Pause between queries
        if idx < len(test_cases):
            time.sleep(1)
    
    # Summary
    print(f"\n\n{'='*80}")
    print(f"📊 PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for idx, (query, desc, elapsed) in enumerate(results, 1):
        cache_indicator = "⚡ CACHED" if "CACHED" in desc else "🐌 COLD"
        print(f"\n{idx}. {desc}")
        print(f"   Query: '{query}'")
        print(f"   Time: {elapsed:.3f}s ({elapsed*1000:.0f}ms) {cache_indicator}")
    
    # Cache speedup calculation
    if len(results) >= 2:
        first = results[0][2]
        second = results[1][2]
        speedup = first / second if second > 0 else 0
        print(f"\n🚀 Cache Speedup (Query 1 vs 2):")
        print(f"   First query: {first:.3f}s")
        print(f"   Cached query: {second:.3f}s")
        print(f"   Speedup: {speedup:.1f}x faster!")
    
    print(f"\n{'='*80}")
    print(f"✅ Testing Complete!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
