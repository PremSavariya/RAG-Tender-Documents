"""
Quick test script for FastAPI endpoints
Tests health, search, and chat endpoints with authentication
"""

import requests
import json
import os
from pprint import pprint

BASE_URL = "http://127.0.0.1:8000"

# API Key Configuration
API_KEY = os.environ.get("RAGBOT_API_KEY", "dev-key-12345")
HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

print(f"Using API Key: {API_KEY[:10]}... (masked)")

def test_health():
    """Test health endpoint."""
    print("\n" + "="*50)
    print("TEST 1: Health Check")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    pprint(response.json())
    

def test_search():
    """Test search endpoint."""
    print("\n" + "="*50)
    print("TEST 2: Search Endpoint (with Authentication)")
    print("="*50)
    
    payload = {
        "query": "IT-Projekte",
        "top_k": 3,
        "search_mode": "hybrid"
    }
    
    print(f"\nRequest: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/search", headers=HEADERS, json=payload)
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuery: {data['query']}")
        print(f"Mode: {data['search_mode']}")
        print(f"Results: {data['total_results']}")
        print(f"Time: {data['process_time']:.3f}s")
        
        print("\nTop 3 Results:")
        for i, doc in enumerate(data['results'], 1):
            print(f"\n{i}. Score: {doc['score']:.4f}")
            print(f"   ID: {doc['id']}")
            print(f"   Content: {doc['content'][:100]}...")
    else:
        print(f"Error: {response.text}")


def test_chat():
    """Test chat endpoint."""
    print("\n" + "="*50)
    print("TEST 3: Chat Endpoint (with Authentication)")
    print("="*50)
    
    payload = {
        "query": "Was sind die Anforderungen für IT-Projekte?",
        "top_k": 5,
        "search_mode": "hybrid"
    }
    
    print(f"\nRequest: {json.dumps(payload, indent=2)}")
    response = requests.post(f"{BASE_URL}/api/v1/chat", headers=HEADERS, json=payload)
    print(f"\nStatus: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nQuestion: {data['query']}")
        print(f"Time: {data['process_time']:.3f}s")
        print(f"\nAnswer:\n{data['answer']}")
        print(f"\nSources used: {len(data['sources'])}")
    else:
        print(f"Error: {response.text}")


def test_invalid_api_key():
    """Test invalid API key authentication."""
    print("\n" + "="*50)
    print("TEST 4: Invalid API Key (Security Test)")
    print("="*50)
    
    invalid_headers = {
        "X-API-Key": "wrong-key-123",
        "Content-Type": "application/json"
    }
    
    payload = {"query": "test", "top_k": 3, "search_mode": "hybrid"}
    
    response = requests.post(f"{BASE_URL}/api/v1/search", headers=invalid_headers, json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 401:
        print("✓ Correctly rejected invalid API key")
        pprint(response.json())
    else:
        print(f"✗ Expected 401, got {response.status_code}")


def test_rate_limit():
    """Test rate limiting."""
    print("\n" + "="*50)
    print("TEST 5: Rate Limiting (60 req/min)")
    print("="*50)
    
    payload = {"query": "test", "top_k": 1, "search_mode": "hybrid"}
    
    print("Making 65 rapid requests...")
    for i in range(65):
        response = requests.post(f"{BASE_URL}/api/v1/search", headers=HEADERS, json=payload)
        
        if response.status_code == 429:
            print(f"\n✓ Rate limited at request #{i+1}")
            pprint(response.json())
            return
    
    print("\n✗ No rate limit triggered (might need more requests)")


if __name__ == "__main__":
    try:
        test_health()
        test_search()
        test_chat()
        test_invalid_api_key()
        test_rate_limit()
        
        print("\n" + "="*50)
        print("✓ ALL TESTS COMPLETED")
        print("="*50)
        print(f"\nAPI Documentation: {BASE_URL}/api/docs")
        print(f"ReDoc: {BASE_URL}/api/redoc")
        print(f"\nAPI Key: Set RAGBOT_API_KEY environment variable or use default")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("Make sure the server is running: python -m uvicorn src.api.main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
