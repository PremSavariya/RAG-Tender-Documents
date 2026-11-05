"""
Security utilities for API authentication and rate limiting.
Zero-cost implementation using in-memory storage.
"""

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, Optional
import os

# API Key configuration
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("RAGBOT_API_KEY", "dev-key-12345")  # Change in production!

# Initialize API key header
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)


# Rate limiting (in-memory)
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests: Dict[str, list] = defaultdict(list)
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        now = datetime.now()
        one_minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if req_time > one_minute_ago
        ]
        
        # Check limit
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_id].append(now)
        return True


# Global rate limiter instance
rate_limiter = RateLimiter(requests_per_minute=60)


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verify API key from request header.
    
    Args:
        api_key: API key from X-API-Key header
        
    Returns:
        Verified API key
        
    Raises:
        HTTPException: If API key is invalid
    """
    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key


async def check_rate_limit(api_key: str = Security(api_key_header)) -> None:
    """
    Check rate limit for API key.
    
    Args:
        api_key: API key from header
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    if not rate_limiter.check_rate_limit(api_key):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Maximum 60 requests per minute."
        )
