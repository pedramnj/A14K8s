#!/usr/bin/env python3
"""
Simple Anti-Caching Solution for AI4K8s
=======================================
"""

import time
from datetime import datetime

def add_no_cache_headers(response):
    """Add no-cache headers to response"""
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    response.headers['Last-Modified'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
    response.headers['ETag'] = f'"{int(time.time())}"'
    return response

def force_fresh_data(func):
    """Decorator to force fresh data collection"""
    def wrapper(*args, **kwargs):
        # Add timestamp to force fresh data
        kwargs['_force_fresh'] = time.time()
        return func(*args, **kwargs)
    return wrapper
