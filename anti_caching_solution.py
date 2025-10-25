#!/usr/bin/env python3
"""
Anti-Caching Solution for AI4K8s Monitoring System
==================================================

This module implements multiple strategies to eliminate caching issues:
1. Force fresh data collection
2. Add cache-busting headers
3. Implement data freshness validation
4. Add monitoring data timestamps
"""

import time
import hashlib
from datetime import datetime, timedelta
from functools import wraps

class AntiCachingManager:
    """Manages anti-caching strategies for the monitoring system"""
    
    def __init__(self):
        self.data_freshness_threshold = 30  # seconds
        self.last_data_collection = {}
        self.data_checksums = {}
    
    def force_fresh_data(self, data_source_func):
        """Decorator to force fresh data collection"""
        @wraps(data_source_func)
        def wrapper(*args, **kwargs):
            # Check if data is fresh enough
            current_time = time.time()
            data_key = f"{data_source_func.__name__}_{str(args)}_{str(kwargs)}"
            
            if data_key in self.last_data_collection:
                time_since_last = current_time - self.last_data_collection[data_key]
                if time_since_last < self.data_freshness_threshold:
                    print(f"🔄 Forcing fresh data collection for {data_source_func.__name__}")
            
            # Collect fresh data
            fresh_data = data_source_func(*args, **kwargs)
            
            # Update timestamps
            self.last_data_collection[data_key] = current_time
            
            # Add freshness metadata
            if isinstance(fresh_data, dict):
                fresh_data['_freshness_timestamp'] = current_time
                fresh_data['_data_source'] = data_source_func.__name__
                fresh_data['_cache_bust'] = hashlib.md5(str(current_time).encode()).hexdigest()[:8]
            
            return fresh_data
        return wrapper
    
    def add_cache_busting_headers(self, response):
        """Add cache-busting headers to HTTP response"""
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Last-Modified'] = datetime.now().strftime('%a, %d %b %Y %H:%M:%S GMT')
        response.headers['ETag'] = f'"{int(time.time())}"'
        return response
    
    def validate_data_freshness(self, data):
        """Validate that data is fresh enough"""
        if not isinstance(data, dict):
            return True
        
        timestamp = data.get('_freshness_timestamp', 0)
        current_time = time.time()
        
        if current_time - timestamp > self.data_freshness_threshold:
            print(f"⚠️ Data is stale: {current_time - timestamp:.1f}s old")
            return False
        
        return True

# Global anti-caching manager
anti_cache = AntiCachingManager()

def no_cache_headers(f):
    """Decorator to add no-cache headers to Flask routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        response = f(*args, **kwargs)
        return anti_cache.add_cache_busting_headers(response)
    return decorated_function

def force_fresh_data(f):
    """Decorator to force fresh data collection"""
    return anti_cache.force_fresh_data(f)

