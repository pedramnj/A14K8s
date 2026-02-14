#!/usr/bin/env python3
"""
AI4K8s LLM-Based Autoscaling Advisor
====================================

Uses LLM (GPT OSS or Groq) to make intelligent autoscaling decisions by analyzing:
- Current resource metrics
- Predictive forecasts
- Historical patterns
- Cost considerations
- Performance requirements

Supports:
- GPT OSS models via OpenAI-compatible API (local/HPC)
- Groq API (cloud-based fallback)

Author: Pedram Nikjooy
Thesis: AI Agent for Kubernetes Management
"""

import json
import os
import logging
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import hashlib
import re

# Configure logger first (before using it)
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only configure if not already configured
    logger.setLevel(logging.WARNING)  # Set to WARNING to see our debug messages
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler - use dynamic path based on current working directory
    try:
        import os
        log_dir = os.path.expanduser('~/ai4k8s')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'llm_advisor.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, continue with console only
        logger.warning(f"Could not set up file logging: {e}")

# Try OpenAI-compatible client first (for GPT OSS), fallback to Groq
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI package not available, will use Groq only")

try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("Groq package not available")

# MCDA optimizer for multi-criteria validation of LLM decisions
try:
    from mcda_optimizer import MCDAAutoscalingOptimizer
    MCDA_AVAILABLE = True
except ImportError:
    MCDA_AVAILABLE = False
    logger.warning("MCDA optimizer not available")

# Configure logger with both file and console handlers
logger = logging.getLogger(__name__)
if not logger.handlers:  # Only configure if not already configured
    logger.setLevel(logging.WARNING)  # Set to WARNING to see our debug messages
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # File handler - use dynamic path based on current working directory
    try:
        import os
        log_dir = os.path.expanduser('~/ai4k8s')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'llm_advisor.log')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # If file logging fails, continue with console only
        logger.warning(f"Could not set up file logging: {e}")

class LLMAutoscalingAdvisor:
    """LLM-powered autoscaling advisor using GPT OSS (primary) or Groq (fallback)"""
    
    def __init__(self, 
                 gpt_oss_api_base: Optional[str] = None,
                 gpt_oss_api_key: Optional[str] = None,
                 gpt_oss_model: Optional[str] = None,
                 groq_api_key: Optional[str] = None, 
                 cache_ttl: int = 300):
        """Initialize LLM advisor with GPT OSS (preferred) or Groq (fallback)
        
        Args:
            gpt_oss_api_base: GPT OSS API base URL (e.g., http://localhost:8000/v1)
            gpt_oss_api_key: GPT OSS API key (optional, can be empty for local)
            gpt_oss_model: GPT OSS model name (e.g., "gpt-4", "deepseek-chat", "mixtral-8x7b")
            groq_api_key: Groq API key (fallback, or from env)
            cache_ttl: Cache TTL in seconds (default 5 minutes)
        """
        # GPT OSS Configuration (primary, for better reasoning)
        self.gpt_oss_api_base = gpt_oss_api_base or os.getenv('GPT_OSS_API_BASE', 'http://localhost:8000/v1')
        self.gpt_oss_api_key = gpt_oss_api_key or os.getenv('GPT_OSS_API_KEY', 'not-needed')
        self.gpt_oss_model = gpt_oss_model or os.getenv('GPT_OSS_MODEL', 'gpt-4')
        
        # Groq Configuration (fallback)
        self.groq_api_key = groq_api_key or os.getenv('GROQ_API_KEY')
        
        # Provider selection: GPT OSS preferred, Groq as fallback
        self.provider = None  # 'gpt_oss' or 'groq'
        self.client = None
        self.model = None
        
        # Deterministic sampling parameters for consistent outputs
        self.temperature = 0.1  # Very low for deterministic outputs (was 0.3)
        self.top_p = 0.95  # Nucleus sampling - focus on top 95% probability mass
        self.top_k = 40  # Top-k sampling - consider top 40 tokens (for Groq only)
        # Reduce max_tokens for GPT OSS (smaller models have 2048 context window)
        # We'll set this dynamically based on provider
        self.max_tokens_gpt_oss = 600  # Reduced to speed up generation while maintaining quality
        self.max_tokens_groq = 1500  # Groq models have larger context
        self.frequency_penalty = 0.0  # No frequency penalty
        self.presence_penalty = 0.0  # No presence penalty
        
        # Caching to prevent rapid successive LLM calls
        self.cache_ttl = cache_ttl  # 5 minutes default
        self.recommendation_cache: Dict[str, Dict[str, Any]] = {}
        # Track last LLM call time to enforce minimum interval
        self.last_llm_call_time: Dict[str, datetime] = {}
        self.min_llm_interval = 30  # Minimum 30 seconds between LLM calls for same deployment
        
        # Initialize GPT OSS client (preferred for better reasoning)
        if OPENAI_AVAILABLE and self.gpt_oss_api_base:
            try:
                self.client = OpenAI(
                    api_key=self.gpt_oss_api_key if self.gpt_oss_api_key != 'not-needed' else 'not-needed',
                    base_url=self.gpt_oss_api_base,
                    timeout=240.0,  # 240 second timeout (Qwen takes 80-120s typically, up to 182s max observed, large buffer to 240s)
                    max_retries=0  # Disable retries to fail fast - no automatic retries
                )
                self.provider = 'gpt_oss'
                self.model = self.gpt_oss_model
                logger.info(f"✅ LLM Autoscaling Advisor initialized with GPT OSS: {self.gpt_oss_api_base}, model: {self.model}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize GPT OSS client: {e}, trying Groq fallback...")
                self.client = None
        
        # Fallback to Groq if GPT OSS not available
        if not self.client and GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.client = groq.Groq(api_key=self.groq_api_key)
                self.provider = 'groq'
                self.model = "llama-3.1-8b-instant"  # Groq model
                self.fallback_model = "llama-3.1-70b-versatile"
                logger.info("✅ LLM Autoscaling Advisor initialized with Groq (fallback)")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None
        
        if not self.client:
            logger.warning("⚠️  No LLM provider available. Set GPT_OSS_API_BASE or GROQ_API_KEY environment variable.")

        # Initialize MCDA optimizer for multi-criteria validation of LLM decisions
        self.mcda_optimizer = None
        if MCDA_AVAILABLE:
            try:
                self.mcda_optimizer = MCDAAutoscalingOptimizer(profile='balanced')
                logger.info("✅ MCDA optimizer initialized for LLM decision validation")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize MCDA optimizer: {e}")

    def _get_cache_key(self, deployment_name: str, namespace: str, 
                      current_metrics: Dict[str, Any], forecast: Dict[str, Any],
                      current_replicas: int, state_management: Optional[str] = None) -> str:
        """Generate cache key from input parameters
        
        Uses very aggressive rounding to create stable cache keys that don't change
        with minor metric fluctuations. This prevents rapid recommendation changes.
        """
        # Round VERY aggressively to create stable cache keys:
        # - CPU/Memory: Round to nearest 25% (e.g., 173% -> 175%, 175% -> 175%, 180% -> 175%)
        #   This ensures 170-180% all use the same cache key
        # - Replicas: Use as-is (discrete value)
        # - Forecast peaks: Round to nearest 25%
        
        def round_to_25(value: float) -> int:
            """Round to nearest 25 (e.g., 173 -> 175, 175 -> 175, 180 -> 175)"""
            return int(round(value / 25.0) * 25)
        
        def round_to_5_percent(value: float) -> int:
            """Round to nearest 5% for memory (e.g., 11.7% -> 10%, 12.3% -> 10%)"""
            return int(round(value / 5.0) * 5)
        
        key_data = {
            'deployment': f"{namespace}/{deployment_name}",
            'replicas': current_replicas,
            # Round CPU to nearest 25% (170-180% all become 175%)
            'cpu': round_to_25(current_metrics.get('cpu_usage', 0)),
            # Round memory to nearest 5% (more stable for lower values)
            'memory': round_to_5_percent(current_metrics.get('memory_usage', 0)),
            # Round forecast peaks similarly
            'cpu_peak': round_to_25(forecast.get('cpu', {}).get('peak', 0)),
            'memory_peak': round_to_5_percent(forecast.get('memory', {}).get('peak', 0)),
            # Also include trend to differentiate between increasing/decreasing patterns
            'cpu_trend': forecast.get('cpu', {}).get('trend', 'unknown'),
            'memory_trend': forecast.get('memory', {}).get('trend', 'unknown'),
            # Include state management in cache key to ensure cache invalidation when state changes
            'state_management': state_management or 'unknown'
        }
        key_str = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.md5(key_str.encode()).hexdigest()
        logger.debug(f"Cache key for {deployment_name}: cpu={key_data['cpu']}%, memory={key_data['memory']}%, key={cache_key[:8]}")
        return cache_key
    
    def analyze_scaling_decision(self, 
                                 deployment_name: str,
                                 namespace: str,
                                 current_metrics: Dict[str, Any],
                                 forecast: Dict[str, Any],
                                 hpa_status: Optional[Dict[str, Any]] = None,
                                 vpa_status: Optional[Dict[str, Any]] = None,
                                 current_resources: Optional[Dict[str, Any]] = None,
                                 historical_patterns: Optional[List[Dict[str, Any]]] = None,
                                 current_replicas: int = 1,
                                 min_replicas: int = 1,
                                 max_replicas: int = 10,
                                 use_cache: bool = True,
                                 hpa_manager=None) -> Dict[str, Any]:
        """
        Use LLM to analyze and recommend scaling decisions
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            current_metrics: Current CPU/Memory usage metrics
            forecast: Predictive forecast data
            hpa_status: Current HPA status (optional)
            historical_patterns: Historical scaling patterns (optional)
            current_replicas: Current number of replicas
            min_replicas: Minimum allowed replicas
            max_replicas: Maximum allowed replicas
        
        Returns:
            Dict with LLM recommendation including action, reasoning, and confidence
        """
        if not self.client:
            return {
                'success': False,
                'error': 'LLM advisor not available (no API key)',
                'fallback': True
            }
        
        # CRITICAL: Detect state management FIRST (before cache check) to include in cache key
        # This ensures cache is invalidated when state management changes
        # Get hpa_manager from parameter or temporary reference
        effective_hpa_manager = hpa_manager or getattr(self, '_temp_hpa_manager', None)
        state_management_for_cache = None
        if effective_hpa_manager:
            try:
                temp_state_info = self._detect_state_management(deployment_name, namespace, effective_hpa_manager)
                if temp_state_info.get('detected') and temp_state_info.get('type'):
                    state_management_for_cache = f"{temp_state_info.get('type')}_{temp_state_info.get('source', 'unknown')}"
                    logger.warning(f"🔍🔍🔍 CACHE: Detected state_management for cache key: {state_management_for_cache}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to detect state management for cache key: {e}")
        else:
            logger.error(f"❌❌❌ CRITICAL: No hpa_manager available for state detection! hpa_manager param={hpa_manager}, _temp_hpa_manager={getattr(self, '_temp_hpa_manager', None)}")
        
        # Check cache and enforce minimum interval between LLM calls
        if use_cache:
            cache_key = self._get_cache_key(deployment_name, namespace, current_metrics, forecast, current_replicas, state_management_for_cache)
            deployment_key = f"{namespace}/{deployment_name}"
            now = datetime.now()
            
            # Check if we have a cached recommendation
            if cache_key in self.recommendation_cache:
                cached = self.recommendation_cache[cache_key]
                cache_age = (now - cached['timestamp']).total_seconds()
                if cache_age < self.cache_ttl:
                    logger.debug(f"✅ Using cached LLM recommendation (age: {cache_age:.1f}s, key: {cache_key[:8]})")
                    return {
                        'success': True,
                        'recommendation': cached['recommendation'],
                        'llm_model': cached.get('llm_model', self.model),
                        'timestamp': cached['timestamp'].isoformat(),
                        'cached': True
                    }
                else:
                    # Cache expired, remove it
                    logger.debug(f"⏰ Cache expired (age: {cache_age:.1f}s), removing")
                    del self.recommendation_cache[cache_key]
            
            # Enforce minimum interval between LLM calls for same deployment
            # BUT: Check if we have ANY cached result for this deployment first (even if cache key doesn't match exactly)
            # This allows us to return cached results even if metrics changed slightly
            deployment_has_recent_cache = False
            for cached_key, cached_data in self.recommendation_cache.items():
                if deployment_key in cached_key or cached_key.startswith(deployment_key.split('/')[0]):  # Check if cache key is for this deployment
                    cache_age = (now - cached_data['timestamp']).total_seconds()
                    if cache_age < self.cache_ttl:
                        deployment_has_recent_cache = True
                        logger.info(f"✅ Found recent cache for deployment {deployment_key} (age: {cache_age:.1f}s), will use it if needed")
                        break
            
            if deployment_key in self.last_llm_call_time:
                time_since_last_call = (now - self.last_llm_call_time[deployment_key]).total_seconds()
                if time_since_last_call < self.min_llm_interval:
                    # If we have a recent cache, return it instead of blocking
                    if deployment_has_recent_cache:
                        logger.info(f"⏸️ Rate-limited but found cached result - returning cached recommendation")
                        # Find and return the most recent cached result for this deployment
                        best_cache = None
                        best_age = float('inf')
                        for cached_key, cached_data in self.recommendation_cache.items():
                            if deployment_key in cached_key or cached_key.startswith(deployment_key.split('/')[0]):
                                cache_age = (now - cached_data['timestamp']).total_seconds()
                                if cache_age < best_age and cache_age < self.cache_ttl:
                                    best_age = cache_age
                                    best_cache = cached_data
                        if best_cache:
                            return {
                                'success': True,
                                'recommendation': best_cache['recommendation'],
                                'llm_model': best_cache.get('llm_model', self.model),
                                'timestamp': best_cache['timestamp'].isoformat(),
                                'cached': True,
                                'rate_limited': True  # Indicate this was rate-limited but we used cache
                            }
                    
                    logger.info(f"⏸️ Skipping LLM call - only {time_since_last_call:.1f}s since last call (min: {self.min_llm_interval}s)")
                    # Return a fallback recommendation indicating we're rate-limiting
                    return {
                        'success': False,
                        'error': f'LLM call rate-limited (last call {time_since_last_call:.1f}s ago, min interval {self.min_llm_interval}s)',
                        'fallback': True,
                        'rate_limited': True
                    }
        
        try:
            # Prepare context for LLM (need hpa_manager for state detection)
            # Use effective_hpa_manager we determined above (from parameter or temporary reference)
            # effective_hpa_manager was already set in the cache detection section above
            logger.warning(f"🔍🔍🔍 Using effective_hpa_manager for context: {effective_hpa_manager is not None}")
            
            context = self._prepare_context(
                deployment_name, namespace, current_metrics, forecast,
                hpa_status, vpa_status, current_resources, historical_patterns, 
                current_replicas, min_replicas, max_replicas, effective_hpa_manager
            )
            
            # Create system prompt
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with context
            user_prompt = self._create_user_prompt(context)
            
            # Call LLM (GPT OSS or Groq)
            provider_name = "GPT OSS" if self.provider == 'gpt_oss' else "Groq"
            print(f"🔍🔍🔍 CALLING {provider_name.upper()} API: model={self.model}, client={self.client is not None}")
            logger.warning(f"🔍🔍🔍 CALLING {provider_name.upper()} API: model={self.model}, client={self.client is not None}")
            print(f"🔍🔍🔍 SYSTEM PROMPT (first 200 chars): {system_prompt[:200]}...")
            logger.warning(f"🔍🔍🔍 SYSTEM PROMPT (first 200 chars): {system_prompt[:200]}...")
            print(f"🔍🔍🔍 USER PROMPT (first 200 chars): {user_prompt[:200]}...")
            logger.warning(f"🔍🔍🔍 USER PROMPT (first 200 chars): {user_prompt[:200]}...")
            
            # Make LLM API call with deterministic parameters
            try:
                import time
                start_time = time.time()
                
                print(f"🔍🔍🔍 Making {provider_name} API call to {self.model}...")
                logger.warning(f"🔍🔍🔍 Making {provider_name} API call to {self.model}...")
                print(f"🔍🔍🔍 GPT OSS Config: base_url={self.gpt_oss_api_base}, model={self.model}, timeout={self.client.timeout if self.client else 'N/A'}s")
                logger.warning(f"🔍🔍🔍 GPT OSS Config: base_url={self.gpt_oss_api_base}, model={self.model}, timeout={self.client.timeout if self.client else 'N/A'}s")
                
                # Test connection first (for GPT OSS) - check if server is responding
                if self.provider == 'gpt_oss':
                    try:
                        import requests
                        from urllib.parse import urlparse
                        # Parse base URL correctly (e.g., http://localhost:8001/v1 -> http://localhost:8001)
                        parsed = urlparse(self.gpt_oss_api_base)
                        base_url_clean = f"{parsed.scheme}://{parsed.netloc}"  # e.g., http://localhost:8001
                        # Test /v1/models endpoint
                        test_url = f"{base_url_clean}/v1/models"
                        test_response = requests.get(test_url, timeout=2)
                        print(f"🔍🔍🔍 GPT OSS connection test: {test_response.status_code} (URL: {test_url})")
                        logger.warning(f"🔍🔍🔍 GPT OSS connection test: {test_response.status_code} (URL: {test_url})")
                    except Exception as health_e:
                        print(f"⚠️⚠️⚠️ GPT OSS connection test failed (non-critical): {health_e}")
                        logger.warning(f"⚠️⚠️⚠️ GPT OSS connection test failed (non-critical): {health_e}")
                
                # Prepare deterministic sampling parameters
                # Use different max_tokens based on provider (GPT OSS has smaller context)
                max_tokens = self.max_tokens_gpt_oss if self.provider == 'gpt_oss' else self.max_tokens_groq
                
                print(f"🔍🔍🔍 API params: max_tokens={max_tokens}, temperature={self.temperature}, provider={self.provider}")
                logger.warning(f"🔍🔍🔍 API params: max_tokens={max_tokens}, temperature={self.temperature}, provider={self.provider}")
                
                api_params = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": self.temperature,  # Very low (0.1) for deterministic outputs
                    "max_tokens": max_tokens,  # Provider-specific token limit
                }
                
                # Add advanced sampling parameters for GPT OSS (more deterministic)
                # Note: llama.cpp OpenAI-compatible API only supports: temperature, top_p, max_tokens
                # It does NOT support: top_k, frequency_penalty, presence_penalty
                if self.provider == 'gpt_oss':
                    api_params.update({
                        "top_p": self.top_p,  # Nucleus sampling (supported by llama.cpp)
                    })
                    print(f"🔍🔍🔍 GPT OSS API params: {list(api_params.keys())}")
                    logger.warning(f"🔍🔍🔍 GPT OSS API params: {list(api_params.keys())}")
                elif self.provider == 'groq':
                    # Groq supports limited parameters (no top_k, frequency_penalty, or presence_penalty)
                    api_params.update({
                        "top_p": self.top_p,
                        # Note: Groq API doesn't support top_k, frequency_penalty, or presence_penalty
                    })
                
                print(f"🔍🔍🔍 Starting API call at {time.time():.2f}...")
                logger.warning(f"🔍🔍🔍 Starting API call at {time.time():.2f}...")
                
                response = self.client.chat.completions.create(**api_params)
                
                elapsed_time = time.time() - start_time
                print(f"🔍🔍🔍 API call completed in {elapsed_time:.2f}s")
                logger.warning(f"🔍🔍🔍 API call completed in {elapsed_time:.2f}s")
                
                response_text = response.choices[0].message.content if response.choices else ""
                
                print(f"✅✅✅ {provider_name.upper()} API CALL SUCCESSFUL! Model: {self.model}, Response length: {len(response_text)}")
                logger.warning(f"✅✅✅ {provider_name.upper()} API CALL SUCCESSFUL! Model: {self.model}, Response length: {len(response_text)}")
                print(f"✅✅✅ {provider_name.upper()} RESPONSE (first 500 chars): {response_text[:500]}...")
                logger.warning(f"✅✅✅ {provider_name.upper()} RESPONSE (first 500 chars): {response_text[:500]}...")
                
            except Exception as e:
                # Check if it's a timeout error (should fallback to Groq)
                is_timeout = 'timeout' in str(e).lower() or 'timed out' in str(e).lower() or 'APITimeoutError' in str(type(e).__name__)
                error_type = type(e).__name__
                
                logger.warning(f"⚠️  GPT OSS API call failed: {error_type}: {e}")
                print(f"⚠️  GPT OSS API call failed: {error_type}: {e}")
                
                # Fallback to Groq if GPT OSS fails (especially on timeout)
                # Check if Groq is available and API key is set
                groq_available = GROQ_AVAILABLE and self.groq_api_key and self.groq_api_key.strip()
                
                if self.provider == 'gpt_oss' and groq_available:
                    logger.warning(f"🔄 GPT OSS failed ({'timeout' if is_timeout else 'error'}), falling back to Groq...")
                    print(f"🔄🔄🔄 FALLING BACK TO GROQ due to GPT OSS {'timeout' if is_timeout else 'failure'}")
                    try:
                        # Initialize Groq client (Groq doesn't support timeout in constructor)
                        self.client = groq.Groq(api_key=self.groq_api_key)
                        self.provider = 'groq'
                        self.model = "llama-3.1-8b-instant"  # Fast Groq model
                        
                        logger.info(f"✅ Groq client initialized, making API call...")
                        print(f"✅ Groq client initialized, making API call...")
                        
                        # Use Groq-specific max_tokens
                        groq_max_tokens = self.max_tokens_groq
                        
                        # Make Groq API call with timeout handling
                        import signal
                        import threading
                        import queue as queue_module
                        
                        groq_result_queue = queue_module.Queue()
                        groq_error_queue = queue_module.Queue()
                        
                        def call_groq():
                            try:
                                # Groq API doesn't support top_k parameter
                                result = self.client.chat.completions.create(
                                    model=self.model,
                                    messages=[
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    temperature=self.temperature,
                                    max_tokens=groq_max_tokens,
                                    top_p=self.top_p  # Groq supports top_p but not top_k
                                )
                                groq_result_queue.put(result)
                            except Exception as e3:
                                groq_error_queue.put(e3)
                        
                        groq_thread = threading.Thread(target=call_groq, daemon=True)
                        groq_thread.start()
                        groq_thread.join(timeout=15.0)  # 15 second timeout for Groq (increased for reliability)
                        
                        if groq_thread.is_alive():
                            logger.error("❌ Groq API call timed out after 15s")
                            raise Exception("Groq API call timed out")
                        elif not groq_error_queue.empty():
                            error = groq_error_queue.get()
                            raise error
                        elif not groq_result_queue.empty():
                            groq_response = groq_result_queue.get()
                            response_text = groq_response.choices[0].message.content if groq_response.choices else ""
                            logger.warning(f"✅✅✅ Groq fallback succeeded! Response length: {len(response_text)}")
                            print(f"✅✅✅ Groq fallback succeeded! Response length: {len(response_text)}")
                        else:
                            raise Exception("Groq API call returned no result")
                            
                    except Exception as e2:
                        logger.error(f"❌❌❌ Groq fallback also failed: {e2}")
                        print(f"❌❌❌ Groq fallback also failed: {e2}")
                        raise Exception(f"All LLM providers failed. GPT OSS: {e}, Groq: {e2}")
                elif self.provider == 'gpt_oss' and not groq_available:
                    # GPT OSS failed but Groq not available
                    logger.error(f"❌ GPT OSS failed and Groq not available (GROQ_AVAILABLE={GROQ_AVAILABLE}, has_key={bool(self.groq_api_key)})")
                    print(f"❌ GPT OSS failed and Groq not available")
                    raise Exception(f"LLM API call failed: {e}. Groq fallback not available.")
                else:
                    # Try fallback model if using Groq
                    if self.provider == 'groq' and hasattr(self, 'fallback_model'):
                        logger.error(f"❌❌❌ Model {self.model} failed: {e}, trying fallback: {self.fallback_model}")
                        try:
                            response = self.client.chat.completions.create(
                                model=self.fallback_model,
                                messages=[
                                    {"role": "system", "content": system_prompt},
                                    {"role": "user", "content": user_prompt}
                                ],
                                temperature=self.temperature,
                                max_tokens=self.max_tokens
                            )
                            self.model = self.fallback_model  # Use fallback for future calls
                            response_text = response.choices[0].message.content if response.choices else ""
                            logger.warning(f"✅✅✅ Fallback model {self.fallback_model} succeeded")
                        except Exception as e2:
                            logger.error(f"❌❌❌ Both models failed: {e2}")
                            raise Exception(f"LLM API call failed: {e2}")
                    else:
                        raise Exception(f"LLM API call failed: {e}")
            
            # Parse LLM response
            llm_output = response_text if 'response_text' in locals() else response.choices[0].message.content
            logger.warning(f"🔍🔍🔍 LLM RAW RESPONSE (first 500 chars): {llm_output[:500]}...")
            provider_name = "GPT OSS" if self.provider == 'gpt_oss' else "Groq"
            logger.warning(f"🔍🔍🔍 LLM MODEL USED: {self.model} (Provider: {provider_name})")
            logger.warning(f"🔍🔍🔍 CONTEXT PASSED TO _parse_llm_response: state_info={context.get('state_management_info', {})}, state_note={context.get('state_management_note', '')[:200]}...")
            
            # Try to extract JSON from response (handle both JSON and text responses)
            # Pass min/max replicas for validation and context for state management check
            # Also pass deployment_name, namespace, and hpa_manager for fallback state detection in enforcement
            recommendation = self._parse_llm_response(llm_output, min_replicas, max_replicas, current_metrics, current_resources, context, deployment_name, namespace, effective_hpa_manager)
            logger.warning(f"🔍🔍🔍 PARSED RECOMMENDATION: scaling_type={recommendation.get('scaling_type') if recommendation else None}, target_replicas={recommendation.get('target_replicas') if recommendation else None}")
            
            now = datetime.now()
            result = {
                'success': True,
                'recommendation': recommendation,
                'llm_model': self.model,
                'timestamp': now.isoformat(),
                'cached': False
            }
            
            # Cache the result and track LLM call time
            if use_cache:
                cache_key = self._get_cache_key(deployment_name, namespace, current_metrics, forecast, current_replicas, state_management_for_cache)
                deployment_key = f"{namespace}/{deployment_name}"
                
                self.recommendation_cache[cache_key] = {
                    'recommendation': recommendation,
                    'llm_model': self.model,
                    'timestamp': now
                }
                self.last_llm_call_time[deployment_key] = now
                
                logger.info(f"✅ LLM recommendation cached (key: {cache_key[:8]}, deployment: {deployment_key})")
                
                # Clean old cache entries (keep only last 100)
                if len(self.recommendation_cache) > 100:
                    # Remove oldest entries
                    sorted_cache = sorted(self.recommendation_cache.items(), 
                                        key=lambda x: x[1]['timestamp'])
                    for key, _ in sorted_cache[:-100]:
                        del self.recommendation_cache[key]
                
                # Clean old LLM call times (keep only last 50)
                if len(self.last_llm_call_time) > 50:
                    sorted_times = sorted(self.last_llm_call_time.items(), 
                                        key=lambda x: x[1])
                    for key, _ in sorted_times[:-50]:
                        del self.last_llm_call_time[key]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in LLM autoscaling advisor: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'raw_response': llm_output if 'llm_output' in locals() else None
            }
    
    def _parse_llm_response(self, llm_output: str, min_replicas: int = 1, max_replicas: int = 10,
                           current_metrics: Optional[Dict[str, Any]] = None,
                           current_resources: Optional[Dict[str, Any]] = None,
                           context: Optional[Dict[str, Any]] = None,
                           deployment_name: Optional[str] = None,
                           namespace: Optional[str] = None,
                           hpa_manager=None) -> Dict[str, Any]:
        """Parse LLM response, extracting JSON if present and validate against constraints"""
        import re
        
        # Ensure current_resources is initialized (use parameter or get from context)
        if current_resources is None:
            current_resources = context.get('deployment', {}).get('current_resources', {}) if context else {}
        # If still None or empty, use defaults
        if not current_resources:
            current_resources = {}
        
        recommendation = None
        
        # Try direct JSON parse first
        try:
            recommendation = json.loads(llm_output)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        if not recommendation:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_output, re.DOTALL)
            if json_match:
                try:
                    recommendation = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass
        
        # Try to find JSON object in text
        if not recommendation:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', llm_output, re.DOTALL)
            if json_match:
                try:
                    recommendation = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        
        # Fallback: create recommendation from text analysis
        if not recommendation:
            logger.warning("Could not parse JSON from LLM response, creating fallback recommendation")
            recommendation = self._create_fallback_recommendation(llm_output)
        
        # Validate scaling_type consistency with reasoning
        # If reasoning mentions state inside pod but scaling_type is HPA, correct it to VPA
        if recommendation:
            reasoning_lower = recommendation.get('reasoning', '').lower()
            scaling_type = recommendation.get('scaling_type', 'hpa').lower()
            
            # Check if reasoning indicates stateful (state inside pod) but scaling_type is HPA
            state_inside_pod_indicators = [
                'state inside the pod',
                'state stored inside',
                'local files',
                'in-memory without externalization',
                'state is not externalized',
                'critical state inside'
            ]
            
            state_externalized_indicators = [
                'redis',
                'external database',
                'external db',
                'external cache',
                'shared storage',
                'state externalized',
                'externalized state'
            ]
            
            has_state_inside = any(indicator in reasoning_lower for indicator in state_inside_pod_indicators)
            has_state_externalized = any(indicator in reasoning_lower for indicator in state_externalized_indicators)
            
            # Check if no explicit state information was provided
            no_state_info_provided = False
            user_explicitly_stateless = False
            if context:
                state_note = context.get('state_management_note', '')
                state_info = context.get('state_management_info', {})
                
                # Check if user explicitly selected stateless (high confidence from annotation)
                if state_info.get('detected') and state_info.get('type') == 'stateless' and \
                   state_info.get('source') == 'annotation' and state_info.get('confidence') == 'high':
                    user_explicitly_stateless = True
                
                no_state_info_provided = not state_note or 'no explicit state' in state_note.lower() or 'not provided' in state_note.lower() or 'no state management information detected' in state_note.lower()
            
            # CRITICAL: If no state info provided and LLM chose HPA, correct to VPA for safety
            # BUT: Respect user's explicit stateless selection (don't override)
            if (has_state_inside and not has_state_externalized and scaling_type == 'hpa' and not user_explicitly_stateless) or \
               (no_state_info_provided and scaling_type == 'hpa' and not has_state_externalized and not user_explicitly_stateless):
                logger.warning(f"⚠️ LLM recommended HPA but state is uncertain or inside pod. Correcting to VPA.")
                recommendation['scaling_type'] = 'vpa'
                # Clear target_replicas for VPA
                recommendation['target_replicas'] = None
                
                # Calculate VPA targets based on current usage and resources
                cpu_usage = current_metrics.get('cpu_usage', 0) if current_metrics else 0
                memory_usage = current_metrics.get('memory_usage', 0) if current_metrics else 0
                
                # Get current resource requests to calculate new targets
                cpu_request_str = current_resources.get('cpu_request', '100m') if current_resources else '100m'
                memory_request_str = current_resources.get('memory_request', '128Mi') if current_resources else '128Mi'
                
                # Parse current CPU request (handle formats like "100m", "0.5", "1")
                try:
                    if cpu_request_str.endswith('m'):
                        cpu_request_m = int(cpu_request_str[:-1])
                    elif cpu_request_str.endswith('n'):
                        cpu_request_m = int(cpu_request_str[:-1]) / 1000000
                    else:
                        cpu_request_m = float(cpu_request_str) * 1000
                except:
                    cpu_request_m = 100  # Default 100m
                
                # Calculate target CPU based on usage (add 20% headroom)
                target_cpu_m = int(cpu_request_m * (1 + cpu_usage / 100) * 1.2) if cpu_usage > 0 else cpu_request_m
                target_cpu_m = max(100, min(target_cpu_m, 4000))  # Clamp between 100m and 4000m
                
                # Parse current memory request (handle formats like "128Mi", "256Mi", "1Gi")
                try:
                    if memory_request_str.endswith('Mi'):
                        memory_request_mi = int(memory_request_str[:-2])
                    elif memory_request_str.endswith('Gi'):
                        memory_request_mi = int(memory_request_str[:-2]) * 1024
                    elif memory_request_str.endswith('Ki'):
                        memory_request_mi = int(memory_request_str[:-2]) / 1024
                    else:
                        memory_request_mi = int(memory_request_str) if memory_request_str.isdigit() else 128
                except:
                    memory_request_mi = 128  # Default 128Mi
                
                # Calculate target memory based on usage (add 30% headroom)
                target_memory_mi = int(memory_request_mi * (1 + memory_usage / 100) * 1.3) if memory_usage > 0 else memory_request_mi
                target_memory_mi = max(128, min(target_memory_mi, 4096))  # Clamp between 128Mi and 4Gi
                
                # Set VPA targets
                if not recommendation.get('target_cpu'):
                    recommendation['target_cpu'] = f"{target_cpu_m}m"
                if not recommendation.get('target_memory'):
                    recommendation['target_memory'] = f"{target_memory_mi}Mi"
                
                recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                    f' [Corrected: Changed from HPA to VPA because state is stored inside the pod. Calculated VPA targets: CPU={recommendation["target_cpu"]}, Memory={recommendation["target_memory"]} based on current usage]')
        
        # CRITICAL: If user explicitly selected stateless or detection found stateless, enforce HPA
        # Also check the annotation directly if context doesn't have it (fallback)
        if recommendation:
            state_info = {}
            state_note = ''
            detected_type = None
            detected_source = None
            detected_confidence = None
            
            if context:
                state_info = context.get('state_management_info', {})
                detected_type = state_info.get('type')
                detected_source = state_info.get('source')
                detected_confidence = state_info.get('confidence')
                state_note = context.get('state_management_note', '')
            
            # FALLBACK: If context doesn't have state info but we have hpa_manager parameter, detect directly
            # Note: hpa_manager, deployment_name, and namespace are function parameters passed to _parse_llm_response
            if not state_info.get('detected') and hpa_manager and deployment_name and namespace:
                try:
                    logger.warning(f"🔍🔍🔍 ENFORCEMENT FALLBACK: Context missing state info, detecting directly...")
                    direct_state_info = self._detect_state_management(deployment_name, namespace, hpa_manager)
                    if direct_state_info.get('detected'):
                        state_info = direct_state_info
                        detected_type = state_info.get('type')
                        detected_source = state_info.get('source')
                        detected_confidence = state_info.get('confidence')
                        logger.warning(f"🔍🔍🔍 ENFORCEMENT FALLBACK: Direct detection found: type={detected_type}, source={detected_source}, confidence={detected_confidence}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to detect state in enforcement fallback: {e}")
            
            logger.warning(f"🔍🔍🔍 ENFORCEMENT DEBUG: State management check: type={detected_type}, source={detected_source}, confidence={detected_confidence}")
            logger.warning(f"🔍🔍🔍 ENFORCEMENT DEBUG: Full state_info: {state_info}")
            logger.warning(f"🔍🔍🔍 ENFORCEMENT DEBUG: State note (first 500 chars): {state_note[:500] if state_note else 'N/A'}...")
            logger.warning(f"🔍🔍🔍 ENFORCEMENT DEBUG: Current recommendation: scaling_type={recommendation.get('scaling_type')}, target_replicas={recommendation.get('target_replicas')}")
            
            # Check multiple indicators for stateless:
            # 1. Direct detection (annotation/label) - HIGHEST PRIORITY
            # 2. State note contains "stateless" and "HPA"
            # 3. State note contains "STATELESS" (uppercase)
            is_stateless_detected = detected_type == 'stateless' and detected_confidence in ['high', 'medium']
            is_stateless_in_note = state_note and 'stateless' in state_note.lower() and ('hpa' in state_note.lower() or 'horizontal' in state_note.lower())
            is_stateless_uppercase = state_note and 'STATELESS' in state_note
            
            logger.warning(f"🔍🔍🔍 ENFORCEMENT DEBUG: Stateless indicators: detected={is_stateless_detected}, in_note={is_stateless_in_note}, uppercase={is_stateless_uppercase}")
            
            # If ANY indicator suggests stateless, enforce HPA
            if is_stateless_detected or is_stateless_in_note or is_stateless_uppercase:
                scaling_type = recommendation.get('scaling_type', 'hpa').lower()
                logger.warning(f"✅✅✅ STATELESS DETECTED! (detected={is_stateless_detected}, in_note={is_stateless_in_note}, uppercase={is_stateless_uppercase}) LLM recommended: {scaling_type}")
                logger.warning(f"🔍🔍🔍 ENFORCEMENT: Checking if correction needed. scaling_type={scaling_type}, is_stateless_detected={is_stateless_detected}")
                
                if scaling_type in ['vpa', 'both']:
                    logger.warning(f"⚠️⚠️⚠️ FORCING HPA: User selected/detected STATELESS but LLM recommended {scaling_type.upper()}. Correcting to HPA.")
                    logger.warning(f"🔍🔍🔍 ENFORCEMENT: BEFORE correction - scaling_type={recommendation.get('scaling_type')}, target_cpu={recommendation.get('target_cpu')}, target_memory={recommendation.get('target_memory')}")
                    recommendation['scaling_type'] = 'hpa'
                    recommendation['target_cpu'] = None
                    recommendation['target_memory'] = None
                    
                    # Calculate HPA target based on current metrics and forecast
                    current_replicas = context.get('deployment', {}).get('current_replicas', 1)
                    effective_max_replicas = max_replicas  # From function parameter
                    
                    # Suggest scaling up if CPU/memory usage is high
                    cpu_usage = current_metrics.get('cpu_usage', 0) if current_metrics else 0
                    if cpu_usage > 70:
                        recommendation['target_replicas'] = min(current_replicas + 2, effective_max_replicas)
                    elif cpu_usage > 50:
                        recommendation['target_replicas'] = min(current_replicas + 1, effective_max_replicas)
                    else:
                        recommendation['target_replicas'] = min(current_replicas, effective_max_replicas)
                    
                    # Ensure target_replicas respects min/max
                    recommendation['target_replicas'] = max(min_replicas, min(recommendation['target_replicas'], effective_max_replicas))
                    
                    recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                        f' [FORCED CORRECTION: Changed from {scaling_type.upper()} to HPA because application is STATELESS. Stateless applications MUST scale horizontally, not vertically.]')
                else:
                    # LLM already recommended HPA, but check if target_replicas is missing or needs validation
                    # Also validate if LLM's reasoning contradicts the actual metrics
                    current_replicas = context.get('deployment', {}).get('current_replicas', 1) if context else 1
                    effective_max_replicas = max_replicas  # From function parameter
                    
                    # CRITICAL: Use forecast's current values (same as UI displays) instead of current_metrics
                    # The UI displays forecast.cpu.current and forecast.memory.current, so enforcement
                    # should use the same values to ensure consistency
                    forecast = context.get('forecast', {}) if context else {}
                    cpu_usage_from_forecast = forecast.get('cpu', {}).get('current', 0) if forecast else 0
                    memory_usage_from_forecast = forecast.get('memory', {}).get('current', 0) if forecast else 0
                    
                    # Fallback to current_metrics if forecast values are missing or zero
                    cpu_usage_from_metrics = current_metrics.get('cpu_usage', 0) if current_metrics else 0
                    memory_usage_from_metrics = current_metrics.get('memory_usage', 0) if current_metrics else 0
                    
                    # Use forecast values if available, otherwise fall back to metrics
                    cpu_usage = cpu_usage_from_forecast if cpu_usage_from_forecast > 0 else cpu_usage_from_metrics
                    memory_usage = memory_usage_from_forecast if memory_usage_from_forecast > 0 else memory_usage_from_metrics
                    
                    cpu_peak = forecast.get('cpu', {}).get('peak', cpu_usage) if forecast else cpu_usage
                    cpu_trend = forecast.get('cpu', {}).get('trend', 'stable') if forecast else 'stable'
                    
                    # Log which metrics source is being used
                    logger.warning(f"🔍🔍🔍 ENFORCEMENT METRICS SOURCE: forecast_cpu={cpu_usage_from_forecast:.1f}%, metrics_cpu={cpu_usage_from_metrics:.1f}%, using={cpu_usage:.1f}%")
                    logger.warning(f"🔍🔍🔍 ENFORCEMENT METRICS SOURCE: forecast_memory={memory_usage_from_forecast:.1f}%, metrics_memory={memory_usage_from_metrics:.1f}%, using={memory_usage:.1f}%")
                    
                    # Use the higher of current or peak for scaling decisions
                    effective_cpu = max(cpu_usage, cpu_peak)
                    
                    # Check if LLM recommendation contradicts actual metrics
                    llm_target = recommendation.get('target_replicas')
                    llm_action = recommendation.get('action', '').lower()
                    needs_recalculation = False
                    
                    # Validate: If CPU is high (>70%) but LLM recommends scale_down, recalculate
                    if effective_cpu > 70 and llm_action == 'scale_down':
                        logger.warning(f"⚠️⚠️⚠️ METRICS MISMATCH: CPU is {effective_cpu:.1f}% (HIGH) but LLM recommended scale_down. Recalculating...")
                        needs_recalculation = True
                    # Validate: If CPU is low (<30%) but LLM recommends scale_up, recalculate
                    elif effective_cpu < 30 and memory_usage < 30 and llm_action == 'scale_up' and cpu_trend != 'increasing':
                        logger.warning(f"⚠️⚠️⚠️ METRICS MISMATCH: CPU is {effective_cpu:.1f}% (LOW) but LLM recommended scale_up. Recalculating...")
                        needs_recalculation = True
                    # Validate: If target_replicas is None or missing, calculate it
                    elif llm_target is None:
                        logger.warning(f"⚠️⚠️⚠️ STATELESS HPA: LLM recommended HPA but target_replicas is None. Calculating target...")
                        needs_recalculation = True
                    
                    if needs_recalculation or llm_target is None:
                        # Log metrics for debugging
                        logger.warning(f"🔍🔍🔍 ENFORCEMENT METRICS: cpu_usage={cpu_usage:.1f}%, memory_usage={memory_usage:.1f}%, cpu_peak={cpu_peak:.1f}%, effective_cpu={effective_cpu:.1f}%, cpu_trend={cpu_trend}, current_replicas={current_replicas}")
                        
                        # Determine target based on current usage, forecast peak, and trend
                        if effective_cpu > 70 or memory_usage > 70:
                            # High usage - scale up
                            recommendation['target_replicas'] = min(current_replicas + 2, effective_max_replicas)
                            recommendation['action'] = 'scale_up'
                            logger.warning(f"🔍🔍🔍 ENFORCEMENT: High usage detected (CPU: {effective_cpu:.1f}%, Memory: {memory_usage:.1f}%) - scaling UP")
                        elif effective_cpu > 50 or memory_usage > 50:
                            # Moderate usage - scale up slightly
                            recommendation['target_replicas'] = min(current_replicas + 1, effective_max_replicas)
                            recommendation['action'] = 'scale_up'
                            logger.warning(f"🔍🔍🔍 ENFORCEMENT: Moderate usage detected (CPU: {effective_cpu:.1f}%, Memory: {memory_usage:.1f}%) - scaling UP slightly")
                        elif cpu_usage < 30 and memory_usage < 30 and cpu_trend != 'increasing':
                            # Low usage and not increasing - can scale down
                            recommendation['target_replicas'] = max(current_replicas - 1, min_replicas)
                            recommendation['action'] = 'scale_down'
                            logger.warning(f"🔍🔍🔍 ENFORCEMENT: Low usage detected (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%) - scaling DOWN")
                        else:
                            # Maintain current replicas (moderate usage or uncertain)
                            recommendation['target_replicas'] = current_replicas
                            recommendation['action'] = 'maintain'
                            logger.warning(f"🔍🔍🔍 ENFORCEMENT: Moderate/uncertain usage (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%) - MAINTAINING")
                        
                        # Ensure target_replicas respects min/max
                        recommendation['target_replicas'] = max(min_replicas, min(recommendation['target_replicas'], effective_max_replicas))
                        
                        # Update action based on final target_replicas
                        if recommendation['target_replicas'] > current_replicas:
                            recommendation['action'] = 'scale_up'
                        elif recommendation['target_replicas'] < current_replicas:
                            recommendation['action'] = 'scale_down'
                        else:
                            recommendation['action'] = 'maintain'
                        
                        logger.warning(f"✅✅✅ STATELESS HPA: Set target_replicas={recommendation['target_replicas']} (current={current_replicas}, min={min_replicas}, max={max_replicas}), action={recommendation['action']}")
                        # Use the same metrics that the UI displays (from forecast)
                        enforcement_cpu = cpu_usage  # Already from forecast.current
                        enforcement_memory = memory_usage  # Already from forecast.current
                        logger.warning(f"🔍🔍🔍 ENFORCEMENT MESSAGE METRICS: cpu={enforcement_cpu:.1f}%, memory={enforcement_memory:.1f}% (from forecast.current, same as UI)")
                        recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                            f' [ENFORCEMENT: Set target_replicas={recommendation["target_replicas"]} ({recommendation["action"]}) for stateless HPA scaling based on current metrics (CPU: {enforcement_cpu:.1f}%, Memory: {enforcement_memory:.1f}%)]')
                    else:
                        logger.info(f"✅ Stateless detected and LLM already recommended HPA with target_replicas={recommendation.get('target_replicas')} - no correction needed")
        
        # CRITICAL: If VPA is recommended but target_cpu/target_memory are missing, calculate them
        if recommendation and recommendation.get('scaling_type', '').lower() == 'vpa':
            if not recommendation.get('target_cpu') or not recommendation.get('target_memory'):
                logger.warning(f"⚠️⚠️⚠️ STATEFUL VPA: LLM recommended VPA but target_cpu/target_memory are missing. Calculating targets...")
                
                # Get current resources and metrics
                current_replicas = context.get('deployment', {}).get('current_replicas', 1) if context else 1
                cpu_usage = current_metrics.get('cpu_usage', 0) if current_metrics else 0
                memory_usage = current_metrics.get('memory_usage', 0) if current_metrics else 0
                # Use current_resources from parameter or context (already initialized at function start)
                if not current_resources and context:
                    current_resources = context.get('deployment', {}).get('current_resources', {})
                if not current_resources:
                    current_resources = {}
                
                # Get current resource requests to calculate new targets
                cpu_request_str = current_resources.get('cpu_request', '100m') if current_resources else '100m'
                memory_request_str = current_resources.get('memory_request', '128Mi') if current_resources else '128Mi'
                
                # Parse current CPU request
                try:
                    if cpu_request_str.endswith('m'):
                        cpu_request_m = int(cpu_request_str[:-1])
                    elif cpu_request_str.endswith('n'):
                        cpu_request_m = int(cpu_request_str[:-1]) / 1000000
                    else:
                        cpu_request_m = float(cpu_request_str) * 1000
                except:
                    cpu_request_m = 100  # Default 100m
                
                # Calculate target CPU based on usage (add 20% headroom, but ensure reasonable bounds)
                if cpu_usage > 70:
                    target_cpu_m = int(cpu_request_m * 1.5)  # Scale up 50% if high usage
                elif cpu_usage > 50:
                    target_cpu_m = int(cpu_request_m * 1.3)  # Scale up 30% if moderate usage
                elif cpu_usage < 30:
                    target_cpu_m = int(cpu_request_m * 0.8)  # Scale down 20% if low usage
                else:
                    target_cpu_m = int(cpu_request_m * 1.1)  # Slight increase for safety
                
                target_cpu_m = max(100, min(target_cpu_m, 4000))  # Clamp between 100m and 4000m
                
                # Parse current memory request
                try:
                    if memory_request_str.endswith('Mi'):
                        memory_request_mi = int(memory_request_str[:-2])
                    elif memory_request_str.endswith('Gi'):
                        memory_request_mi = int(memory_request_str[:-2]) * 1024
                    elif memory_request_str.endswith('Ki'):
                        memory_request_mi = int(memory_request_str[:-2]) / 1024
                    else:
                        memory_request_mi = int(memory_request_str) if memory_request_str.isdigit() else 128
                except:
                    memory_request_mi = 128  # Default 128Mi
                
                # Calculate target memory based on usage (add 30% headroom)
                if memory_usage > 70:
                    target_memory_mi = int(memory_request_mi * 1.5)  # Scale up 50% if high usage
                elif memory_usage > 50:
                    target_memory_mi = int(memory_request_mi * 1.3)  # Scale up 30% if moderate usage
                elif memory_usage < 30:
                    target_memory_mi = int(memory_request_mi * 0.9)  # Scale down 10% if low usage
                else:
                    target_memory_mi = int(memory_request_mi * 1.2)  # Slight increase for safety
                
                target_memory_mi = max(128, min(target_memory_mi, 4096))  # Clamp between 128Mi and 4Gi
                
                # Set VPA targets
                if not recommendation.get('target_cpu'):
                    recommendation['target_cpu'] = f"{target_cpu_m}m"
                if not recommendation.get('target_memory'):
                    recommendation['target_memory'] = f"{target_memory_mi}Mi"
                
                # Update action based on resource changes
                if target_cpu_m > cpu_request_m or target_memory_mi > memory_request_mi:
                    recommendation['action'] = 'scale_up'
                elif target_cpu_m < cpu_request_m or target_memory_mi < memory_request_mi:
                    recommendation['action'] = 'scale_down'
                else:
                    recommendation['action'] = 'maintain'
                
                logger.warning(f"✅✅✅ STATEFUL VPA: Set target_cpu={recommendation['target_cpu']}, target_memory={recommendation['target_memory']} (current CPU: {cpu_request_str}, Memory: {memory_request_str}), action={recommendation['action']}")
                recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                    f' [ENFORCEMENT: Set target_cpu={recommendation["target_cpu"]}, target_memory={recommendation["target_memory"]} for stateful VPA scaling based on current metrics (CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}%)]')
        
        # Validate and enforce min/max replica constraints (only for HPA)
        logger.warning(f"🔍🔍🔍 VALIDATION DEBUG: recommendation={recommendation is not None}, scaling_type={recommendation.get('scaling_type') if recommendation else None}, target_replicas={recommendation.get('target_replicas') if recommendation else None}, min_replicas={min_replicas}, max_replicas={max_replicas}")
        if recommendation and recommendation.get('scaling_type', 'hpa').lower() in ['hpa', 'both']:
            if 'target_replicas' in recommendation and recommendation['target_replicas'] is not None:
                target = recommendation['target_replicas']
                logger.warning(f"🔍🔍🔍 VALIDATION: Checking target={target} against min={min_replicas}, max={max_replicas}")
                if target > max_replicas:
                    logger.error(f"❌❌❌ CRITICAL: LLM recommended {target} replicas but max is {max_replicas}. Capping to {max_replicas}.")
                    recommendation['target_replicas'] = max_replicas
                    recommendation['action'] = 'at_max'
                    recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                        f" [CRITICAL: Original recommendation was {target} replicas, but capped to max_replicas={max_replicas}]")
                    logger.warning(f"🔍🔍🔍 VALIDATION: After capping, target_replicas={recommendation['target_replicas']}")
                elif target < min_replicas:
                    logger.error(f"❌❌❌ CRITICAL: LLM recommended {target} replicas but min is {min_replicas}. Setting to {min_replicas}.")
                    recommendation['target_replicas'] = min_replicas
                    recommendation['action'] = 'maintain'
                    recommendation['reasoning'] = (recommendation.get('reasoning', '') + 
                        f" [CRITICAL: Original recommendation was {target} replicas, but set to min_replicas={min_replicas}]")
                    logger.warning(f"🔍🔍🔍 VALIDATION: After setting min, target_replicas={recommendation['target_replicas']}")
                else:
                    logger.warning(f"✅ VALIDATION: target_replicas={target} is within range [{min_replicas}, {max_replicas}]")
            else:
                logger.warning(f"⚠️ VALIDATION: No target_replicas in recommendation or it's None")
        else:
            logger.warning(f"⚠️ VALIDATION: Skipping validation - scaling_type={recommendation.get('scaling_type') if recommendation else None} is not HPA/both")

        # MCDA Validation: Cross-check LLM decision against formal multi-criteria optimization
        if (self.mcda_optimizer and recommendation and
                recommendation.get('scaling_type', 'hpa').lower() in ['hpa', 'both'] and
                recommendation.get('target_replicas') is not None):
            try:
                llm_target = recommendation['target_replicas']
                llm_action = recommendation.get('action', 'maintain')
                current_reps = context.get('deployment', {}).get('current_replicas', 1) if context else 1

                # Build metrics and forecast for MCDA
                forecast_data = context.get('forecast', {}) if context else {}
                mcda_metrics = {
                    'cpu_percent': current_metrics.get('cpu_usage', 0) if current_metrics else 0,
                    'memory_percent': current_metrics.get('memory_usage', 0) if current_metrics else 0
                }
                mcda_forecast = {
                    'predicted_cpu': forecast_data.get('cpu', {}).get('predictions', [mcda_metrics['cpu_percent']] * 6),
                    'predicted_memory': forecast_data.get('memory', {}).get('predictions', [mcda_metrics['memory_percent']] * 6),
                    'cpu_trend': forecast_data.get('cpu', {}).get('trend', 'stable')
                }

                validation = self.mcda_optimizer.validate_llm_decision(
                    llm_action=llm_action,
                    llm_target=llm_target,
                    current_replicas=current_reps,
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    metrics=mcda_metrics,
                    forecast=mcda_forecast
                )

                # Attach MCDA validation data to recommendation
                recommendation['mcda_validation'] = {
                    'agreement': validation['agreement'],
                    'llm_score': validation['llm_score'],
                    'mcda_score': validation['mcda_score'],
                    'mcda_target': validation['mcda_target'],
                    'score_difference': validation['score_difference'],
                    'dominance_margin': validation['dominance_margin'],
                    'criteria_weights': validation['criteria_weights'],
                    'should_override': validation['should_override'],
                    'validation_note': validation['validation_note']
                }

                if validation['should_override']:
                    old_target = recommendation['target_replicas']
                    old_action = recommendation['action']
                    recommendation['target_replicas'] = max(min_replicas, min(validation['mcda_target'], max_replicas))

                    if recommendation['target_replicas'] > current_reps:
                        recommendation['action'] = 'scale_up'
                    elif recommendation['target_replicas'] < current_reps:
                        recommendation['action'] = 'scale_down'
                    else:
                        recommendation['action'] = 'maintain'

                    recommendation['reasoning'] = (recommendation.get('reasoning', '') +
                        f' [MCDA OVERRIDE: Changed from {old_action}→{old_target} to '
                        f'{recommendation["action"]}→{recommendation["target_replicas"]} '
                        f'(MCDA score={validation["mcda_score"]:.4f} vs LLM score={validation["llm_score"]:.4f}, '
                        f'agreement={validation["agreement"]})]')

                    logger.warning(f"📊 MCDA OVERRIDE: {old_action}→{old_target} changed to "
                                   f"{recommendation['action']}→{recommendation['target_replicas']} "
                                   f"(score gap={validation['score_difference']:.4f})")
                else:
                    logger.warning(f"📊 MCDA VALIDATION: LLM decision confirmed "
                                   f"(agreement={validation['agreement']}, "
                                   f"score_gap={validation['score_difference']:.4f})")

            except Exception as e:
                logger.warning(f"⚠️ MCDA validation failed (non-fatal): {e}")

        return recommendation
    
    def _create_fallback_recommendation(self, llm_output: str) -> Dict[str, Any]:
        """Create a fallback recommendation from text response"""
        import re
        
        # Try to extract action and replicas from text
        action_match = re.search(r'(scale_up|scale_down|maintain|at_max)', llm_output.lower())
        action = action_match.group(1) if action_match else 'maintain'
        
        replica_match = re.search(r'(\d+)\s*replicas?', llm_output.lower())
        target_replicas = int(replica_match.group(1)) if replica_match else 0
        
        return {
            'action': action,
            'target_replicas': target_replicas,
            'confidence': 0.5,
            'reasoning': llm_output[:500],  # First 500 chars
            'factors_considered': [],
            'risk_assessment': 'medium',
            'cost_impact': 'medium',
            'performance_impact': 'neutral',
            'recommended_timing': 'immediate'
        }
    
    def _detect_state_management(self, deployment_name: str, namespace: str, 
                                 hpa_manager) -> Dict[str, Any]:
        """
        Detect state management information from multiple sources:
        1. Deployment annotations
        2. Environment variables
        3. Service dependencies
        4. Volume mounts (persistent storage)
        5. Deployment labels
        """
        state_info = {
            'detected': False,
            'source': None,
            'type': None,  # 'stateless', 'stateful', 'unknown'
            'details': [],
            'confidence': 'low'
        }
        
        try:
            # 1. Get deployment details
            deployment_result = hpa_manager._execute_kubectl(
                f"get deployment {deployment_name} -n {namespace} -o json"
            )
            
            if not deployment_result.get('success'):
                return state_info
            
            deployment = deployment_result.get('result', {})
            if not deployment:
                return state_info
            
            metadata = deployment.get('metadata', {})
            annotations = metadata.get('annotations', {})
            labels = metadata.get('labels', {})
            spec = deployment.get('spec', {})
            template = spec.get('template', {})
            pod_spec = template.get('spec', {})
            containers = pod_spec.get('containers', []) if pod_spec else []
            # Safety check: ensure containers is a list, not None
            if containers is None:
                containers = []
            
            # 2. Check deployment annotations (user-provided or explicit) - HIGHEST PRIORITY
            state_annotation = annotations.get('ai4k8s.io/state-management')
            logger.warning(f"🔍🔍🔍 STATE DETECTION DEBUG: Checking annotation 'ai4k8s.io/state-management': {state_annotation}")
            logger.warning(f"🔍🔍🔍 STATE DETECTION DEBUG: All annotations keys: {list(annotations.keys())}")
            logger.warning(f"🔍🔍🔍 STATE DETECTION DEBUG: Full annotations dict: {annotations}")
            if state_annotation:
                state_info['detected'] = True
                state_info['source'] = 'annotation'
                state_info['confidence'] = 'high'  # User selection is always high confidence
                annotation_lower = state_annotation.lower().strip()
                logger.info(f"✅ Found state-management annotation: '{state_annotation}' (lowercase: '{annotation_lower}')")
                if annotation_lower in ['stateless', 'external', 'redis', 'database', 'db']:
                    state_info['type'] = 'stateless'
                    state_info['details'].append(f"User annotation indicates STATELESS: {state_annotation}")
                    logger.info(f"✅✅✅ Detected STATELESS from user annotation: {state_annotation}")
                elif annotation_lower in ['stateful', 'internal', 'local']:
                    state_info['type'] = 'stateful'
                    state_info['details'].append(f"User annotation indicates STATEFUL: {state_annotation}")
                    logger.info(f"✅✅✅ Detected STATEFUL from user annotation: {state_annotation}")
                else:
                    # Unknown value, but still detected
                    state_info['type'] = 'unknown'
                    state_info['details'].append(f"Annotation value: {state_annotation}")
                    logger.warning(f"⚠️ Unknown annotation value: {state_annotation}")
                return state_info
            else:
                logger.debug(f"ℹ️ No 'ai4k8s.io/state-management' annotation found")
            
            # 3. Check environment variables for external state indicators
            env_indicators = {
                'stateless': ['REDIS', 'DATABASE', 'DB_', 'POSTGRES', 'MYSQL', 'MONGO', 
                             'CASSANDRA', 'ELASTICSEARCH', 'EXTERNAL', 'CACHE_'],
                'stateful': ['LOCAL_STORAGE', 'PERSISTENT', 'VOLUME', 'STATE_DIR']
            }
            
            for container in containers:
                env = container.get('env', [])
                env_str = ' '.join([str(e) for e in env]).upper()
                
                # Check for stateless indicators
                for indicator in env_indicators['stateless']:
                    if indicator in env_str:
                        state_info['detected'] = True
                        state_info['source'] = 'environment_variables'
                        state_info['type'] = 'stateless'
                        state_info['confidence'] = 'medium'
                        state_info['details'].append(f"Environment variable indicates external state: {indicator}")
                        break
                
                # Check for stateful indicators
                if not state_info['detected']:
                    for indicator in env_indicators['stateful']:
                        if indicator in env_str:
                            state_info['detected'] = True
                            state_info['source'] = 'environment_variables'
                            state_info['type'] = 'stateful'
                            state_info['confidence'] = 'medium'
                            state_info['details'].append(f"Environment variable indicates internal state: {indicator}")
                            break
                
                if state_info['detected']:
                    break
            
            # 4. Check volume mounts for persistent storage
            volumes = pod_spec.get('volumes', []) if pod_spec else []
            # Safety check: ensure volumes is a list, not None
            if volumes is None:
                volumes = []
            volume_mounts = []
            for container in containers:
                volume_mounts_list = container.get('volumeMounts', [])
                if volume_mounts_list:  # Only extend if not None/empty
                    volume_mounts.extend(volume_mounts_list if isinstance(volume_mounts_list, list) else [])
            
            persistent_volume_types = ['persistentVolumeClaim', 'hostPath', 'local']
            for volume in volumes:
                volume_type = None
                if 'persistentVolumeClaim' in volume:
                    volume_type = 'persistentVolumeClaim'
                elif 'hostPath' in volume:
                    volume_type = 'hostPath'
                elif 'emptyDir' in volume:
                    # emptyDir is ephemeral, not persistent
                    continue
                
                if volume_type in persistent_volume_types:
                    state_info['detected'] = True
                    state_info['source'] = 'volume_mounts'
                    state_info['type'] = 'stateful'
                    state_info['confidence'] = 'high'
                    state_info['details'].append(f"Persistent volume detected: {volume_type}")
                    break
            
            # 5. Check service dependencies (look for Redis, DB services in namespace)
            if not state_info['detected']:
                services_result = hpa_manager._execute_kubectl(
                    f"get services -n {namespace} -o json"
                )
                
                if services_result.get('success'):
                    result_data = services_result.get('result', {})
                    services = result_data.get('items', []) if result_data else []
                    # Safety check: ensure services is a list, not None
                    if services is None:
                        services = []
                    service_names = [svc.get('metadata', {}).get('name', '').lower() for svc in services]
                    
                    # Check for common external state services
                    external_state_services = ['redis', 'postgres', 'mysql', 'mongo', 'cassandra', 
                                              'elasticsearch', 'database', 'db', 'cache']
                    
                    for service_name in service_names:
                        for indicator in external_state_services:
                            if indicator in service_name:
                                state_info['detected'] = True
                                state_info['source'] = 'service_dependencies'
                                state_info['type'] = 'stateless'
                                state_info['confidence'] = 'medium'
                                state_info['details'].append(f"External state service detected: {service_name}")
                                break
                        if state_info['detected']:
                            break
            
            # 6. Check deployment labels
            if not state_info['detected']:
                state_label = labels.get('ai4k8s.io/state-management')
                if state_label:
                    state_info['detected'] = True
                    state_info['source'] = 'label'
                    state_info['confidence'] = 'high'
                    if state_label.lower() in ['stateless', 'external']:
                        state_info['type'] = 'stateless'
                    elif state_label.lower() in ['stateful', 'internal']:
                        state_info['type'] = 'stateful'
                    state_info['details'].append(f"Label indicates: {state_label}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error detecting state management: {e}")
            # Return unknown state on error
            state_info['type'] = 'unknown'
        
        return state_info
    
    def _prepare_context(self, deployment_name: str, namespace: str,
                        current_metrics: Dict[str, Any], forecast: Dict[str, Any],
                        hpa_status: Optional[Dict[str, Any]],
                        vpa_status: Optional[Dict[str, Any]],
                        current_resources: Optional[Dict[str, Any]],
                        historical_patterns: Optional[List[Dict[str, Any]]],
                        current_replicas: int, min_replicas: int, max_replicas: int,
                        hpa_manager=None) -> Dict[str, Any]:
        """Prepare context dictionary for LLM"""
        context = {
            'deployment': {
                'name': deployment_name,
                'namespace': namespace,
                'current_replicas': current_replicas,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas
            },
            'current_metrics': {
                'cpu_usage_percent': current_metrics.get('cpu_usage', 0),
                'memory_usage_percent': current_metrics.get('memory_usage', 0),
                'pod_count': current_metrics.get('pod_count', 0),
                'running_pods': current_metrics.get('running_pod_count', 0)
            },
            'forecast': {
                'cpu': {
                    'current': forecast.get('cpu', {}).get('current', 0),
                    'peak': forecast.get('cpu', {}).get('peak', 0),
                    'trend': forecast.get('cpu', {}).get('trend', 'unknown'),
                    'predictions': forecast.get('cpu', {}).get('predictions', [])
                },
                'memory': {
                    'current': forecast.get('memory', {}).get('current', 0),
                    'peak': forecast.get('memory', {}).get('peak', 0),
                    'trend': forecast.get('memory', {}).get('trend', 'unknown'),
                    'predictions': forecast.get('memory', {}).get('predictions', [])
                }
            }
        }
        
        # Add current resource requests/limits if available
        if current_resources:
            context['current_resources'] = current_resources
        else:
            context['current_resources'] = {
                'cpu_request': 'N/A',
                'memory_request': 'N/A',
                'cpu_limit': 'N/A',
                'memory_limit': 'N/A'
            }
        
        # Detect state management from multiple sources
        state_info = {
            'detected': False,
            'type': 'unknown',
            'note': 'No explicit state management information available. DO NOT assume external state (Redis, DB) unless explicitly mentioned.'
        }
        
        if hpa_manager:
            state_info = self._detect_state_management(deployment_name, namespace, hpa_manager)
            logger.info(f"🔍 State detection result: detected={state_info.get('detected')}, type={state_info.get('type')}, source={state_info.get('source')}, confidence={state_info.get('confidence')}")
        else:
            logger.warning(f"⚠️ No hpa_manager provided, cannot detect state management")
        
        # Build state management note based on detection
        if state_info['detected']:
            # Ensure details is always a list (not None)
            details = state_info.get('details', [])
            if details is None:
                details = []
            
            if state_info['type'] == 'stateless':
                # Check if stateless was detected from actual evidence (env vars, services) or just annotation
                source = state_info.get('source', '')
                has_evidence = source in ['environment_variables', 'service_dependencies'] or any(
                    'Redis' in d or 'DATABASE' in d or 'DB' in d or 'external' in d.lower() 
                    for d in details if d  # Only iterate if d is not None/empty
                )
                
                details_str = ', '.join(str(d) for d in details if d) if details else 'No specific details'
                
                if has_evidence:
                    # Stateless detected from actual evidence (Redis, DB, etc.)
                    state_note = (
                        f"✅✅✅ CRITICAL: State Management Detected ({state_info['source']}, confidence: {state_info['confidence']}): "
                        f"Application is STATELESS - state is externalized (Redis, DB, external cache, etc.) as detected from deployment configuration. "
                        f"Details: {details_str}. "
                        f"**YOU MUST RECOMMEND HPA (horizontal scaling) - DO NOT recommend VPA for stateless applications.**"
                    )
                else:
                    # Stateless detected from annotation only (no evidence of Redis/DB)
                    state_note = (
                        f"✅✅✅ CRITICAL: State Management Detected ({state_info['source']}, confidence: {state_info['confidence']}): "
                        f"Application is marked as STATELESS (from user annotation or auto-detection). "
                        f"**DO NOT assume** Redis, DB, or external cache exists unless explicitly mentioned in deployment details. "
                        f"Details: {details_str}. "
                        f"**YOU MUST RECOMMEND HPA (horizontal scaling) - DO NOT recommend VPA for stateless applications.**"
                    )
            elif state_info['type'] == 'stateful':
                details_str = ', '.join(str(d) for d in details if d) if details else 'No specific details'
                state_note = (
                    f"✅✅✅ CRITICAL: State Management Detected ({state_info['source']}, confidence: {state_info['confidence']}): "
                    f"Application is STATEFUL - state is stored inside the pod. "
                    f"Details: {details_str}. "
                    f"**YOU MUST RECOMMEND VPA (vertical scaling) - DO NOT recommend HPA for stateful applications.**"
                )
            else:
                state_note = state_info['note']
        else:
            state_note = (
                "⚠️ No state management information detected from deployment analysis. "
                "Checked: annotations, environment variables, volume mounts, service dependencies, labels. "
                "When uncertain, prefer VPA (vertical scaling) as it's safer for applications that may store state inside pods."
            )
        
        context['state_management_note'] = state_note
        context['state_management_info'] = state_info
        
        if hpa_status:
            context['hpa'] = {
                'exists': True,
                'current_replicas': hpa_status.get('current_replicas', current_replicas),
                'desired_replicas': hpa_status.get('desired_replicas', current_replicas),
                'target_cpu': hpa_status.get('target_cpu', 70),
                'target_memory': hpa_status.get('target_memory', 80),
                'scaling_status': hpa_status.get('scaling_status', 'stable')
            }
        else:
            context['hpa'] = {'exists': False}
        
        if vpa_status:
            context['vpa'] = {
                'exists': True,
                'update_mode': vpa_status.get('update_mode', 'Auto'),
                'recommendations': vpa_status.get('recommendations', [])
            }
        else:
            context['vpa'] = {'exists': False}
        
        if historical_patterns:
            context['historical_patterns'] = historical_patterns[-10:]  # Last 10 patterns (Qwen can handle full context for accuracy)
        
        return context
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for LLM"""
        return """You are an expert Kubernetes autoscaling advisor with deep knowledge of:
- Resource optimization and cost management
- Performance requirements and SLA considerations
- Scaling best practices and anti-patterns
- Predictive analysis and trend interpretation
- Horizontal Pod Autoscaling (HPA) vs Vertical Pod Autoscaling (VPA)

Your role is to analyze deployment metrics, forecasts, and patterns to make intelligent scaling recommendations.

**IMPORTANT: You must decide between TWO scaling strategies:**

1. **HORIZONTAL SCALING (HPA)**: Scale by adjusting the NUMBER of replicas (pods)
   - Use when: Load can be distributed across multiple pods, need high availability
   - **CRITICAL RULE**: Applications that externalize their state (use Redis, external databases, external cache, shared storage) should be treated as STATELESS and prefer HPA
   - Example: 3 pods → 5 pods (same resources per pod)
   - Pros: Better fault tolerance, load distribution, can handle traffic spikes
   - Cons: More pods = more overhead, may hit node limits

2. **VERTICAL SCALING (VPA)**: Scale by adjusting RESOURCE requests/limits per pod
   - Use when: Application keeps critical state INSIDE the pod (not externalized), cannot scale horizontally, single-pod bottleneck
   - **CRITICAL RULE**: Only prefer VPA if state is stored INSIDE the pod. If state is externalized (Redis, DB, external cache), treat as stateless and prefer HPA instead
   - Example: CPU 100m → 200m, Memory 128Mi → 256Mi (same number of pods)
   - Pros: Better resource utilization, fewer pods, simpler architecture
   - Cons: Pod restart required, single point of failure, limited by node capacity

**IMPORTANT STATE MANAGEMENT RULES:**
- **NEVER assume "stateful = only VPA"**
- **ALWAYS check if state is externalized before ruling out HPA**
- If application uses Redis, external databases, external cache, or shared storage → treat as STATELESS → prefer HPA
- If application keeps critical state inside the pod (local files, in-memory state without externalization) → prefer VPA

Consider these weighted criteria (Multi-Criteria Decision Analysis):
1. **Performance** (weight: 0.30): Ensure adequate resources to meet performance requirements. Target CPU utilization: 60-80% sweet spot.
2. **Stability** (weight: 0.25): Avoid rapid scaling changes that could cause instability. Penalize large replica changes.
3. **Cost** (weight: 0.20): Minimize resource usage while maintaining performance. Fewer replicas = lower cost.
4. **Forecast Alignment** (weight: 0.15): Use forecast data to proactively scale before demand arrives. Ensure predicted peak stays under 80%.
5. **Response Time** (weight: 0.10): Minimize latency impact. Higher CPU utilization increases response time.
6. **Constraints**: **CRITICAL - You MUST respect min/max replica limits. target_replicas MUST be between min_replicas and max_replicas (inclusive). NEVER recommend more than max_replicas or less than min_replicas.**
7. **Scaling Type**: Choose HPA (horizontal) or VPA (vertical) based on application characteristics

**IMPORTANT CONSTRAINTS:**
- If you recommend HPA scaling, target_replicas MUST be >= min_replicas AND <= max_replicas
- If current replicas is already at max_replicas and you need more, recommend "at_max" action or suggest VPA instead
- If current replicas is already at min_replicas and you need less, recommend "maintain" action

Respond in JSON format with:
{
  "scaling_type": "hpa" | "vpa" | "both" | "maintain",
  "action": "scale_up" | "scale_down" | "maintain" | "at_max",
  "target_replicas": <number> (for HPA, MUST be between min_replicas and max_replicas, null if VPA),
  "target_cpu": "<value>" (for VPA, e.g., "200m", null if HPA),
  "target_memory": "<value>" (for VPA, e.g., "256Mi", null if HPA),
  "confidence": <0.0-1.0>,
  "reasoning": "<detailed explanation including why HPA vs VPA was chosen and why target_replicas respects min/max limits>",
  "criteria_scores": {
    "performance": <0.0-1.0 how well this decision optimizes performance>,
    "stability": <0.0-1.0 how stable this decision is>,
    "cost": <0.0-1.0 how cost-efficient this decision is>,
    "forecast_alignment": <0.0-1.0 how well it aligns with predicted demand>,
    "response_time": <0.0-1.0 how well it minimizes latency>
  },
  "factors_considered": ["factor1", "factor2", ...],
  "risk_assessment": "low" | "medium" | "high",
  "cost_impact": "low" | "medium" | "high",
  "performance_impact": "positive" | "neutral" | "negative",
  "recommended_timing": "immediate" | "gradual" | "scheduled"
}"""
    
    def _create_user_prompt(self, context: Dict[str, Any]) -> str:
        """Create user prompt with context"""
        prompt = f"""Analyze the following Kubernetes deployment autoscaling scenario and provide a recommendation:

**Deployment Information:**
- Name: {context['deployment']['name']}
- Namespace: {context['deployment']['namespace']}
- Current Replicas: {context['deployment']['current_replicas']}
- Min Replicas: {context['deployment']['min_replicas']}
- Max Replicas: {context['deployment']['max_replicas']}

**Current Resource Usage:**
- CPU: {context['current_metrics']['cpu_usage_percent']:.1f}%
- Memory: {context['current_metrics']['memory_usage_percent']:.1f}%
- Running Pods: {context['current_metrics']['running_pods']}/{context['current_metrics']['pod_count']}

**Forecast Data:**
- CPU Current: {context['forecast']['cpu']['current']:.1f}%, Peak: {context['forecast']['cpu']['peak']:.1f}%, Trend: {context['forecast']['cpu']['trend']}
- Memory Current: {context['forecast']['memory']['current']:.1f}%, Peak: {context['forecast']['memory']['peak']:.1f}%, Trend: {context['forecast']['memory']['trend']}
- CPU Predictions (next 3 hours): {context['forecast']['cpu']['predictions'][:6] if len(context['forecast']['cpu'].get('predictions', [])) > 6 else context['forecast']['cpu'].get('predictions', [])}
- Memory Predictions (next 3 hours): {context['forecast']['memory']['predictions'][:6] if len(context['forecast']['memory'].get('predictions', [])) > 6 else context['forecast']['memory'].get('predictions', [])}

**Current Resource Configuration:**
- CPU Request: {context['current_resources'].get('cpu_request', 'N/A')}
- CPU Limit: {context['current_resources'].get('cpu_limit', 'N/A')}
- Memory Request: {context['current_resources'].get('memory_request', 'N/A')}
- Memory Limit: {context['current_resources'].get('memory_limit', 'N/A')}

**State Management Information:**
{context.get('state_management_note', 'No explicit state management information available. DO NOT assume external state (Redis, DB) unless explicitly mentioned.')}

**State Decision:**
- STATELESS/externalized state → HPA
- STATEFUL/internal state → VPA
- Unknown → VPA (safer)

**HPA Status:**
"""
        
        if context['hpa']['exists']:
            prompt += f"""- HPA Active: Yes
- Current Replicas: {context['hpa']['current_replicas']}
- Desired Replicas: {context['hpa']['desired_replicas']}
- Target CPU: {context['hpa']['target_cpu']}%
- Target Memory: {context['hpa']['target_memory']}%
- Scaling Status: {context['hpa']['scaling_status']}
"""
        else:
            prompt += "- HPA Active: No (reactive horizontal scaling not configured)\n"
        
        prompt += "\n**VPA Status:**\n"
        if context['vpa']['exists']:
            prompt += f"""- VPA Active: Yes
- Update Mode: {context['vpa']['update_mode']}
- Recommendations: {json.dumps(context['vpa'].get('recommendations', []), indent=2)}
"""
        else:
            prompt += "- VPA Active: No (vertical scaling not configured)\n"
        
        if context.get('historical_patterns'):
            # Include last 3 historical patterns (truncate to reduce prompt size while maintaining accuracy)
            patterns_to_include = context['historical_patterns'][-3:] if len(context['historical_patterns']) > 3 else context['historical_patterns']
            prompt += f"\n**Historical Patterns (last 3):**\n{json.dumps(patterns_to_include, indent=1)}\n"
        
        prompt += """
**Analysis:**
Consider: resource pressure, forecast trends, cost, performance, stability. Choose HPA (replicas) or VPA (resources per pod) based on state management above.

**Rules:**
- HPA: stateless OR externalized state (Redis/DB/cache) → set scaling_type="hpa", target_replicas=<number>, target_cpu=null, target_memory=null
- VPA: internal state → set scaling_type="vpa", target_replicas=null, target_cpu="<value>", target_memory="<value>"
- target_replicas MUST be between min_replicas and max_replicas

**DO NOT scale down stateful applications (state inside pod) by reducing replicas - use VPA instead to adjust resources per pod.**

Provide your recommendation in the specified JSON format with scaling_type field."""
        
        return prompt
    
    def get_intelligent_recommendation(self, deployment_name: str, namespace: str,
                                      current_metrics: Dict[str, Any],
                                      forecast: Dict[str, Any],
                                      hpa_status: Optional[Dict[str, Any]] = None,
                                      vpa_status: Optional[Dict[str, Any]] = None,
                                      current_resources: Optional[Dict[str, Any]] = None,
                                      current_replicas: int = 1,
                                      min_replicas: int = 1,
                                      max_replicas: int = 10,
                                      hpa_manager=None) -> Dict[str, Any]:
        """
        Get intelligent LLM-based scaling recommendation
        
        This is a convenience method that wraps analyze_scaling_decision
        """
        return self.analyze_scaling_decision(
            deployment_name=deployment_name,
            namespace=namespace,
            current_metrics=current_metrics,
            forecast=forecast,
            hpa_status=hpa_status,
            vpa_status=vpa_status,
            current_resources=current_resources,
            historical_patterns=None,  # Can be added later
            current_replicas=current_replicas,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            hpa_manager=hpa_manager
        )
    
    def explain_scaling_decision(self, recommendation: Dict[str, Any]) -> str:
        """Generate human-readable explanation of scaling decision"""
        if not recommendation.get('success'):
            return f"❌ Error: {recommendation.get('error', 'Unknown error')}"
        
        rec = recommendation.get('recommendation', {})
        action = rec.get('action', 'unknown')
        target = rec.get('target_replicas', 0)
        reasoning = rec.get('reasoning', 'No reasoning provided')
        confidence = rec.get('confidence', 0)
        risk = rec.get('risk_assessment', 'unknown')
        
        action_emoji = {
            'scale_up': '⬆️',
            'scale_down': '⬇️',
            'maintain': '➡️',
            'at_max': '⚠️'
        }.get(action, '❓')
        
        # Build base explanation
        explanation = f"""{action_emoji} **LLM Recommendation: {action.replace('_', ' ').title()}**

**Target Replicas:** {target}
**Confidence:** {confidence:.0%}
**Risk Level:** {risk.upper()}

**Reasoning:**
{reasoning}

**Factors Considered:** {', '.join(rec.get('factors_considered', []))}
**Cost Impact:** {rec.get('cost_impact', 'unknown').upper()}
**Performance Impact:** {rec.get('performance_impact', 'unknown').upper()}
**Recommended Timing:** {rec.get('recommended_timing', 'unknown').title()}"""

        # Add MCDA validation info if available
        mcda = rec.get('mcda_validation')
        if mcda:
            explanation += f"""

**MCDA Validation:** {mcda.get('agreement', 'N/A').upper()}
**MCDA Score (LLM choice):** {mcda.get('llm_score', 0):.4f}
**MCDA Optimal Score:** {mcda.get('mcda_score', 0):.4f}
**Override Applied:** {'Yes' if mcda.get('should_override') else 'No'}"""

        # Add criteria scores if available
        criteria = rec.get('criteria_scores')
        if criteria:
            explanation += "\n\n**Criteria Scores:**"
            for k, v in criteria.items():
                if isinstance(v, (int, float)):
                    explanation += f"\n  - {k}: {v:.2f}"

        return explanation

