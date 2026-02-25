#!/usr/bin/env python3
"""
Central runtime configuration defaults for AutoSage services.
"""

import os

# MCP connectivity
MCP_SERVER_BASE_URL = os.getenv("MCP_SERVER_BASE_URL", "http://127.0.0.1:5002")
MCP_HTTP_ENDPOINT = os.getenv("MCP_HTTP_ENDPOINT", "/mcp")
MCP_MESSAGE_ENDPOINT = os.getenv("MCP_MESSAGE_ENDPOINT", "http://172.18.0.1:5002/message")

# Autoscaling cadence
PREDICTIVE_SCALING_INTERVAL_SECONDS = int(os.getenv("PREDICTIVE_SCALING_INTERVAL_SECONDS", "300"))

# Monitoring cadence
DEFAULT_MONITORING_INTERVAL_SECONDS = int(os.getenv("DEFAULT_MONITORING_INTERVAL_SECONDS", "300"))

# Async job cleanup
JOB_EXPIRY_SECONDS = int(os.getenv("JOB_EXPIRY_SECONDS", "300"))
