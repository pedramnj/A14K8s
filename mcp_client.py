#!/usr/bin/env python3
import asyncio
import json
import subprocess
import sys
import requests
from typing import Optional, Dict, Any
from contextlib import AsyncExitStack

class MCPKubernetesClient:
    def __init__(self):
        self.available_tools = {}
        self.server_url = "http://172.18.0.1:5002"
        self.endpoint = "/mcp"
        
    async def connect_to_server(self, server_script_path: str = None):
        try:
            # Test connection to HTTP MCP server
            response = requests.post(
                f"{self.server_url}{self.endpoint}",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data and 'tools' in data['result']:
                    self.available_tools = {tool['name']: tool for tool in data['result']['tools']}
                    print(f'✅ Connected to MCP server with {len(self.available_tools)} tools')
                    return True
                else:
                    print(f'⚠️  Invalid response format: {data}')
                    return False
            else:
                print(f'⚠️  HTTP {response.status_code}: {response.text}')
                return False
            
        except Exception as e:
            print(f'⚠️  Error connecting to MCP server: {e}')
            return False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not self.available_tools:
                return {'error': 'Not connected to MCP server'}
                
            # Call tool via HTTP
            response = requests.post(
                f"{self.server_url}{self.endpoint}",
                json={
                    "jsonrpc": "2.0", 
                    "id": 1, 
                    "method": "tools/call", 
                    "params": {
                        "name": tool_name,
                        "arguments": arguments
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'result' in data:
                    return {
                        'success': True,
                        'result': data['result']
                    }
                else:
                    return {'error': data.get('error', 'Unknown error')}
            else:
                return {'error': f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {'error': str(e)}
    
    async def cleanup(self):
        pass  # No cleanup needed for HTTP client

# Global MCP client instance
mcp_client = None

async def get_mcp_client():
    global mcp_client
    if mcp_client is None:
        mcp_client = MCPKubernetesClient()
        await mcp_client.connect_to_server()
    return mcp_client

async def call_mcp_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    client = await get_mcp_client()
    return await client.call_tool(tool_name, arguments)
